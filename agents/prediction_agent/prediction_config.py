"""Centralized configuration for the AquaMind AI Prediction Agent (training).

This module owns every tunable value and path used by the OFFLINE prediction
training pipeline: dataset locations, the prediction-task definitions (which
dataset, target and features to use), the training hyperparameters, and the
factory that builds the candidate regression algorithms to compare.

Nothing is hardcoded inside the training components -- the orchestrator reads
this config and injects the values, so behaviour changes in exactly one place.

Design goals
------------
* **Task-driven.** A :class:`PredictionTaskConfig` fully describes one
  prediction problem. Adding a new prediction task later (e.g. rainfall or
  river level) is a matter of adding another task entry -- no component code
  changes. ``groundwater_level`` is the initial task.
* **Offline only.** This module trains nothing and predicts nothing. It only
  declares configuration. Training happens in ``training_pipeline.py``;
  prediction happens later in the runtime Prediction Agent by loading saved
  artifacts.
"""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("aquamind.prediction.config")

# --------------------------------------------------------------------------- #
# Paths (derived, never hardcoded to an absolute location)
# --------------------------------------------------------------------------- #

PREDICTION_AGENT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PREDICTION_AGENT_DIR.parents[1]  # prediction_agent -> agents -> root
MASTER_DATASETS_DIR: Path = PROJECT_ROOT / "structured_data" / "master_datasets"
MODELS_DIR: Path = PREDICTION_AGENT_DIR / "models"


# --------------------------------------------------------------------------- #
# Prediction task definition
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PredictionTaskConfig:
    """Declarative description of a single supervised regression task.

    The same object drives both training and (future) inference, which is what
    guarantees the raw feature contract stays identical across the two phases.
    """

    name: str
    dataset_filename: str
    target_column: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    #: Source column parsed into calendar parts (e.g. month). May be ``None``.
    datetime_column: str | None = None
    #: Subset of ``numeric_features`` that is DERIVED from ``datetime_column``
    #: (not read from the CSV). Everything else is a raw column.
    derived_features: tuple[str, ...] = ()
    #: Raw columns whose missing values make a row unusable (dropped on load).
    required_columns: tuple[str, ...] = ()

    @property
    def raw_columns_to_load(self) -> list[str]:
        """The exact CSV columns to read (selective load keeps memory small)."""
        columns: set[str] = set(self.categorical_features)
        columns |= {c for c in self.numeric_features if c not in self.derived_features}
        if self.datetime_column:
            columns.add(self.datetime_column)
        columns.add(self.target_column)
        return sorted(columns)

    @property
    def feature_columns(self) -> list[str]:
        """The final model input columns (numeric first, then categorical)."""
        return list(self.numeric_features) + list(self.categorical_features)


#: Initial task: predict groundwater level (metres below ground) from spatial
#: (latitude, longitude, district), temporal (year, derived month) and
#: measurement-type signals. 'year' is a native column; 'month' is derived from
#: the observation timestamp.
GROUNDWATER_LEVEL_TASK = PredictionTaskConfig(
    name="groundwater_level",
    dataset_filename="master_groundwater_level.csv",
    target_column="groundwater_level_m",
    numeric_features=("latitude", "longitude", "year", "month"),
    categorical_features=("district", "measurement_type"),
    datetime_column="observation_time",
    derived_features=("month",),
    required_columns=("latitude", "longitude"),
)

#: Registry of available prediction tasks, keyed by name. New prediction tasks
#: are added here without touching any component.
TASKS: dict[str, PredictionTaskConfig] = {
    GROUNDWATER_LEVEL_TASK.name: GROUNDWATER_LEVEL_TASK,
}

DEFAULT_TASK: str = GROUNDWATER_LEVEL_TASK.name


# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and controls for the offline training run."""

    # --- split & determinism ---
    test_size: float = 0.2
    random_state: int = 42

    # --- data volume ---
    #: The groundwater dataset has millions of rows. Training on a fixed,
    #: deterministically sampled subset keeps offline training fast and
    #: reproducible while remaining representative. Set to ``None`` to train on
    #: the full dataset.
    max_training_rows: int | None = 80_000

    # --- model selection ---
    #: Metric used to pick the winner. "rmse"/"mae"/"mape" -> lower is better;
    #: "r2" -> higher is better.
    selection_metric: str = "rmse"

    # --- Random Forest ---
    rf_n_estimators: int = 120
    rf_max_depth: int | None = 16
    rf_min_samples_leaf: int = 10

    # --- Gradient Boosting ---
    gb_n_estimators: int = 150
    gb_max_depth: int = 3
    gb_learning_rate: float = 0.1
    gb_subsample: float = 0.8

    # --- XGBoost ---
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1

    # --- LightGBM ---
    lgbm_n_estimators: int = 300
    lgbm_num_leaves: int = 31
    lgbm_learning_rate: float = 0.1


TRAINING_CONFIG = TrainingConfig()


# --------------------------------------------------------------------------- #
# Dataset integration configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class IntegrationConfig:
    """Controls the Dataset Integrator's balancing and join validation.

    The groundwater dataset (the only source carrying the prediction target) is
    the base. It is downsampled with grouped/stratified sampling so no single
    district or year dominates. The remaining datasets are joined on as
    enrichment features and are never allowed to add or duplicate base rows.
    """

    #: Approximate number of rows in the balanced base sample.
    base_sample_target: int = 80_000
    #: Grouping keys for stratified/grouped sampling of the base (balance across
    #: districts and years so heavily-monitored places do not dominate).
    stratify_columns: tuple[str, ...] = ("district_key", "year")
    #: An enrichment source whose validated join matches fewer than this
    #: percentage of base rows is EXCLUDED (its features are not added), rather
    #: than fabricating a relationship. Documented in the integration report.
    min_join_match_pct: float = 1.0


INTEGRATION_CONFIG = IntegrationConfig()


# --------------------------------------------------------------------------- #
# Candidate model factory
# --------------------------------------------------------------------------- #

def build_candidate_models(config: TrainingConfig = TRAINING_CONFIG) -> "OrderedDict[str, Any]":
    """Build the ordered set of candidate regressors to compare.

    A linear baseline plus tree ensembles are always included. XGBoost and
    LightGBM are added only if they are already installed in the project
    (never a hard dependency). Every estimator is seeded for determinism.
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    models: "OrderedDict[str, Any]" = OrderedDict()

    # Baseline: interpretable, extremely fast inference.
    models["LinearRegression"] = LinearRegression()

    # Bagged trees: robust, low-variance, parallel.
    models["RandomForest"] = RandomForestRegressor(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_leaf=config.rf_min_samples_leaf,
        random_state=config.random_state,
        n_jobs=-1,
    )

    # Boosted trees (scikit-learn implementation).
    models["GradientBoosting"] = GradientBoostingRegressor(
        n_estimators=config.gb_n_estimators,
        max_depth=config.gb_max_depth,
        learning_rate=config.gb_learning_rate,
        subsample=config.gb_subsample,
        random_state=config.random_state,
    )

    # Optional high-performance boosters -- only if already available.
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:
        logger.info("xgboost not installed; skipping XGBoost candidate.")

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=config.lgbm_n_estimators,
            num_leaves=config.lgbm_num_leaves,
            learning_rate=config.lgbm_learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        logger.info("lightgbm not installed; skipping LightGBM candidate.")

    return models


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def configure_logging(level: int = logging.INFO) -> None:
    """Configure clean console logging once (no ``print`` debugging anywhere)."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
        )
