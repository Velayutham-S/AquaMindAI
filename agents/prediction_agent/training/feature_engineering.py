"""Feature engineering for the AquaMind AI Prediction Agent.

Single responsibility: turn a task's cleaned DataFrame into the model's input
matrix, and provide the reusable scikit-learn preprocessor that performs
imputation, scaling and categorical encoding.

Why this design guarantees train/inference parity
--------------------------------------------------
Feature engineering happens in two layers, and BOTH are reused unchanged at
inference time by the future Prediction Agent:

1. **Raw feature construction** (:meth:`FeatureEngineer.build_features`):
   deriving calendar parts (e.g. ``month``) from the observation timestamp and
   selecting the task's feature columns. This is a deterministic, stateless
   transformation, so calling it on a single incoming query row reproduces the
   exact training-time features.

2. **Fitted preprocessing** (:meth:`FeatureEngineer.build_preprocessor`): a
   ``ColumnTransformer`` (impute + scale numerics, impute + one-hot encode
   categoricals). It is fitted during training and saved INSIDE the model
   pipeline, so the identical fitted statistics/encodings are applied at
   inference. Unknown categories seen at inference are ignored, never crash.

This module trains nothing and predicts nothing.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("aquamind.prediction.feature_engineering")

#: Fill value used for missing categorical values (kept explicit and stable so
#: the same token is used at training and inference).
_CATEGORICAL_FILL = "unknown"


class FeatureEngineer:
    """Builds task feature matrices and the reusable preprocessing transformer."""

    def __init__(self, task) -> None:
        self._task = task

    # ------------------------------------------------------------------ #
    # Raw feature construction (stateless, reused at inference)
    # ------------------------------------------------------------------ #

    def build_features(
        self, frame: pd.DataFrame, *, require_target: bool = True
    ) -> tuple[pd.DataFrame, "pd.Series | None"]:
        """Return ``(X, y)`` where ``X`` holds the task feature columns.

        ``y`` is ``None`` when the target column is absent (inference). The same
        method is used for training and for scoring incoming query rows.
        """
        work = frame.copy()
        self._add_datetime_parts(work)

        missing = [c for c in self._task.feature_columns if c not in work.columns]
        if missing:
            raise ValueError(f"Missing required feature columns for '{self._task.name}': {missing}")

        # Categoricals as plain strings so imputation/encoding behave predictably.
        features = work[self._task.feature_columns].copy()
        for column in self._task.categorical_features:
            features[column] = features[column].astype("object")

        target: "pd.Series | None" = None
        if self._task.target_column in work.columns:
            target = work[self._task.target_column].astype(float)
        elif require_target:
            raise ValueError(
                f"Target column '{self._task.target_column}' missing but required."
            )
        return features, target

    def _add_datetime_parts(self, frame: pd.DataFrame) -> None:
        """Derive requested calendar parts (currently ``month``) in place."""
        column = self._task.datetime_column
        if not column or not self._task.derived_features:
            return
        if column not in frame.columns:
            # No timestamp available (e.g. a minimal inference payload) -> leave
            # derived columns to be created as NaN and handled by the imputer.
            for part in self._task.derived_features:
                if part not in frame.columns:
                    frame[part] = pd.NA
            return

        parsed = pd.to_datetime(
            frame[column], format="%d-%m-%Y %H:%M", errors="coerce"
        )
        # Fallback for any rows not matching the primary format.
        if parsed.isna().any():
            fallback = pd.to_datetime(frame[column], errors="coerce", dayfirst=True)
            parsed = parsed.fillna(fallback)

        if "month" in self._task.derived_features:
            frame["month"] = parsed.dt.month

    # ------------------------------------------------------------------ #
    # Fitted preprocessing (saved inside the model pipeline)
    # ------------------------------------------------------------------ #

    def build_preprocessor(self):
        """Build an (unfitted) ``ColumnTransformer`` for the task features.

        Numeric: median imputation + standard scaling (scaling is harmless for
        trees and required by the linear baseline, so one shared preprocessor
        serves every candidate). Categorical: constant imputation + one-hot
        encoding that ignores unseen categories at inference.
        """
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=_CATEGORICAL_FILL)),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, list(self._task.numeric_features)),
                ("categorical", categorical_pipeline, list(self._task.categorical_features)),
            ],
            remainder="drop",
        )
        # Emit named DataFrame output so every estimator sees consistent feature
        # names at both fit and inference (avoids sklearn's feature-name mismatch
        # warning and keeps the transformed space introspectable).
        preprocessor.set_output(transform="pandas")
        return preprocessor

    # ------------------------------------------------------------------ #
    # Metadata (recorded in the model registry)
    # ------------------------------------------------------------------ #

    def feature_metadata(self) -> dict:
        """Describe the feature contract and preprocessing for the registry."""
        return {
            "numeric_features": list(self._task.numeric_features),
            "categorical_features": list(self._task.categorical_features),
            "datetime_column": self._task.datetime_column,
            "derived_features": list(self._task.derived_features),
            "raw_columns_expected": self._task.raw_columns_to_load,
            "preprocessing": {
                "numeric": ["SimpleImputer(median)", "StandardScaler"],
                "categorical": [
                    f"SimpleImputer(constant='{_CATEGORICAL_FILL}')",
                    "OneHotEncoder(handle_unknown='ignore')",
                ],
            },
        }
