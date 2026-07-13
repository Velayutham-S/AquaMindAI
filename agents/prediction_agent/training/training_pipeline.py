"""Offline training pipeline orchestrator for the AquaMind AI Prediction Agent.

Wires the dependency-injected components into the end-to-end flow:

    all master datasets -> DatasetIntegrator (balanced, enriched, validated joins)
                        -> FeatureEngineer -> train/validation split
                        -> ModelTrainer (candidate pipelines) -> ModelEvaluator
                        -> best-model selection -> ModelRegistry -> saved artifacts

Running this module (``python training_pipeline.py``) trains the default task,
selects the best regression algorithm by objective validation metrics, saves
the fitted pipeline + metadata, and prints the required training report.

This is OFFLINE only. Prediction happens later in the runtime Prediction Agent,
which loads the saved artifacts and never retrains.
"""

from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd

# --- wire up config + sibling training components (no __init__.py; import by location) ---
_TRAINING_DIR = Path(__file__).resolve().parent
_PREDICTION_AGENT_DIR = _TRAINING_DIR.parent
for _path in (_PREDICTION_AGENT_DIR, _TRAINING_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import config  # noqa: E402
from dataset_integrator import DatasetIntegrator  # noqa: E402
from feature_engineering import FeatureEngineer  # noqa: E402
from model_trainer import ModelTrainer  # noqa: E402
from model_evaluator import ModelEvaluator  # noqa: E402
from model_registry import ModelRegistry  # noqa: E402

logger = logging.getLogger("aquamind.prediction.training_pipeline")


def _library_versions() -> dict:
    """Record versions relevant to reproducing the saved model."""
    versions = {"python": platform.python_version()}
    for name in ("sklearn", "pandas", "numpy", "joblib", "xgboost", "lightgbm"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def _measure_prediction_latency(pipeline, sample_row: pd.DataFrame, iterations: int = 200) -> float:
    """Return mean single-row prediction latency in milliseconds."""
    pipeline.predict(sample_row)  # warm up
    start = perf_counter()
    for _ in range(iterations):
        pipeline.predict(sample_row)
    return (perf_counter() - start) / iterations * 1000.0


def train_task(
    task_name: str = config.DEFAULT_TASK,
    training_config: config.TrainingConfig = config.TRAINING_CONFIG,
) -> dict:
    """Train, evaluate, select and persist the best model for ``task_name``.

    Returns a summary dict (also reflected in the saved metadata).
    """
    from sklearn.model_selection import train_test_split

    config.configure_logging()

    if task_name not in config.TASKS:
        raise ValueError(f"Unknown prediction task '{task_name}'. Known: {list(config.TASKS)}")
    task = config.TASKS[task_name]

    # --- components (dependency injection) ---
    integrator = DatasetIntegrator(
        config.MASTER_DATASETS_DIR,
        config.INTEGRATION_CONFIG,
        random_state=training_config.random_state,
    )
    evaluator = ModelEvaluator()
    registry = ModelRegistry(config.MODELS_DIR)

    pipeline_start = perf_counter()

    # --- integrate all master datasets -> one balanced, enriched DataFrame ---
    integration = integrator.integrate(task, base_sample_target=training_config.max_training_rows)
    effective_task = integration.effective_task

    # --- features (existing FeatureEngineer, driven by the effective task) ---
    engineer = FeatureEngineer(effective_task)
    trainer = ModelTrainer(preprocessor_factory=engineer.build_preprocessor)
    features, target = engineer.build_features(integration.frame, require_target=True)

    # --- split ---
    x_train, x_val, y_train, y_val = train_test_split(
        features, target,
        test_size=training_config.test_size,
        random_state=training_config.random_state,
    )

    # --- train candidates ---
    candidates = config.build_candidate_models(training_config)
    logger.info("Training %d candidate models: %s", len(candidates), ", ".join(candidates))
    trained = trainer.train(candidates, x_train, y_train)

    # --- evaluate + select ---
    logger.info("Evaluating candidates on the validation split:")
    results = evaluator.compare(trained, x_val, y_val)
    best = evaluator.select_best(results, metric=training_config.selection_metric)
    best_pipeline = trained[best.name].pipeline

    # --- latency + transformed feature count ---
    single_row = x_val.iloc[[0]]
    latency_ms = _measure_prediction_latency(best_pipeline, single_row)
    transformed_feature_count = int(
        best_pipeline.named_steps["preprocessor"].transform(single_row).shape[1]
    )

    total_seconds = perf_counter() - pipeline_start

    # --- metadata ---
    metadata = {
        "task": task.name,
        "target": task.target_column,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": _library_versions(),
        "training_config": {
            "test_size": training_config.test_size,
            "random_state": training_config.random_state,
            "max_training_rows": training_config.max_training_rows,
            "selection_metric": training_config.selection_metric,
        },
        "feature_metadata": engineer.feature_metadata(),
        "metrics_notes": {
            "units": "target and MAE/RMSE are in metres",
            "mape": "computed only on validation rows with |target| >= 1 m "
                    "(target can be near zero, where MAPE is undefined/meaningless)",
        },
        "integration": {
            "strategy": integration.strategy_description,
            "feature_columns_added": integration.feature_columns_added,
            "sources": [s.as_dict() for s in integration.source_reports],
            "joins": [j.as_dict() for j in integration.join_reports],
        },
        "data": {
            "integrated_rows": int(len(integration.frame)),
            "training_rows": int(len(x_train)),
            "validation_rows": int(len(x_val)),
            "features_used": list(effective_task.feature_columns),
            "transformed_feature_count": transformed_feature_count,
        },
        "selected_model": best.name,
        "evaluation": {
            "selection_metric": training_config.selection_metric,
            "selected_model": best.name,
            "candidates": [r.as_dict() for r in results],
        },
    }

    # --- persist ---
    model_file, _ = registry.save(task.name, best_pipeline, metadata)
    model_size_bytes = registry.model_file_size_bytes(task.name)

    _print_report(
        task=effective_task,
        integration=integration,
        training_rows=len(x_train),
        validation_rows=len(x_val),
        transformed_feature_count=transformed_feature_count,
        results=results,
        best=best,
        total_seconds=total_seconds,
        model_size_bytes=model_size_bytes,
        latency_ms=latency_ms,
        model_file=model_file,
    )

    return {
        "task": task.name,
        "selected_model": best.name,
        "training_rows": len(x_train),
        "validation_rows": len(x_val),
        "integrated_rows": int(len(integration.frame)),
        "features_used": list(effective_task.feature_columns),
        "feature_columns_added": integration.feature_columns_added,
        "results": [r.as_dict() for r in results],
        "model_file": str(model_file),
        "model_size_bytes": model_size_bytes,
        "latency_ms": latency_ms,
        "total_seconds": total_seconds,
    }


def _format_mape(mape: float) -> str:
    """Render MAPE as a percentage, or 'n/a' when it is undefined."""
    if mape != mape:  # NaN
        return "n/a"
    return f"{mape * 100:.2f}%"


def _print_report(*, task, integration, training_rows, validation_rows,
                  transformed_feature_count, results, best,
                  total_seconds, model_size_bytes, latency_ms, model_file) -> None:
    """Print the required training report block (with dataset integration audit)."""
    line = "=" * 72
    print(f"\n{line}")
    print(f"PREDICTION TRAINING REPORT  --  task: {task.name}")
    print(line)
    print(f"Target                 : {task.target_column} (metres)")

    # --- dataset integration audit ---
    print("-" * 72)
    print("DATASET INTEGRATION")
    print(f"  {'Source dataset':<38}{'Total':>10}{'Used':>9}  Role")
    for s in integration.source_reports:
        print(f"  {s.name:<38}{s.total_rows:>10}{s.used_rows:>9}  {s.role}")
    print("\n  Join validation (base = groundwater):")
    print(f"  {'Right dataset':<34}{'Match%':>8}{'Matched':>10}{'Unmatched':>11}  Incl")
    for j in integration.join_reports:
        print(f"  {j.right:<34}{j.match_pct:>7.2f}%{j.matched_rows:>10}"
              f"{j.unmatched_rows:>11}  {'yes' if j.included else 'no'}")
    print(f"\n  Join keys used         : "
          f"{sorted({k for j in integration.join_reports if j.included for k in j.join_keys}) or 'none'}")
    print(f"  Sampling strategy      : grouped/stratified across "
          f"{list(config.INTEGRATION_CONFIG.stratify_columns)} (per-group cap)")
    print(f"  Integrated Rows        : {len(integration.frame)}")
    print(f"  Features Used ({len(task.feature_columns):>2})     : {', '.join(task.feature_columns)}")
    print(f"  Enrichment Added       : {', '.join(integration.feature_columns_added) or 'none'}")
    print(f"  Final Feature Count    : {transformed_feature_count} (after encoding)")

    print("-" * 72)
    print(f"Training Samples       : {training_rows}")
    print(f"Validation Samples     : {validation_rows}")
    print(f"Algorithms Tested      : {', '.join(r.name for r in results)}")
    print(f"Selected Model         : {best.name}")
    print("-" * 72)
    print("Candidate comparison (validation, best by RMSE marked *):")
    print(f"  {'Model':<18}{'MAE':>9}{'RMSE':>9}{'R2':>9}{'MAPE':>11}{'Fit(s)':>9}")
    for r in results:
        marker = "*" if r.name == best.name else " "
        print(f"{marker} {r.name:<18}{r.mae:>9.4f}{r.rmse:>9.4f}"
              f"{r.r2:>9.4f}{_format_mape(r.mape):>11}{r.fit_seconds:>9.2f}")
    print("-" * 72)
    print("Selected model evaluation metrics:")
    print(f"  MAE                  : {best.mae:.4f} m")
    print(f"  RMSE                 : {best.rmse:.4f} m")
    print(f"  R2 Score             : {best.r2:.4f}")
    print(f"  MAPE                 : {_format_mape(best.mape)} (|target| >= 1 m only)")
    print("-" * 72)
    print(f"Training Time (total)  : {total_seconds:.2f} s")
    print(f"Model File             : {model_file}")
    print(f"Model File Size        : {model_size_bytes / 1024:.2f} KB "
          f"({model_size_bytes / (1024 * 1024):.2f} MB)")
    print(f"Prediction Latency     : {latency_ms:.4f} ms (single prediction)")
    print(f"{line}\n")


def main() -> int:
    train_task()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
