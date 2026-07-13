"""Temporary integration test for the AquaMind AI Prediction Agent training.

NOT production code. It exercises the full offline flow (now with dataset
integration) and the runtime loading contract:

    1. Train the model on the INTEGRATED dataset and save artifacts.
    2. Reload the saved model + metadata from the registry (as the runtime
       Prediction Agent will).
    3. Reconstruct the feature contract from metadata, build several prediction
       examples (core fields provided; enrichment features left NaN to prove a
       minimal runtime query still works via imputation), run them through the
       reloaded pipeline.
    4. Verify predictions are produced, the model reloads correctly, and the
       saved artifacts are reusable and deterministic.

Run:
    python agents/prediction_agent/tests/prediction_pipeline_test.py
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PREDICTION_AGENT_DIR = TEST_DIR.parent
TRAINING_DIR = PREDICTION_AGENT_DIR / "training"
for _path in (PREDICTION_AGENT_DIR, TRAINING_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import pandas as pd  # noqa: E402

import config  # noqa: E402
import training_pipeline  # noqa: E402
from feature_engineering import FeatureEngineer  # noqa: E402
from model_registry import ModelRegistry  # noqa: E402

#: Core raw fields a runtime caller always provides (everything else is enriched).
_CORE_FIELDS = {"latitude", "longitude", "year", "month"}


def _example_rows(numeric_features, categorical_features) -> pd.DataFrame:
    """Realistic raw inference rows: core fields set, enrichment features NaN.

    This mirrors a minimal runtime query -- the caller supplies location/date and
    the model still predicts, with unknown enrichment features imputed. Enrichment
    columns are present (so FeatureEngineer can select them) but left NaN.
    """
    rows = [
        {"latitude": 11.5611, "longitude": 79.5542, "year": 2015,
         "observation_time": "01-01-2015 00:00", "district": "Cuddalore",
         "measurement_type": "Quarterly Manual"},
        {"latitude": 13.0827, "longitude": 80.2707, "year": 2020,
         "observation_time": "01-08-2020 00:00", "district": "Chennai",
         "measurement_type": "Quarterly Manual"},
        {"latitude": 9.9252, "longitude": 78.1198, "year": 2018,
         "observation_time": "15-04-2018 00:00", "district": "Madurai",
         "measurement_type": "Quarterly Manual"},
        {"latitude": 10.7905, "longitude": 78.7047, "year": 2022,
         "observation_time": None, "district": "UnknownDistrict",
         "measurement_type": "Quarterly Manual"},
    ]
    frame = pd.DataFrame(rows)
    # Ensure every enrichment numeric feature exists (NaN -> imputed at inference).
    for column in numeric_features:
        if column not in _CORE_FIELDS and column not in frame.columns:
            frame[column] = float("nan")
    for column in categorical_features:
        if column not in frame.columns:
            frame[column] = "unknown"
    return frame


def _effective_task_from_metadata(metadata: dict):
    """Rebuild the task's feature contract from the saved metadata."""
    fm = metadata["feature_metadata"]
    base_task = config.TASKS[config.DEFAULT_TASK]
    return dataclasses.replace(
        base_task,
        numeric_features=tuple(fm["numeric_features"]),
        categorical_features=tuple(fm["categorical_features"]),
    )


def main() -> int:
    checks: list[tuple[str, bool]] = []
    task_name = config.DEFAULT_TASK

    # --- 1. Train (on integrated data) + save ---
    print("[1/4] Training model on the integrated dataset (offline)...")
    summary = training_pipeline.train_task()
    checks.append(("training produced a selected model", bool(summary.get("selected_model"))))
    checks.append(("training used integrated rows", summary.get("integrated_rows", 0) > 0))
    checks.append(("enrichment features were added", len(summary.get("feature_columns_added", [])) > 0))

    # --- 2. Reload saved artifacts ---
    print("[2/4] Reloading saved model + metadata from registry...")
    registry = ModelRegistry(config.MODELS_DIR)
    pipeline, metadata = registry.load(task_name)
    checks.append(("saved model file exists", registry.model_path(task_name).exists()))
    checks.append(("metadata has selected_model", "selected_model" in metadata))
    checks.append(("metadata has integration audit", "integration" in metadata))
    checks.append(("reloaded pipeline exposes predict", hasattr(pipeline, "predict")))

    # --- 3. Run several prediction examples via the feature contract ---
    print("[3/4] Running prediction examples through the reloaded pipeline...")
    effective_task = _effective_task_from_metadata(metadata)
    engineer = FeatureEngineer(effective_task)
    raw = _example_rows(effective_task.numeric_features, effective_task.categorical_features)
    features, _ = engineer.build_features(raw, require_target=False)
    predictions = pipeline.predict(features)

    checks.append(("one prediction per input row", len(predictions) == len(raw)))
    all_finite = all(math.isfinite(float(p)) for p in predictions)
    checks.append(("all predictions are finite numbers", all_finite))

    print("      Example predictions (groundwater level, m below ground):")
    for (_, row), pred in zip(raw.iterrows(), predictions):
        print(f"        {row['district']:<16} {int(row['year'])}  -> {float(pred):8.3f} m")

    # --- 4. Confirm artifacts are reusable (predict twice = identical) ---
    print("[4/4] Verifying deterministic reuse of the saved model...")
    predictions_again = pipeline.predict(features)
    checks.append(("reloaded model is deterministic",
                   list(map(float, predictions)) == list(map(float, predictions_again))))

    # --- summary ---
    line = "=" * 60
    print(f"\n{line}\nPREDICTION PIPELINE INTEGRATION TEST\n{line}")
    passed = 0
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        passed += int(ok)
    overall = passed == len(checks)
    print("-" * 60)
    print(f"Selected Model     : {summary.get('selected_model')}")
    print(f"Integrated Rows    : {summary.get('integrated_rows')}")
    print(f"Enrichment Added   : {', '.join(summary.get('feature_columns_added', [])) or 'none'}")
    print(f"Checks Passed      : {passed}/{len(checks)}")
    print(f"Overall Status     : {'PASS' if overall else 'FAIL'}")
    print(line)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
