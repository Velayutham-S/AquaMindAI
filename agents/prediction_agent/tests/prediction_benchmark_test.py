"""Permanent prediction regression benchmark for the AquaMind AI Prediction Agent.

Runs a fixed set of ~100 realistic groundwater-level prediction queries through
the FROZEN saved model and writes a deterministic report. Re-run this after any
retraining or model change to compare outputs and catch regressions.

It does NOT retrain. It loads the saved pipeline + metadata from the model
registry (exactly as the future runtime Prediction Agent will) and turns each
natural-language query -- described by (district, year, month) -- into a feature
row using a district reference lookup:

* latitude / longitude : per-district centroid from the groundwater dataset,
* district-level assessment + firka features : reused from the DatasetIntegrator's
  own aggregation (so benchmark enrichment matches training semantics),
* temporal enrichment (year-specific rainfall / river level) : left NaN, since a
  forecast for a future year has no such record -> imputed by the pipeline.

Scope: the model is a single-target regressor for groundwater level, so every
query is a groundwater-level prediction. "Trend"/"extraction-stage" style
questions are out of scope for this model and are intentionally excluded.

Run:
    python agents/prediction_agent/tests/prediction_benchmark_test.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

TEST_DIR = Path(__file__).resolve().parent
PREDICTION_AGENT_DIR = TEST_DIR.parent
TRAINING_DIR = PREDICTION_AGENT_DIR / "training"
PROJECT_ROOT = PREDICTION_AGENT_DIR.parents[1]
for _path in (PREDICTION_AGENT_DIR, TRAINING_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import pandas as pd  # noqa: E402

import prediction_config as config  # noqa: E402  (unique module name; avoids sys.modules 'config' collision)
from dataset_integrator import DatasetIntegrator, _DISTRICT_KEY  # noqa: E402
from feature_engineering import FeatureEngineer  # noqa: E402
from model_registry import ModelRegistry  # noqa: E402

BENCHMARK_PATH = PREDICTION_AGENT_DIR / "prediction_benchmark.json"
OUTPUT_PATH = PROJECT_ROOT / "prediction_benchmark_report.txt"
_CORE_FIELDS = {"latitude", "longitude", "year", "month"}
_DEFAULT_MEASUREMENT_TYPE = "Quarterly Manual"


def _effective_task_from_metadata(metadata: dict):
    """Rebuild the task feature contract from saved metadata."""
    fm = metadata["feature_metadata"]
    base_task = config.TASKS[config.DEFAULT_TASK]
    return dataclasses.replace(
        base_task,
        numeric_features=tuple(fm["numeric_features"]),
        categorical_features=tuple(fm["categorical_features"]),
    )


def _build_district_reference(integrator: DatasetIntegrator, base_task) -> pd.DataFrame:
    """Per-district reference: centroid location + district/firka enrichment.

    Reuses the integrator's own aggregation for the enrichment features so the
    benchmark feeds the model values consistent with how it was trained.
    """
    # District-level and firka-level enrichment (reused from the integrator).
    reference = None
    for enrichment in (integrator._district_features(), integrator._firka_features()):
        if enrichment is None:
            continue
        reference = (enrichment.aggregated if reference is None
                     else reference.merge(enrichment.aggregated, on=_DISTRICT_KEY, how="outer"))

    # Per-district centroid from the groundwater dataset (representative location).
    gw_path = config.MASTER_DATASETS_DIR / base_task.dataset_filename
    gw = pd.read_csv(
        gw_path,
        usecols=["district", "latitude", "longitude"],
        dtype={"district": "category", "latitude": "float32", "longitude": "float32"},
    )
    gw[_DISTRICT_KEY] = DatasetIntegrator._normalize_key(gw["district"])
    centroid = gw.groupby(_DISTRICT_KEY, as_index=False, observed=True)[["latitude", "longitude"]].mean()
    reference = centroid if reference is None else reference.merge(centroid, on=_DISTRICT_KEY, how="outer")

    return reference.set_index(_DISTRICT_KEY)


def _build_query_frame(queries: list[dict], reference: pd.DataFrame,
                       effective_task) -> tuple[pd.DataFrame, list[bool]]:
    """Turn benchmark queries into a raw feature DataFrame the pipeline can score."""
    rows: list[dict] = []
    matched_reference: list[bool] = []
    ref_columns = list(reference.columns)

    for item in queries:
        district = str(item["district"])
        key = district.strip().upper()
        year = int(item["year"])
        month = int(item["month"])

        row: dict = {
            "district": district,
            "measurement_type": _DEFAULT_MEASUREMENT_TYPE,
            "year": year,
            # Timestamp so FeatureEngineer derives 'month' exactly as in training.
            "observation_time": f"15-{month:02d}-{year} 00:00",
        }
        if key in reference.index:
            matched_reference.append(True)
            for column in ref_columns:
                row[column] = reference.at[key, column]
        else:
            matched_reference.append(False)
            for column in ref_columns:
                row[column] = float("nan")

        # Any remaining model features not supplied (e.g. temporal enrichment for a
        # future year) are left NaN and imputed by the saved pipeline.
        for column in effective_task.numeric_features:
            if column not in _CORE_FIELDS and column not in row:
                row[column] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows), matched_reference


def main() -> int:
    task_name = config.DEFAULT_TASK
    queries = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    # --- load the frozen model (no retraining) ---
    registry = ModelRegistry(config.MODELS_DIR)
    try:
        pipeline, metadata = registry.load(task_name)
    except FileNotFoundError:
        print("No saved model found. Train it first: python training/training_pipeline.py")
        return 1
    effective_task = _effective_task_from_metadata(metadata)

    # --- reference lookup + feature rows ---
    integrator = DatasetIntegrator(config.MASTER_DATASETS_DIR)
    reference = _build_district_reference(integrator, effective_task)
    raw, matched_reference = _build_query_frame(queries, reference, effective_task)

    # --- predict (single frozen pipeline call) ---
    engineer = FeatureEngineer(effective_task)
    features, _ = engineer.build_features(raw, require_target=False)

    warm = features.iloc[[0]]
    pipeline.predict(warm)
    start = perf_counter()
    predictions = pipeline.predict(features)
    latency_ms = (perf_counter() - start) / len(features) * 1000.0

    # --- write report ---
    lines: list[str] = []
    section = "=" * 72
    finite = 0
    for item, pred, matched in zip(queries, predictions, matched_reference):
        value = float(pred)
        finite += int(math.isfinite(value))
        lines.append(section)
        lines.append(f"Benchmark ID : {item['id']}")
        lines.append(f"Query        : {item['query']}")
        lines.append(f"District     : {item['district']} "
                     f"(reference {'matched' if matched else 'not found -> imputed'})")
        lines.append(f"Year / Month : {item['year']} / {item['month']:02d}")
        lines.append(f"Target       : {item['target']}")
        lines.append(f"Prediction   : {value:.3f} m below ground level")
        lines.append("")

    districts = sorted({q["district"] for q in queries})
    years = sorted({q["year"] for q in queries})
    values = [float(p) for p in predictions if math.isfinite(float(p))]
    matched_count = sum(matched_reference)

    lines.append(section)
    lines.append("PREDICTION BENCHMARK SUMMARY")
    lines.append(section)
    lines.append(f"Generated (UTC)          : {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Model                    : {metadata.get('selected_model')}")
    lines.append(f"Model created (UTC)      : {metadata.get('created_utc')}")
    lines.append(f"Total Queries            : {len(queries)}")
    lines.append(f"Successful Predictions   : {finite}")
    lines.append(f"Failed Predictions       : {len(queries) - finite}")
    lines.append(f"Districts Covered        : {len(districts)}")
    lines.append(f"Years Covered            : {min(years)}-{max(years)} ({len(years)} distinct)")
    lines.append(f"Reference Matched        : {matched_count}/{len(queries)}")
    if values:
        lines.append(f"Prediction min/mean/max  : "
                     f"{min(values):.3f} / {sum(values) / len(values):.3f} / {max(values):.3f} m")
    lines.append(f"Feature Count (encoded)  : {metadata['data'].get('transformed_feature_count')}")
    lines.append(f"Prediction Latency       : {latency_ms:.4f} ms/query")
    overall = "PASS" if finite == len(queries) else "FAIL"
    lines.append(f"Overall Status           : {overall}")
    lines.append(section)

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Ran {len(queries)} benchmark queries through '{metadata.get('selected_model')}'.")
    print(f"Reference matched: {matched_count}/{len(queries)} | "
          f"Successful: {finite}/{len(queries)}")
    print(f"Report written to {OUTPUT_PATH}")
    print(f"Overall Status: {overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
