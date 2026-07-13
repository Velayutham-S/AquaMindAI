"""Temporary integration test for the AquaMind AI Prediction Formatter.

NOT production code. It sources REAL runtime predictions by reusing the existing
prediction pipeline (the benchmark harness helpers -- no retraining, no dataset
reload, no evaluation), converts each into Prediction Evidence via the
PredictionFormatter, and writes a human-readable report for manual review.

    prediction_benchmark.json
      -> existing prediction pipeline (frozen model)  -> runtime prediction result
      -> PredictionFormatter.format()                 -> structured response

Run:
    python agents/prediction_agent/tests/prediction_formatter_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

TEST_DIR = Path(__file__).resolve().parent
PREDICTION_AGENT_DIR = TEST_DIR.parent
TRAINING_DIR = PREDICTION_AGENT_DIR / "training"
FORMATTER_DIR = PREDICTION_AGENT_DIR / "formatter"
PROJECT_ROOT = PREDICTION_AGENT_DIR.parents[1]
for _path in (PREDICTION_AGENT_DIR, TRAINING_DIR, FORMATTER_DIR, TEST_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import config  # noqa: E402
import prediction_benchmark_test as bench  # noqa: E402  (reuse the existing pipeline)
from dataset_integrator import DatasetIntegrator  # noqa: E402
from feature_engineering import FeatureEngineer  # noqa: E402
from model_registry import ModelRegistry  # noqa: E402
from prediction_formatter import PredictionFormatter  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "prediction_formatter_output.txt"
SECTION = "=" * 60


def _runtime_predictions() -> tuple[list[dict], list[float], str, str]:
    """Produce real runtime predictions via the existing (frozen) pipeline."""
    queries = json.loads(bench.BENCHMARK_PATH.read_text(encoding="utf-8"))

    registry = ModelRegistry(config.MODELS_DIR)
    pipeline, metadata = registry.load(config.DEFAULT_TASK)
    effective_task = bench._effective_task_from_metadata(metadata)

    integrator = DatasetIntegrator(config.MASTER_DATASETS_DIR)
    reference = bench._build_district_reference(integrator, effective_task)
    raw, _matched = bench._build_query_frame(queries, reference, effective_task)

    engineer = FeatureEngineer(effective_task)
    features, _ = engineer.build_features(raw, require_target=False)
    predictions = [float(p) for p in pipeline.predict(features)]

    return queries, predictions, metadata.get("selected_model"), effective_task.target_column


def _is_valid(response: dict, prediction_result: dict) -> bool:
    """A response is valid when it carries the expected envelope and echoes the value."""
    if response.get("agent_name") != "prediction_agent":
        return False
    if response.get("query_type") != "prediction":
        return False
    if response.get("status") != "SUCCESS":
        return False
    prediction = response.get("prediction")
    if not isinstance(prediction, dict):
        return False
    return prediction.get("predicted_value") == prediction_result["predicted_value"]


def main() -> int:
    queries, predictions, model_name, target = _runtime_predictions()
    formatter = PredictionFormatter()

    lines: list[str] = []
    successful = 0
    failed = 0
    format_times: list[float] = []

    for item, predicted_value in zip(queries, predictions):
        prediction_result = {
            "district": item["district"],
            "prediction_year": int(item["year"]),
            "prediction_month": int(item["month"]),
            "target": target,
            "predicted_value": predicted_value,
            "model_name": model_name,
        }

        start = perf_counter()
        response = formatter.format(prediction_result)
        format_times.append(perf_counter() - start)

        if _is_valid(response, prediction_result):
            successful += 1
        else:
            failed += 1

        lines.append(SECTION)
        lines.append(f"Benchmark ID : {item['id']}")
        lines.append(f"User Query   : {item['query']}")
        lines.append("Prediction Result:")
        lines.append(json.dumps(prediction_result, indent=2, ensure_ascii=False, default=str))
        lines.append("Formatted Prediction Response:")
        lines.append(json.dumps(response, indent=2, ensure_ascii=False, default=str))
        lines.append(SECTION)
        lines.append("")

    average_format_time = (sum(format_times) / len(format_times)) if format_times else 0.0
    overall_pass = failed == 0 and successful == len(queries)

    lines.append(SECTION)
    lines.append("PREDICTION FORMATTER SUMMARY")
    lines.append(SECTION)
    lines.append(f"Model                   : {model_name}")
    lines.append(f"Total Queries           : {len(queries)}")
    lines.append(f"Successful Formatting   : {successful}")
    lines.append(f"Failed Formatting       : {failed}")
    lines.append(f"Average Formatting Time : {average_format_time:.6f}s")
    lines.append(f"Overall Status          : {'PASS' if overall_pass else 'FAIL'}")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Formatted {len(queries)} runtime predictions via PredictionFormatter.")
    print(f"Report written to {OUTPUT_PATH}")
    print(f"Overall Status: {'PASS' if overall_pass else 'FAIL'} "
          f"({successful}/{len(queries)} successful)")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
