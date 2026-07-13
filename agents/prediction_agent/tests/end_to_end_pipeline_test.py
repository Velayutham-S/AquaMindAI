"""Permanent end-to-end integration test for the AquaMind AI Prediction Agent.

Verifies the complete Prediction Agent pipeline exactly as production runs it,
for every query in the benchmark, and writes a report to ``final_prediction.txt``
at the project root:

    User Query (prediction_benchmark.json)
      -> parse query into a feature row     (existing runtime parsing logic)
      -> load trained model (Model Registry) (loaded once, served for all queries)
      -> run prediction                       (frozen saved pipeline)
      -> PredictionFormatter.format()         (structured Prediction Evidence)
      -> Prediction Agent response

It reuses the existing production components exactly as they are -- it does not
retrain, reload training datasets, evaluate models, regenerate benchmarks, call
any LLM, or alter prediction values. A failing query is recorded with its reason
and the run continues.

This is the permanent regression test for the Prediction Agent; keep it in the
repository to validate the agent after any future model retraining.

Run:
    python agents/prediction_agent/tests/end_to_end_pipeline_test.py
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
import prediction_benchmark_test as bench  # noqa: E402  (reuse existing runtime parsing)
from dataset_integrator import DatasetIntegrator  # noqa: E402
from feature_engineering import FeatureEngineer  # noqa: E402
from model_registry import ModelRegistry  # noqa: E402
from prediction_formatter import PredictionFormatter  # noqa: E402

BENCHMARK_PATH = PREDICTION_AGENT_DIR / "prediction_benchmark.json"
OUTPUT_PATH = PROJECT_ROOT / "final_prediction.txt"
SECTION = "=" * 50
RULE = "-" * 50


class PredictionAgentRuntime:
    """Thin orchestrator that runs the existing components end-to-end per query.

    Loads the trained model once (as a production service would) and reuses the
    existing parsing, model registry, feature engineering and formatter for every
    query -- no component logic is reimplemented here.
    """

    def __init__(self) -> None:
        registry = ModelRegistry(config.MODELS_DIR)
        self.pipeline, self.metadata = registry.load(config.DEFAULT_TASK)
        self.effective_task = bench._effective_task_from_metadata(self.metadata)
        self.model_name = self.metadata.get("selected_model")
        self.target = self.effective_task.target_column

        # Reference lookup used by the existing runtime parsing (built once).
        integrator = DatasetIntegrator(config.MASTER_DATASETS_DIR)
        self._reference = bench._build_district_reference(integrator, self.effective_task)
        self._engineer = FeatureEngineer(self.effective_task)
        self._formatter = PredictionFormatter()

    def run(self, query: dict) -> dict:
        """Execute the full pipeline for one query; returns timings + payloads."""
        start = perf_counter()

        # 1. Parse the query into a single feature row (existing runtime logic).
        raw, _matched = bench._build_query_frame([query], self._reference, self.effective_task)
        features, _ = self._engineer.build_features(raw, require_target=False)

        # 2. Predict with the already-loaded model.
        predict_start = perf_counter()
        predicted_value = float(self.pipeline.predict(features)[0])
        prediction_seconds = perf_counter() - predict_start

        # 3. Assemble the runtime prediction result.
        prediction_result = {
            "district": query["district"],
            "prediction_year": int(query["year"]),
            "prediction_month": int(query["month"]),
            "target": self.target,
            "predicted_value": predicted_value,
            "model_name": self.model_name,
        }

        # 4. Format into the Prediction Agent response.
        format_start = perf_counter()
        response = self._formatter.format(prediction_result)
        formatting_seconds = perf_counter() - format_start

        return {
            "prediction_result": prediction_result,
            "response": response,
            "prediction_seconds": prediction_seconds,
            "formatting_seconds": formatting_seconds,
            "end_to_end_seconds": perf_counter() - start,
        }


def _is_success(response: dict) -> bool:
    return (
        response.get("agent_name") == "prediction_agent"
        and response.get("status") == "SUCCESS"
        and isinstance(response.get("prediction"), dict)
    )


def main() -> int:
    queries = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    runtime = PredictionAgentRuntime()

    lines: list[str] = []
    successful = 0
    failed = 0
    prediction_times: list[float] = []
    formatting_times: list[float] = []
    end_to_end_times: list[float] = []
    prediction_method = PredictionFormatter.PREDICTION_METHOD

    for item in queries:
        lines.append(SECTION)
        lines.append(f"Benchmark ID : {item.get('id')}")
        lines.append(f"User Query   : {item.get('query')}")
        lines.append(RULE)

        try:
            outcome = runtime.run(item)
            prediction_times.append(outcome["prediction_seconds"])
            formatting_times.append(outcome["formatting_seconds"])
            end_to_end_times.append(outcome["end_to_end_seconds"])

            if _is_success(outcome["response"]):
                successful += 1
            else:
                failed += 1

            lines.append("Runtime Prediction Result")
            lines.append(json.dumps(outcome["prediction_result"], indent=2,
                                    ensure_ascii=False, default=str))
            lines.append(RULE)
            lines.append("Formatted Prediction Response")
            lines.append(json.dumps(outcome["response"], indent=2,
                                    ensure_ascii=False, default=str))
        except Exception as error:  # noqa: BLE001 - record and continue
            failed += 1
            lines.append("FAILED")
            lines.append(f"Reason: {type(error).__name__}: {error}")
            print(f"FAIL Benchmark ID {item.get('id')}: {type(error).__name__}: {error}")

        lines.append(SECTION)
        lines.append("")

    def _avg_ms(values: list[float]) -> str:
        return f"{(sum(values) / len(values) * 1000):.4f} ms" if values else "n/a"

    overall_pass = failed == 0 and successful == len(queries)

    lines.append(SECTION)
    lines.append("Prediction Agent End-to-End Summary")
    lines.append(SECTION)
    lines.append(f"Total Queries            : {len(queries)}")
    lines.append(f"Successful Predictions   : {successful}")
    lines.append(f"Failed Predictions       : {failed}")
    lines.append(f"Average Prediction Time  : {_avg_ms(prediction_times)}")
    lines.append(f"Average Formatting Time  : {_avg_ms(formatting_times)}")
    lines.append(f"Average End-to-End Time  : {_avg_ms(end_to_end_times)}")
    lines.append(f"Loaded Model             : {runtime.model_name}")
    lines.append(f"Prediction Method        : {prediction_method}")
    lines.append(f"Overall Status           : {'PASS' if overall_pass else 'FAIL'}")
    lines.append(SECTION)

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Ran {len(queries)} queries end-to-end through the Prediction Agent "
          f"('{runtime.model_name}').")
    print(f"Report written to {OUTPUT_PATH}")
    print(f"Overall Status: {'PASS' if overall_pass else 'FAIL'} "
          f"({successful}/{len(queries)} successful)")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
