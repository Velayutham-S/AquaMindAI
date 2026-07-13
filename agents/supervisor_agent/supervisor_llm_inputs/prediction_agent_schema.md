# Prediction Agent — Capability Specification

> **Audience:** This document is part of the **Supervisor Planner LLM prompt**. It is not developer documentation and not user documentation. The Supervisor Planner reads this file to decide **whether the Prediction Agent should execute** and **what to expect back**. Treat every statement here as authoritative and binding. Everything described here reflects the **actual current implementation** — no future or hypothetical capability is included.

---

## PRIMARY RESPONSIBILITY

> The Prediction Agent is responsible **only** for **predicting groundwater-related values using a previously trained machine-learning model**. It performs **online inference** with a model that was trained **offline**. It returns a single **structured prediction result** — it never trains, never modifies datasets, never retrieves SQL data or documents, never reasons, never recommends, and never generates natural-language answers.

**Route to the Prediction Agent when the query asks for a predicted / forecast / estimated / future groundwater value.** Route elsewhere for measured/historical data (Data Agent), explanations/definitions (Knowledge Agent), or greetings/chit-chat (General Conversation LLM).

| Property | Value |
|---|---|
| Agent name (in response) | `prediction_agent` |
| Query type (in response) | `prediction` |
| Prediction method | `machine_learning` |
| Prediction target (current) | `groundwater_level_m` (groundwater level) |
| Prediction unit | **metres below ground level** |
| Selected production model | **XGBoost regressor** |
| Training | **Offline only** (never at runtime) |
| Inference | **Online, deterministic** (load saved model → predict) |
| Typical latency | ~11–15 ms per single prediction (in-memory, no I/O beyond one model load) |

---

## 1. Prediction Agent Overview

The **Prediction Agent** is a supervised **regression inference engine**. Given a structured prediction request (district + year + month + target), it produces a **numeric prediction** of groundwater level in **metres below ground level**, wrapped in a structured `UniversalAgentResponse`.

The model is trained **offline** (see §3) and saved to a model registry. At runtime the agent **loads the saved model + its embedded preprocessing pipeline** and calls `predict`. It does **not** retrain, re-evaluate, or modify anything.

**The Prediction Agent produces structured predictions. It does NOT:**

- train, retrain, evaluate, or select models at runtime,
- read or modify datasets,
- retrieve database (SQL) values or historical records,
- retrieve documents, embeddings, or knowledge,
- explain, reason, recommend, or generate prose,
- handle greetings, intent classification, or conversation memory.

---

## 2. Prediction Pipeline (Runtime / Online Inference)

The runtime pipeline is fixed, read-only, and deterministic. Training is entirely separate and offline.

```
User Query (natural language)
        │
        ▼
Supervisor supplies a STRUCTURED prediction request   → { district, prediction_year, prediction_month, target }
        │
        ▼
Load Saved Model (Model Registry)                      → <task>_model.joblib  (a complete fitted pipeline)
        │
        ▼
Load Saved Preprocessing Pipeline                      → bundled INSIDE the saved model (imputers, scaler, one-hot encoder)
        │
        ▼
Feature Engineering                                    → build the model's feature row (derive month from date; assemble
        │                                                  spatial/temporal/enrichment features per the saved feature contract)
        ▼
Model Prediction                                       → fitted pipeline.predict(features) → numeric groundwater level (metres)
        │
        ▼
Prediction Formatter                                   → wraps the numeric result into a structured envelope (pure formatting)
        │
        ▼
UniversalAgentResponse                                 → { agent_name, status, query_type, prediction_method, model_name, prediction }
```

**Pipeline properties:**

| Property | Guarantee |
|---|---|
| **Offline training** | Models are trained once, offline, via the training pipeline. Never at runtime. |
| **Online inference** | Runtime only loads the saved artifact and predicts. |
| **No retraining** | The runtime path can never modify or retrain the model. |
| **Deterministic inference** | Same request → same prediction (the saved pipeline is fixed; preprocessing is embedded). |
| **Preprocessing parity** | The exact fitted imputation / scaling / encoding used in training is saved **inside** the model, so training and inference preprocessing can never drift. |

---

## 3. Model Information (Actual Production Model)

| Item | Value |
|---|---|
| **Selected model** | **XGBoost** (gradient-boosted trees regressor) |
| **Selection strategy** | Best (lowest) **RMSE** on a held-out validation split |
| **Candidate models evaluated** | LinearRegression (baseline), RandomForest, GradientBoosting, XGBoost, LightGBM |
| **Training mode** | Offline, deterministic (`random_state = 42`, `test_size = 0.2`) |
| **Training data** | Balanced, integrated groundwater dataset — 73,541 integrated rows (58,832 train / 14,709 validation) |
| **Transformed feature count** | 54 (after one-hot encoding) |
| **Model registry artifacts** | `groundwater_level_model.joblib` (full pipeline: preprocessing + estimator) and `groundwater_level_metadata.json` (feature + training + evaluation metadata) |
| **Library versions** | scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.6.0, pandas 2.3.3, numpy 2.4.4, joblib 1.5.3 |

### 3.1 Candidate Evaluation (validation metrics; XGBoost selected by RMSE)

| Candidate | MAE (m) | RMSE (m) | R² | Selected |
|---|---:|---:|---:|:---:|
| LinearRegression | 56.80 | 164.29 | 0.069 | |
| RandomForest | 29.55 | 126.40 | 0.449 | |
| GradientBoosting | 45.28 | 149.28 | 0.232 | |
| **XGBoost** | **32.82** | **122.68** | **0.481** | ✅ |
| LightGBM | 32.53 | 125.08 | 0.461 | |

> Metrics are in **metres**. R² ≈ 0.48 indicates a **moderate-strength** model — predictions are directional estimates, not precise measurements (see Limitations, §16).

### 3.2 Saved Metadata (what the registry stores)

- **Feature metadata:** the exact numeric + categorical feature contract, the datetime-derived features, and the fitted preprocessing steps.
- **Training metadata:** split ratio, random seed, sample size, selection metric.
- **Evaluation metadata:** per-candidate MAE / RMSE / R² / MAPE and the selected model.
- **Integration metadata:** which datasets were joined in as enrichment and their validated match rates.

---

## 4. Supported Prediction Tasks

The Prediction Agent currently supports **exactly one** task: **groundwater level prediction** (`groundwater_level`, target `groundwater_level_m`). The architecture is task-driven, but **only this task exists today** — do not assume any other.

| Task | Supported | Notes |
|---|:---:|---|
| Predict groundwater level | ✅ | Core capability |
| Predict **future** groundwater level (future year) | ✅ | Extrapolation — see Limitations |
| Predict **historical-year** groundwater level | ✅ | Year within/near training range |
| Predict groundwater level **for a district** | ✅ | District is a model feature |
| Predict groundwater level **for a given month** | ✅ | Month is a model feature |
| Predict groundwater level **for a given year** | ✅ | Year is a model feature |
| Natural-language groundwater prediction requests | ✅ | Supervisor converts them to a structured request first |
| Predict rainfall / river discharge / any non-groundwater-level target | ❌ | Not trained; only `groundwater_level_m` exists |

**Required inputs for a prediction:** `district`, `prediction_year`, `prediction_month`, `target` (see §6).

---

## 5. Supported Question Types

Natural-language requests the Supervisor should route here (it must translate them into a structured request):

- "Predict groundwater level in Salem for 2030."
- "Forecast groundwater level next year."
- "Estimate groundwater level in Coimbatore during June 2028."
- "How deep is groundwater expected to be in Madurai next summer?"
- "Predict groundwater level."
- "Predict groundwater for Tirunelveli."
- "Future groundwater prediction."
- "Groundwater forecast."

**Trigger words:** *predict, forecast, estimate, projection, expected, future, next year, will be*.

---

## 6. Expected Input (Structured Runtime Prediction Request)

The Supervisor supplies a structured request. The Prediction Agent does **not** parse free text or read conversation memory — the Supervisor must resolve follow-ups and produce a complete request.

```json
{
  "district": "Salem",
  "prediction_year": 2030,
  "prediction_month": 6,
  "target": "groundwater_level_m"
}
```

| Field | Type | Required | Description |
|---|---|:---:|---|
| `district` | string | Yes | Tamil Nadu district name (model feature; unknown districts are tolerated but less specific). |
| `prediction_year` | integer | Yes | Calendar year to predict for (historical or future). |
| `prediction_month` | integer (1–12) | Yes | Calendar month to predict for (captures seasonality). |
| `target` | string | Yes | Must be `groundwater_level_m` (the only trained target). |

> If the user omits year/month, the Supervisor should fill sensible defaults (e.g., resolve "next year" / "next summer" to a concrete year/month) **before** calling the Prediction Agent. Enrichment features (rainfall, assessment, river level) are looked up or imputed internally — the Supervisor does not supply them.

---

## 7. Expected Output (Actual PredictionFormatter Response)

### 7.1 Success

```json
{
  "agent_name": "prediction_agent",
  "status": "SUCCESS",
  "query_type": "prediction",
  "prediction_method": "machine_learning",
  "model_name": "XGBoost",
  "prediction": {
    "district": "Salem",
    "prediction_year": 2030,
    "prediction_month": 6,
    "target": "groundwater_level_m",
    "predicted_value": -5.92,
    "unit": "metres below ground level"
  }
}
```

### 7.2 No Prediction (missing / non-finite value)

```json
{
  "agent_name": "prediction_agent",
  "status": "NO_PREDICTION",
  "query_type": "prediction",
  "prediction": null
}
```

| Field | Meaning |
|---|---|
| `agent_name` | Always `prediction_agent`. |
| `status` | `SUCCESS` when a finite prediction is produced; `NO_PREDICTION` otherwise. Never raises. |
| `query_type` | Always `prediction`. |
| `prediction_method` | Always `machine_learning`. |
| `model_name` | The model that produced the value (currently `XGBoost`). |
| `prediction.district` / `prediction_year` / `prediction_month` | Echo of the request context. |
| `prediction.target` | The predicted target (`groundwater_level_m`). |
| `prediction.predicted_value` | The numeric prediction (float). |
| `prediction.unit` | Human-readable unit (`metres below ground level`). |

---

## 8. Supported Features (Runtime Model Inputs)

The model consumes **13 features** (expanded to 54 columns after one-hot encoding). These are assembled internally per the saved feature contract; the Supervisor only supplies district/year/month/target.

| Feature | Type | Origin |
|---|---|---|
| `latitude` | numeric | Spatial (district-representative location) |
| `longitude` | numeric | Spatial |
| `year` | numeric | Temporal (from request) |
| `month` | numeric | Temporal (derived from the observation date) |
| `district` | categorical | Spatial (from request; one-hot encoded) |
| `measurement_type` | categorical | Observation metadata |
| `district_rainfall_mm_total` | numeric | **Integrated** district-level rainfall/assessment enrichment |
| `district_gw_recharge_total_ham` | numeric | **Integrated** district groundwater-recharge enrichment |
| `district_extraction_stage_pct` | numeric | **Integrated** district extraction-stage enrichment |
| `district_net_gw_availability_ham` | numeric | **Integrated** district net-availability enrichment |
| `firka_over_exploited_ratio` | numeric | **Integrated** firka-level over-exploitation enrichment |
| `rainfall_year_mm` | numeric | **Integrated** rainfall (district-year) enrichment |
| `river_level_year_m` | numeric | **Integrated** river-water-level (district-year) enrichment |

**Preprocessing (saved inside the model):** numeric features → median imputation + standard scaling; categorical features → constant-`unknown` imputation + one-hot encoding with `handle_unknown='ignore'`. Unknown districts and missing enrichment values are handled gracefully (imputed), so a prediction is still produced.

> Enrichment features were integrated at training time from multiple master datasets (district assessment, firka assessment, rainfall, river water level). River discharge was **excluded** during training (unreliable join coverage). These features are internal — **not** inputs the Supervisor provides.

---

## 9. Knowledge Boundaries

The Prediction Agent knows **nothing** about, and must never be asked about:

- SQL / database schema / database statistics
- PDF documents / vector database / embeddings
- Groundwater concepts, definitions, or hydrogeology
- Government policies or guidelines
- Conversation memory
- Recommendations
- Response generation / natural language
- LLMs

It knows only: **how to turn a structured groundwater-level request into a numeric prediction** using its saved model.

---

## 10. Unsupported Tasks (NEVER route these to the Prediction Agent)

| Unsupported request | Correct route |
|---|---|
| Historical / current measured groundwater values, SQL retrieval, statistics | **Data Agent** |
| Groundwater explanation, definitions, policy, concepts, "what is …" | **Knowledge Agent** |
| Document retrieval / evidence | **Knowledge Agent** |
| Recommendation generation | Recommendation stage |
| General conversation / greetings | **General Conversation LLM** |
| Intent classification, routing | Supervisor |
| Training / retraining / evaluating / updating models or datasets | **Not available at runtime** (offline pipeline only) |

---

## 11. Core Capabilities (Lookup Table)

| Capability | Supported | Route |
|---|:---:|---|
| Predict groundwater level | ✅ | **Prediction Agent** |
| Forecast future groundwater level | ✅ | **Prediction Agent** |
| Historical-year groundwater prediction | ✅ | **Prediction Agent** |
| Groundwater prediction for a district / month / year | ✅ | **Prediction Agent** |
| Explain groundwater / concepts / policy | ❌ | Knowledge Agent |
| Groundwater statistics / measured / historical data | ❌ | Data Agent |
| Recommendations | ❌ | Recommendation stage |
| Greeting / small talk | ❌ | General Conversation LLM |
| Train / retrain / evaluate models | ❌ | Offline pipeline only (not runtime) |

---

## 12. Model Coverage

| Dimension | Coverage |
|---|---|
| **Prediction target** | Groundwater level (`groundwater_level_m`) — the only trained target. |
| **Geographic coverage** | Tamil Nadu districts (model trained on Tamil Nadu groundwater observations; benchmarked across ~30 districts). Unknown districts are tolerated (imputed) but yield less specific predictions. |
| **Temporal coverage** | Any calendar year + month. Historical/near-range years are interpolative; far-future years are **extrapolations** (see Limitations). Month captures seasonality (1–12). |
| **Supported inputs** | `district`, `prediction_year`, `prediction_month`, `target` (structured request). |
| **Supported outputs** | One numeric predicted value + unit, in a structured response. |
| **Prediction units** | Metres below ground level. |
| **Prediction confidence** | Moderate (validation R² ≈ 0.48, MAE ≈ 33 m, RMSE ≈ 123 m). The model returns a **point estimate**, not a calibrated confidence interval. Treat outputs as directional estimates. |
| **Limitations** | Single target; Tamil Nadu focus; point estimate only; future years are extrapolations; not a substitute for measured data. |

---

## 13. Prediction Intent Keywords

The Supervisor Planner first receives a **natural-language query**, not structured JSON. The following vocabulary **strongly indicates** the Prediction Agent should be selected. If a query contains any of these signals **and** concerns groundwater level, route to the Prediction Agent (translate it into a structured request first).

**Prediction verbs / signals**

- predict, prediction
- forecast, forecasting
- estimate, estimated
- expected, expectation
- project, projection, projected
- future, future groundwater
- upcoming
- will be / how deep will

**Temporal phrases (future or specific-period intent)**

- next year, next month, next summer
- after five years, in five years
- by 2030, in 2028, during 2035
- future level, future groundwater level
- upcoming groundwater level

**Examples that trigger the Prediction Agent**

- "Predict groundwater level in Salem."
- "Forecast groundwater in Coimbatore."
- "Estimate groundwater level in 2029."
- "Expected groundwater level next year."
- "Projection of groundwater for Madurai."

> **Guardrail:** A prediction keyword only routes here when the *target is groundwater level*. "Predict rainfall" or "forecast river discharge" is **not** supported (only `groundwater_level_m` is trained) — do not route those here.

---

## 14. Routing Guidance

| User Query | Route To | Reason |
|---|---|---|
| "Predict groundwater level in Salem." | **Prediction Agent** | Prediction of groundwater level |
| "Forecast groundwater in Coimbatore." | **Prediction Agent** | Forecast = prediction |
| "Estimate groundwater level in Madurai in June 2028." | **Prediction Agent** | Future estimate |
| "Groundwater level in Salem today." | **Data Agent** | Current/measured value, not a forecast |
| "Average groundwater level last year." | **Data Agent** | Historical statistic from the database |
| "Explain groundwater recharge." | **Knowledge Agent** | Concept/explanation |
| "Who are you?" | **Knowledge Agent** | AquaMind AI system identity (documents) |
| "Hello." | **General Conversation LLM** | Greeting |

**Disambiguation heuristics:**

- **predict / forecast / estimate / future / next year / expected** → **Prediction Agent**.
- **current / today / measured / historical / how many / average / statistics** → **Data Agent** (not Prediction).
- **explain / define / what is / concept / policy / guideline** → **Knowledge Agent** (not Prediction).
- **greeting / thanks / chit-chat** → **General Conversation LLM**.

---

## 15. Decision Matrix (Quick Routing Lookup)

| Query | Route |
|---|---|
| Predict groundwater level. | **Prediction Agent** |
| Forecast groundwater. | **Prediction Agent** |
| Predict groundwater level in Salem for 2030. | **Prediction Agent** |
| Estimate groundwater in Coimbatore next summer. | **Prediction Agent** |
| Groundwater level today. | **Data Agent** |
| Groundwater statistics. | **Data Agent** |
| How many over-exploited firkas? | **Data Agent** |
| Explain groundwater. | **Knowledge Agent** |
| What is artificial recharge? | **Knowledge Agent** |
| Who are you? | **Knowledge Agent** |
| Hello. | **General Conversation LLM** |
| Thank you. | **General Conversation LLM** |

> When a query mixes prediction with explanation or data, route to **multiple agents in sequence** — see §17.

---

## 16. Do NOT Route Here (Explicit Negative Routing)

These queries **must never** be routed to the Prediction Agent, even if they mention groundwater. Route them to the indicated agent instead.

| Query | Do NOT route to Prediction — route to |
|---|---|
| "What is groundwater?" | **Knowledge Agent** (definition / concept) |
| "Explain artificial recharge." | **Knowledge Agent** (explanation) |
| "What are CGWB guidelines?" | **Knowledge Agent** (guideline lookup) |
| "Who are you?" / "What can AquaMind AI do?" | **Knowledge Agent** (system identity) |
| "What is the groundwater level in Salem today?" | **Data Agent** (current / measured value) |
| "Average groundwater level last year." | **Data Agent** (historical statistic) |
| "How many over-exploited firkas?" | **Data Agent** (database count) |
| "Show groundwater data for Coimbatore." | **Data Agent** (record retrieval) |
| "Hello" / "Thank you" | **General Conversation LLM** (greeting / small talk) |
| "Predict rainfall." / "Forecast river discharge." | **Not supported** — only `groundwater_level_m` is trained |

**Key distinctions the Planner must respect:**

- **"today / current / now / measured / recorded / historical / how many / average / show data"** → **Data Agent**, *not* Prediction — even if the word "groundwater level" appears.
- **"what is / explain / define / how does / why / concept / policy / guideline"** → **Knowledge Agent**, *not* Prediction.
- **A prediction verb with a non-groundwater-level target** (rainfall, river discharge, quality) → **not supported** by the Prediction Agent.

---

## 17. Multi-Agent Execution

The Prediction Agent supplies the **numeric forecast**; other agents supply data or explanation. The Supervisor issues each agent a **self-contained** request.

| User Query | Execution Plan |
|---|---|
| "Predict groundwater level in Salem for 2030 and explain why." | **Step 1 — Prediction Agent:** forecast Salem 2030 groundwater level. **Step 2 — Knowledge Agent:** retrieve passages explaining groundwater decline / recharge / extraction factors. |
| "Predict groundwater level and compare it with the current level." | **Step 1 — Prediction Agent:** forecast the level. **Step 2 — Data Agent:** retrieve the current/measured level for comparison. |
| "Predict groundwater level, compare with current level, and explain recharge methods." | **Step 1 — Prediction Agent** (forecast) → **Step 2 — Data Agent** (current measured level) → **Step 3 — Knowledge Agent** (recharge-method explanation). |

**Pattern:** Prediction Agent = the *forecast*; Data Agent = *measured/current facts*; Knowledge Agent = *explanation/definitions*.

---

## 18. Limitations

- **It predicts — it does not explain, retrieve, reason, or recommend.**
- **Single target:** only `groundwater_level_m`. No other quantity is predicted.
- **Point estimate only:** returns one number with a unit; no confidence interval is produced. R² ≈ 0.48, so treat outputs as **directional estimates**, not precise measurements.
- **Future years are extrapolations:** tree-based models do not extrapolate trends beyond the training range; far-future predictions are approximate.
- **Tamil Nadu focus:** trained on Tamil Nadu groundwater; other regions are not covered.
- **No live/measured data:** it forecasts; it does not return actual recorded values (that is the Data Agent).
- **No document knowledge:** it cannot explain or define anything (that is the Knowledge Agent).
- **No greetings, routing, intent classification, or conversation memory.**
- **No training at runtime:** it never trains, retrains, evaluates, selects, or updates models or datasets — it only performs **inference** with a previously trained, saved model.
- **Requires a structured request:** it does not parse free-text or resolve follow-ups; the Supervisor must supply `district`, `prediction_year`, `prediction_month`, and `target`.

---

**Summary for the Supervisor Planner:** Route **prediction / forecast / estimate / future groundwater-level** queries to the **Prediction Agent**. Supply a structured request (`district`, `prediction_year`, `prediction_month`, `target = groundwater_level_m`). Expect a `UniversalAgentResponse` with a single numeric predicted value in **metres below ground level** produced by the saved **XGBoost** model — never a measurement, explanation, or recommendation. Pair it with the **Data Agent** (current/measured values) or the **Knowledge Agent** (explanations) when the query needs a forecast **plus** facts or explanation.
