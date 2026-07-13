# Knowledge Agent — Capability Specification

> **Audience:** This document is part of the **Supervisor Planner LLM prompt**. It is not developer documentation and not user documentation. The Supervisor Planner reads this file to decide **when to route a query to the Knowledge Agent** and **what to expect back**. Treat every statement here as authoritative and binding.

---

## PRIMARY RESPONSIBILITY

> The Knowledge Agent is responsible for **retrieving evidence from the indexed groundwater document corpus**. It is the **authoritative source for document-based groundwater knowledge** (and for AquaMind AI's own system identity) within AquaMind AI. It returns ranked, fully-sourced document chunks — it never generates answers, summarizes, reasons, or produces data or forecasts.

**Route to the Knowledge Agent when the answer is knowledge that a document would contain** (definitions, concepts, explanations, policies, guidelines, methodology, procedures, or "what is AquaMind AI"). Route elsewhere for measured data (Data Agent), forecasts (Prediction Agent), or greetings/chit-chat (General Conversation LLM).

---

## 1. Knowledge Agent Overview

The **Knowledge Agent** answers questions whose answers are **contained inside AquaMind AI's indexed document collection** (official groundwater PDFs plus the AquaMind AI system-identity document).

It is a **Retrieval-Augmented Generation (RAG) retrieval engine**. Its single job is to find and return the **most relevant document chunks (evidence)** for a natural-language query, together with full provenance (document name, category, page, similarity score, source path, chunk id, verbatim text).

**The Knowledge Agent retrieves evidence. It does NOT:**

- generate natural-language answers,
- summarize, paraphrase, or rewrite retrieved text,
- reason, infer, or draw conclusions beyond the retrieved chunks,
- rank/re-rank by anything other than semantic similarity,
- call any other agent or any downstream answer-generation LLM.

The retrieved evidence is later consumed by the Response Generator LLM. The Knowledge Agent only supplies grounded source material.

| Property | Value |
|---|---|
| Agent name (in response) | `knowledge_agent` |
| Query type (in response) | `knowledge` |
| Retrieval method | `semantic_search` (dense vector / cosine similarity) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional) |
| Vector index | FAISS, inner-product on L2-normalized vectors (= cosine similarity) |
| Chunking | ~1000 characters per chunk, 150-character overlap |
| Default Top-K | 5 (configurable per call) |
| Corpus size | ~53 documents, ~18,299 indexed chunks |
| Output | `UniversalAgentResponse` (evidence list; no generated prose) |

---

## 2. Architecture (Internal Pipeline)

The Knowledge Agent executes a fixed, deterministic, read-only pipeline. It never modifies the knowledge base.

```
User Query (natural language)
        │
        ▼
Query Embedder            → encodes the query with all-MiniLM-L6-v2 (same model used at indexing), L2-normalized
        │
        ▼
FAISS Retriever           → cosine-similarity search over the vector index; returns Top-K (embedding_id, score) hits
        │
        ▼
Metadata Resolver         → maps each hit to its metadata (document, category, page, section, source_path, chunk_id)
        │                     and deterministically reconstructs the exact chunk text
        ▼
Retrieved Chunks          → ordered best-first by similarity score
        │
        ▼
Knowledge Formatter       → wraps chunks into structured evidence (pure transformation; no LLM, no edits)
        │
        ▼
UniversalAgentResponse    → { agent_name, status, query_type, total_evidence, retrieval_method, evidence[] }
```

**Pipeline guarantees:**

- **Deterministic:** identical query + identical Top-K → identical evidence.
- **Verbatim:** chunk text is returned exactly as it appears in the source document; nothing is altered, summarized, or generated.
- **Ordered:** evidence is sorted by descending similarity score (most relevant first).
- **Fail-safe:** if nothing relevant is found, it returns `status = NO_RESULTS` with an empty evidence list — it never fabricates content.

---

## 3. Knowledge Sources (Indexed Document Categories)

Every document is grouped by a **category** equal to its source folder. The following categories currently exist in the index. **Do not assume any category that is not listed here.**

| Category | Description | Representative Documents |
|---|---|---|
| `aquifer mapping` | Aquifer mapping & management studies (NAQUIM) for Tamil Nadu aquifer systems and river basins. Largest category. | Aquifer Mapping & Management of Palar / Lower Cauvery / Vaigai / Tambraparni / Parambikulam-Aliyar / Gundar / Nambiyar / Vaippar / Varahanadi / Agniyar / Kodayar aquifer systems; Amaravathi, Bhavani, Chennai, Upper Cauvery, Upper Ponnaiyar basins; firka-level studies (Chengam, Coimbatore South, Karumathampatty, Pothaneri, Valappadi, Ammankoil, etc.) |
| `aquifer management` | District/block aquifer management plans and NAQUIM 2.0 studies for stressed/over-exploited areas. | Aquifer Management Plans for Tiruppur, Krishnagiri, Namakkal & Salem, Ariyalur; NAQUIM 2.0 (Salem blocks, Theni); Lower Vellar watershed pilot |
| `resources assessment` | Dynamic groundwater resource estimation (GEC-2015 based) for Tamil Nadu. | Dynamic Ground Water Resources of Tamil Nadu 2024 & 2025; Dynamic GW Resources + GW Level + GW Quality reports; 2023 status summary |
| `groundwater quality` | Groundwater quality reports, yearbooks and parameter analysis. | State Report on Groundwater Quality in Tamil Nadu & Puducherry; Annual GW Quality Report 2023-24; GW Quality Year Book 2024-25; Annexure |
| `year book` | Groundwater year books (water levels, trends, monitoring). | Ground Water Year Book of Tamil Nadu 2023-24; Groundwater Year Book, Tamil Nadu & UT of Puducherry 2024-25 |
| `faq` | Frequently-asked-question and regulation booklets. | CGWB FAQ; Rainwater Harvesting FAQ; Ground Water Regulations |
| `artificial recharge` | Artificial recharge / rainwater-harvesting planning. | District Recharge Plan of Tamil Nadu |
| `groundwater modelling` | Numerical groundwater flow simulation studies. | Groundwater Flow Simulation & Aquifer Management Plan for Chennai Aquifer System |
| `groundwater_books` | General groundwater science reference book. | Groundwater (web book) — concepts, hydrogeology, budgets, management |
| `policy and guidelines` | Institutional agreements / policy documents. | MOU between CGWB and Anna University |
| `pdf` | Root-level technical manual (folder is the PDF root). | GEC-2015 Guidelines (Ground Water Resource Estimation methodology) |
| `system` | **AquaMind AI system-identity documentation.** Source of truth for "who/what is AquaMind AI". | AquaMind AI System Identity |

**Geographic & thematic scope of the corpus:** Tamil Nadu (and UT of Puducherry) groundwater — aquifer systems, districts, blocks, firkas, river basins; groundwater assessment, quality, recharge, monitoring, policy, and general hydrogeology concepts; plus AquaMind AI's own system identity.

### 3.1 Knowledge Coverage (topics the corpus authoritatively covers)

If a query concerns any of the following topics, the Knowledge Agent is the correct source:

- Hydrogeology
- Aquifers
- Groundwater (general concepts)
- Groundwater Quality
- Groundwater Assessment (GEC-2015 resource estimation)
- Groundwater Monitoring
- Artificial Recharge
- Rainwater Harvesting
- Aquifer Mapping (NAQUIM)
- CGWB Publications
- Government Guidelines
- Policies & Regulations
- Year Books
- Tamil Nadu Groundwater Reports
- Groundwater Modelling
- Groundwater Sustainability
- Groundwater Management
- Over-exploited / Critical Areas (concepts & definitions)
- AquaMind AI System Identity

> These topics **belong to the Knowledge Agent**. Numeric values, records, or forecasts *about* these topics still belong to the Data Agent or Prediction Agent respectively — the Knowledge Agent supplies the **explanatory / conceptual** content.

---

## 4. Supported Question Types

The Knowledge Agent supports **conceptual, explanatory, definitional, policy, guideline, and document-grounded** questions whose answers live in the indexed PDFs. Examples (non-exhaustive):

- **Definitions & concepts:** "What is groundwater?", "What is an aquifer?", "What is groundwater recharge?", "Explain hydrogeology.", "What is specific yield / water table?"
- **Quality & contamination:** "Explain groundwater contamination.", "What parameters indicate groundwater quality?", "What causes salinity ingress?"
- **Assessment methodology:** "Explain the GEC-2015 methodology.", "How is annual extractable groundwater resource assessed?", "What are safe / semi-critical / critical / over-exploited categories?"
- **Recharge & conservation:** "What is artificial recharge?", "Explain rainfall recharge.", "Explain river / stream recharge.", "What recharge structures exist (percolation tanks, check dams, recharge shafts)?"
- **Policy, regulation & guidelines:** "What are CGWB guidelines?", "What regulations govern groundwater extraction?", "Explain rainwater-harvesting mandates."
- **Management & sustainability:** "Explain groundwater management strategies.", "What are over-exploited blocks?", "Explain sustainable groundwater management."
- **Monitoring & hydrology:** "How is groundwater level monitored?", "Explain observation wells / piezometers.", "What is aquifer mapping?"
- **System identity (from the `system` document):** "Who are you?", "What is AquaMind AI?", "What can AquaMind AI do?"

**Rule of thumb:** if the question asks to **explain, define, describe, or reference knowledge that a document would contain**, it is a Knowledge Agent question.

---

## 5. Supported Retrieval Tasks

| Task | Description |
|---|---|
| Definition lookup | Retrieve passages defining a term or concept. |
| Concept explanation | Retrieve passages explaining a groundwater topic. |
| Policy lookup | Retrieve policy / regulation passages. |
| Guideline lookup | Retrieve CGWB / GEC / procedural guideline passages. |
| Technical explanation | Retrieve methodology / technical-manual passages. |
| Document-based Q&A | Retrieve passages that answer a document-grounded question. |
| Best practices / procedures | Retrieve recommended practices or government procedures described in documents. |
| System-identity lookup | Retrieve passages describing AquaMind AI itself. |

Anything **directly supported by the indexed PDFs** is in scope. Anything requiring computation, live data, prediction, or reasoning beyond the text is **out of scope** (see Section 6).

---

## 6. Unsupported Tasks (NEVER route these to the Knowledge Agent)

The Knowledge Agent **must not** be selected for:

- **Structured data / statistics** — numeric values, records, counts, aggregations, per-station/district/firka figures, historical time-series lookups → **Data Agent**.
- **SQL / database queries** of any kind → **Data Agent**.
- **Machine-learning prediction / numerical forecasting** (future groundwater levels, projections) → **Prediction Agent**.
- **Recommendation generation** → Recommendation stage (not this agent).
- **General conversation / greetings / small talk** ("Hi", "Thanks", "How are you?") → **General Conversation LLM**.
- **Intent classification, routing, response generation, conversation memory** — these are Supervisor / other-component responsibilities, not the Knowledge Agent.
- **Reasoning or answers beyond the retrieved evidence** — the Knowledge Agent only returns source text; it never concludes.
- **Any information not contained in the indexed documents** — it cannot answer about topics outside the corpus and will return `NO_RESULTS`.

---

## 7. Expected Inputs

| Input | Required | Description |
|---|---|---|
| `query` | Yes | A natural-language question or topic string. |
| `top_k` | No | Number of chunks to retrieve (default 5). Increase for broader evidence, decrease for precision. |
| Retrieved context / conversation context | No | If the Supervisor has resolved follow-ups (e.g., replaced a pronoun or reused the current topic), it should pass a **fully-formed, self-contained query**. The Knowledge Agent performs **no** follow-up resolution itself. |

**Important:** The Knowledge Agent does not read conversation memory. The Supervisor must rewrite context-dependent queries (e.g., "explain that" → "explain over-exploited blocks") **before** calling it.

---

## 8. Expected Outputs

A single `UniversalAgentResponse` object:

```json
{
  "agent_name": "knowledge_agent",
  "status": "SUCCESS",              // or "NO_RESULTS"
  "query_type": "knowledge",
  "total_evidence": 5,
  "retrieval_method": "semantic_search",
  "evidence": [
    {
      "chunk_id": "…",
      "document": "…​.pdf",
      "category": "aquifer mapping",
      "page": 12,
      "section": "…",              // may be null
      "source_path": "pdf/…/…​.pdf",
      "similarity_score": 0.78,      // cosine similarity, higher = more relevant
      "content": "…verbatim chunk text…"
    }
    // …up to Top-K items, ordered best-first
  ]
}
```

| Field | Meaning |
|---|---|
| `status` | `SUCCESS` when ≥1 chunk found; `NO_RESULTS` when none. Never raises. |
| `total_evidence` | Number of evidence items returned. |
| `evidence[].similarity_score` | Cosine similarity in ~[0, 1]; higher = closer match. Use to judge confidence. |
| `evidence[].content` | Verbatim source text — safe to ground a generated answer on. |
| `evidence[].document` / `page` / `source_path` | Provenance for citation. |

---

## 9. Core Capabilities

### 9.1 Capability Lookup Table

| Capability | Supported | Route |
|---|:---:|---|
| Explain groundwater concepts | ✅ | **Knowledge Agent** |
| Explain policies | ✅ | **Knowledge Agent** |
| Explain guidelines (CGWB / GEC) | ✅ | **Knowledge Agent** |
| Explain aquifer mapping | ✅ | **Knowledge Agent** |
| Explain groundwater quality | ✅ | **Knowledge Agent** |
| Explain groundwater assessment / methodology | ✅ | **Knowledge Agent** |
| Explain artificial recharge / rainwater harvesting | ✅ | **Knowledge Agent** |
| Explain groundwater management / sustainability | ✅ | **Knowledge Agent** |
| Explain hydrogeology / monitoring | ✅ | **Knowledge Agent** |
| Explain AquaMind AI (identity / capabilities) | ✅ | **Knowledge Agent** |
| Retrieve groundwater measurements / statistics | ❌ | Data Agent |
| Historical database records / counts | ❌ | Data Agent |
| Predict / forecast future groundwater | ❌ | Prediction Agent |
| Generate recommendations | ❌ | Recommendation stage |
| Greeting / small talk | ❌ | General Conversation LLM |

### 9.2 Capability Descriptions

- **Semantic retrieval:** finds relevant passages by meaning, not keyword matching (handles paraphrase and synonymy).
- **Full provenance:** every result carries document, category, page, section, source path, and chunk id — ready for citation.
- **Confidence signal:** similarity scores let the Supervisor / downstream LLM judge how well the corpus covers the query.
- **Multi-domain coverage:** aquifer mapping & management, resource assessment (GEC-2015), quality, recharge, policy, guidelines, year books, hydrogeology, modelling, and AquaMind AI system identity.
- **Configurable breadth:** Top-K controls how much evidence is returned.
- **Deterministic & non-destructive:** repeatable results; the knowledge base is read-only.
- **Grounded-only:** returns nothing when the corpus lacks the answer, preventing hallucinated sourcing.

---

## 10. Routing Guidance (for the Supervisor Planner)

Route to the **Knowledge Agent** when the answer is **explanatory / definitional / policy / guideline / concept / document-grounded**.

| User Query | Route To | Reason |
|---|---|---|
| "What is groundwater?" | **Knowledge Agent** | Definition from documents |
| "What is artificial recharge?" | **Knowledge Agent** | Concept from documents |
| "Explain groundwater quality." | **Knowledge Agent** | Explanatory, document-grounded |
| "Explain the GEC-2015 methodology." | **Knowledge Agent** | Methodology in manuals |
| "What are CGWB guidelines for rainwater harvesting?" | **Knowledge Agent** | Guideline lookup |
| "What are over-exploited blocks?" | **Knowledge Agent** | Concept/definition from documents |
| "Who are you?" | **Knowledge Agent** | Answered from the `system` (AquaMind AI System Identity) document |
| "What can AquaMind AI do?" | **Knowledge Agent** | Answered from the `system` document |
| "What is the groundwater level in Salem?" | **Data Agent** | Structured numeric data lookup |
| "How many over-exploited firkas are there?" | **Data Agent** | Count/statistic from the database |
| "Predict groundwater level in 2030." | **Prediction Agent** | Numerical forecasting |
| "Hello" / "Thanks" | **General Conversation LLM** | Greeting / small talk |

**Disambiguation heuristics:**

- Mentions of **explain / define / describe / what is / concept / policy / guideline / methodology / procedure** → Knowledge Agent.
- Mentions of **specific numeric values, a specific district/firka/station's measured data, counts, historical records** → Data Agent (not Knowledge Agent).
- Mentions of **predict / forecast / future year / projection / expected level** → Prediction Agent (not Knowledge Agent).
- Questions about **AquaMind AI itself** → Knowledge Agent (`system` document).

### 10.1 Decision Matrix (quick routing lookup)

| Query | Route |
|---|---|
| What is groundwater? | **Knowledge** |
| Explain groundwater recharge | **Knowledge** |
| Explain groundwater contamination | **Knowledge** |
| Explain aquifer | **Knowledge** |
| Explain GEC-2015 | **Knowledge** |
| What are CGWB guidelines? | **Knowledge** |
| Explain over-exploited blocks | **Knowledge** |
| Explain artificial recharge | **Knowledge** |
| Who are you? | **Knowledge** |
| What can AquaMind AI do? | **Knowledge** |
| Groundwater level in Salem | **Data** |
| Average groundwater level | **Data** |
| How many over-exploited firkas? | **Data** |
| Predict groundwater level in 2030 | **Prediction** |
| Forecast rainfall impact | **Prediction** |
| Hi | **General LLM** |
| Thank you | **General LLM** |

> When a query combines two needs (e.g., *"predict … and explain …"*), route to **both** agents in sequence — see Section 11.

---

## 11. Multi-Agent Examples

The Knowledge Agent frequently runs **alongside** another agent: the other agent supplies data or a prediction, and the Knowledge Agent supplies the explanatory evidence. The Supervisor issues each a **self-contained** query.

| User Query | Execution Plan |
|---|---|
| "Compare groundwater level in Salem and explain why it decreased." | **Step 1 — Data Agent:** retrieve Salem groundwater level data. **Step 2 — Knowledge Agent:** retrieve passages explaining causes of groundwater decline / over-exploitation. |
| "Predict groundwater level in Coimbatore and explain possible causes." | **Step 1 — Prediction Agent:** forecast Coimbatore groundwater level. **Step 2 — Knowledge Agent:** retrieve passages on factors affecting groundwater levels / recharge / extraction. |
| "Show over-exploited blocks and explain what over-exploitation means." | **Step 1 — Data Agent:** list over-exploited assessment units. **Step 2 — Knowledge Agent:** retrieve the definition/explanation of "over-exploited". |
| "Forecast 2030 groundwater and describe recommended recharge measures." | **Step 1 — Prediction Agent:** forecast. **Step 2 — Knowledge Agent:** retrieve recharge best-practice / structure passages. |

**Pattern:** the Knowledge Agent handles the **"explain / define / describe"** portion; the Data or Prediction Agent handles the **numeric / forecast** portion.

---

## 12. Limitations (do not overstate capabilities)

- **No answer generation:** returns evidence chunks only — never a written answer, summary, or conclusion.
- **No reasoning or inference** beyond the retrieved text; it cannot synthesize across documents or compute anything.
- **No live or structured data:** cannot return measured values, statistics, counts, or database records (that is the Data Agent).
- **No prediction:** cannot forecast or project future values (that is the Prediction Agent).
- **Corpus-bounded:** can only surface content that exists in the indexed PDFs. Out-of-corpus topics → `NO_RESULTS`.
- **Tamil Nadu / Puducherry focus:** coverage reflects the indexed Tamil Nadu (and Puducherry) groundwater documents plus AquaMind AI system identity; other regions are not covered.
- **No follow-up resolution:** does not use conversation memory; the Supervisor must pass fully-resolved, self-contained queries.
- **Retrieval quality depends on phrasing and corpus coverage:** low similarity scores signal weak coverage; a `NO_RESULTS` or low-score result means the corpus likely does not contain the answer.
- **Not for greetings, chit-chat, routing, or recommendations.**

---

**Summary for the Supervisor Planner:** Route explanatory, definitional, conceptual, policy, guideline, methodology, and AquaMind-AI-identity questions to the **Knowledge Agent**. Expect a `UniversalAgentResponse` containing ranked, fully-sourced evidence chunks — never a finished answer. Pair it with the Data Agent (for measured data) or the Prediction Agent (for forecasts) when a query needs both facts/forecasts **and** explanation.
