# NeoStats Credit Risk Platform - Implementation Plan

## Goal

Build a demo-ready, Dockerized credit-risk platform within 10 hours using:

- Streamlit for the five-section user interface
- PostgreSQL for curated analytics and conversation checkpoints
- Logistic Regression as the baseline model
- LightGBM as the shadow model and feature-pruning mechanism
- Explainable Boosting Machine (EBM) as the non-negotiable production model
- LangGraph and Gemini for the agentic chatbot
- Three chatbot tools: PostgreSQL analytics, knowledge-base search, and DDGS web search

## Locked decisions

| Area | Decision |
|---|---|
| Runtime | Python 3.11 |
| Hardware | Train locally on 16 GB RAM; do not spend time configuring the 4 GB GPU |
| Data split | Stratified 70% train, 15% validation, 15% final test |
| Production model | EBM |
| Feature pruning | Shadow LightGBM selects approximately 25-30 raw features |
| Explainability | Native EBM global/local explanations plus sampled LightGBM SHAP |
| Risk bands | Validation-score percentiles: Low < P60, Medium P60-P85, High >= P85 |
| Database | PostgreSQL with one-row-per-applicant analytics tables/views |
| UI | Streamlit with five tabs |
| UI design reference | Streamlit Stock Peer Analysis dashboard: dark navy theme, compact controls, metric cards, bordered interactive charts, and wide layout |
| Default LLM | Gemini 2.5 Flash |
| Complex LLM | Gemini 2.5 Flash, configurable independently if required |
| Web search | DDGS; maximum five sourced results |
| Deployment | Docker Compose with PostgreSQL, one-shot data loader, and Streamlit app |
| Canonical EDA | `notebooks/Home_Credit_Default_Risk_EDA.ipynb` - complete; filename remains unchanged |

Risk bands are internal model-risk segments, not RBI-defined ratings. The UI must label them as model-estimated payment-difficulty risk bands.

## Confirmed EDA findings

These findings come from the executed canonical notebook and must be reused consistently in the Streamlit EDA page, knowledge base, README, and presentation:

1. The portfolio default rate is 8.1%, an approximately 11.4:1 majority-to-minority imbalance. Model selection must prioritize ROC-AUC, PR-AUC, calibration, and threshold metrics rather than accuracy.
2. Applicants aged 20-29 have the highest age-band default rate at 11.4%, compared with the 8.1% portfolio average.
3. Among the demographic segments assessed with adequate support, applicants living in rented apartments have the highest observed lift at 1.53x, based on 4,881 applicants.
4. Credit-to-income and annuity-to-income ratios separate repayment outcomes more clearly than raw credit or income values, supporting affordability-based engineered features.
5. `EXT_SOURCE_3` is the strongest of the three external scores and must be retained with explicit missing-value treatment.
6. Prior repayment behaviour - overdue bureau records, previous refusals, and late instalments - provides stronger risk signal than demographic characteristics and justifies the history aggregation work.

These are associations, not causal or regulatory conclusions. Demographic attributes must be used for fairness diagnostics and must not be translated directly into adverse-action rules.

## Progress

- [x] Professional EDA completed and executed
- [x] Repository setup
- [x] Applicant-level feature pipeline
- [x] Logistic Regression baseline
- [x] Shadow LightGBM and feature pruning
- [x] Production EBM
- [x] Evaluation, explanations, and surrogate rules
- [x] PostgreSQL analytics layer (live Docker/pgvector load and access controls verified)
- [x] Three-tool agentic chatbot with PostgreSQL checkpoint memory
- [x] Five-tab Streamlit application with Chatbot as the landing tab
- [x] Docker Compose deployment
- [x] Gold-path testing (33 automated tests)
- [x] README and presentation PDF

### Measured ML snapshot

| Model | ROC-AUC | PR-AUC | Brier score | Role |
|---|---:|---:|---:|---|
| Logistic Regression | 0.759 | 0.238 | 0.199 | Interpretable baseline |
| LightGBM | 0.778 | 0.265 | 0.174 | Shadow ceiling and feature pruning |
| Calibrated EBM | 0.764 | 0.250 | 0.068 | Production model |

The EBM was trained on a stratified 120,000-row sample using the top 30 LightGBM features. On the untouched test set, observed default rates increase monotonically from 3.3% (Low) to 9.8% (Medium) and 24.3% (High). The surrogate tree has 68.0% risk-band agreement and is presented only as an approximation of the EBM.

## Ten-hour schedule

| Time | Section | Required output |
|---|---|---|
| 0:00-0:30 | Project setup | Final folders, configuration, requirements, EDA notebook moved |
| 0:30-1:15 | Data pipeline | Applicant-level features, history aggregates, stratified split |
| 1:15-1:35 | Logistic Regression | Balanced baseline and metrics |
| 1:35-1:55 | LightGBM | Shadow model, metrics, and top 25-30 features |
| 1:55-2:45 | EBM | Trained, reloadable production model |
| 2:45-3:30 | Evaluation | Comparison table, risk bands, explanations, surrogate rules |
| 3:30-4:45 | PostgreSQL | Analytics schema, indexes, read-only role, memory tables |
| 4:45-6:30 | Chatbot | Three tools, routing, SQL safety, answer synthesis, memory |
| 6:30-7:45 | Streamlit | EDA, Prediction, Explainability, Rules, Chatbot tabs |
| 7:45-8:45 | Docker | One-command startup and automated database initialization |
| 8:45-9:30 | Testing | Model, SQL accuracy/safety, memory, and Docker tests |
| 9:30-10:00 | Submission | README, screenshots, presentation PDF, GitHub cleanup |

## Planned repository structure

```text
credit_risk_platform/
|-- app.py
|-- data/                         # mounted, never committed
|-- documents/
|   `-- project_presentation.pdf
|-- notebooks/
|   `-- Home_Credit_Default_Risk_EDA.ipynb
|-- artifacts/
|   |-- eda/
|   |-- evaluation/
|   `-- rules/
|-- models/
|   |-- logistic_pipeline.joblib
|   |-- lightgbm_shadow.joblib
|   |-- ebm_bundle.joblib
|   |-- feature_manifest.json
|   `-- risk_thresholds.json
|-- src/
|   |-- data/
|   |   |-- loader.py
|   |   `-- preprocessor.py
|   |-- ml/
|   |   |-- train.py
|   |   |-- predict.py
|   |   |-- evaluate.py
|   |   `-- explain.py
|   |-- rules/
|   |   `-- derive_rules.py
|   |-- db/
|   |   |-- connection.py
|   |   |-- load_data.py
|   |   `-- knowledge_base.py
|   |-- talk_to_data/
|   |   |-- graph.py
|   |   |-- router.py
|   |   |-- nl_to_sql.py
|   |   |-- sql_validator.py
|   |   |-- query_runner.py
|   |   |-- knowledge_search.py
|   |   |-- web_search.py
|   |   `-- prompt_templates.py
|   |-- ui/
|   |   |-- eda.py
|   |   |-- prediction.py
|   |   |-- explainability.py
|   |   |-- rules.py
|   |   `-- chatbot.py
|   `-- utils/
|       |-- config.py
|       |-- logger.py
|       `-- helpers.py
|-- sql/
|   |-- 001_extensions_roles.sql
|   |-- 002_schema.sql
|   |-- 003_analytics_views.sql
|   `-- 004_indexes.sql
|-- scripts/
|-- tests/
|   |-- unit/
|   |-- integration/
|   `-- gold_questions.yaml
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md
```

## ML deliverables

### Feature pipeline

- Preserve exactly one row per `SK_ID_CURR`.
- Create age, affordability, credit-to-goods, employment, and anomaly features.
- Aggregate bureau, previous-application, and instalment history.
- Distinguish no matched history from genuine zero values.
- Fit imputers, encoders, scalers, and selectors using training data only.
- Save feature names, types, package versions, seed, and risk thresholds.

### Models

1. Fit balanced Logistic Regression as the interpretable floor.
2. Fit balanced LightGBM on the training data and select 25-30 raw features.
3. Fit EBM on the selected features using balanced sample weights.
4. If EBM is too slow, use a stratified 100-120k training sample, then reduce interactions and outer bags if necessary. Do not remove EBM.

### Evaluation and explanation

- Compare ROC-AUC, PR-AUC, Brier score, calibration, and training time.
- Verify monotonically increasing observed default rate from Low to High risk.
- Produce EBM global and applicant-level explanations.
- Produce a sampled LightGBM SHAP summary.
- Train a depth 3-4 surrogate tree and export readable rules with support, default rate, and fidelity.

## Chatbot tools

The chatbot must expose exactly these three tools for the core demo:

1. `query_database`
   - Runs validated, read-only PostgreSQL analytics queries.
   - Uses curated applicant-level objects by default.
   - Returns a maximum of 200 rows.

2. `search_knowledge_base`
   - Retrieves official feature definitions, confirmed EDA findings, EBM rationale, and validated business rules.
   - Never contains raw applicant records.

3. `search_web`
   - Uses DDGS for general credit-risk concepts unavailable internally.
   - Returns at most five results with source URLs.
   - Never answers questions about internal portfolio data.
   - Fails gracefully when external search is unavailable.

## Chatbot performance and safety

- Use deterministic routing for obvious feature-definition questions.
- Use Gemini structured output for all other routing decisions.
- Use no more than two LLM calls in a normal turn.
- Permit no more than two tool executions and one SQL repair attempt.
- Parse SQL with `sqlglot`.
- Accept one `SELECT` or `WITH ... SELECT` statement only.
- Reject DDL, DML, comments, multiple statements, unapproved tables, and unsafe functions.
- Execute through a read-only PostgreSQL role with statement timeout and row limit.
- Refuse to guess when tool evidence is empty.
- Show the answer, result table/chart, tool used, assumptions, sources, and expandable SQL.
- Persist LangGraph checkpoints in PostgreSQL using a stable session ID.

## Streamlit tabs

### Visual design direction

Use the [Streamlit Stock Peer Analysis dashboard](https://demo-stockpeers.streamlit.app/) as the visual reference, without attempting a pixel-for-pixel copy:

- Dark navy theme with restrained blue, teal, amber, and coral accents.
- Wide page layout with a compact filter/control panel beside a dominant chart.
- Bordered containers, concise metric cards, pill-style controls, clear legends, and informative tooltips.
- Native Streamlit and Plotly/Altair components wherever possible; avoid extensive custom HTML.
- Provide fullscreen, show-data, and downloadable-table affordances where they materially help the demo.
- Limit each tab to two or three decision-relevant visuals instead of long sequences of repeated charts.
- Load precomputed chart datasets and cache expensive resources so interactions stay responsive.

Expected incremental implementation time: approximately 30-45 minutes over a basic Streamlit interface.

1. **Chatbot** - conversation, evidence, SQL, sources, and result visualization.
2. **EDA** - precomputed metrics, selected charts, and business findings.
3. **Risk Prediction** - applicant input, probability, percentile, and Low/Medium/High band.
4. **Explainability** - EBM global and local explanations.
5. **Rules** - validated surrogate-tree rules and their fidelity.

The Streamlit app must load precomputed artifacts and cached resources. It must not scan or aggregate the full raw CSV dataset during a rerun.

## Minimum verified chatbot questions

1. What is the overall default rate?
2. Which age band has the highest default rate?
3. Compare default rates across education levels.
4. How does previous late-payment behaviour relate to default?
5. What does `EXT_SOURCE_3` mean?
6. Why was EBM selected?
7. What is probability of default in general? Use external sources.
8. A follow-up question that depends on the previous turn.
9. A destructive SQL request that must be rejected.

## Definition of done

- All three models train and their metrics are saved.
- The EBM bundle reloads and produces a probability, risk band, and explanation.
- Risk bands are monotonic and evaluated on the untouched test set.
- PostgreSQL preserves one row per applicant in the main analytics object.
- All three chatbot tools work and identify their evidence sources.
- At least five SQL questions return independently verified values.
- Unsafe SQL is rejected and empty evidence is not hallucinated.
- Conversation memory survives a Streamlit refresh.
- All five Streamlit tabs work end to end.
- `docker compose up --build` initializes and launches the demo.
- README instructions pass a fresh-start test.
- The final presentation PDF is stored under `documents/`.

## Cut line if time is lost

Cut these first:

1. Importing POS/cash and credit-card raw history into PostgreSQL
2. Hypothetical scoring through the chatbot
3. CSV downloads
4. Authentication
5. Extensive UI styling
6. Broad hyperparameter tuning

Do not cut EBM, the three chatbot tools, SQL validation, conversation memory, saved artifacts, the five Streamlit tabs, Docker startup, README, or the presentation.
