# NeoStats Credit Risk Intelligence Platform - Presentation Source Pack

> Purpose: This is an intentionally detailed information dump for generating a polished 12-15 slide presentation. It is not intended to be copied word-for-word onto slides. Use the slide headlines, recommended visuals, and speaker notes to create a concise deck; use the appendices as the factual source of truth.

## Instructions for the presentation designer

- Target length: 15 slides. For a 12-slide version, combine Slides 2 and 3, Slides 8 and 9, and Slides 12 and 13.
- Target speaking time: 10-12 minutes, followed by a live demo and questions.
- Audience: AI engineering interview panel with interest in data science, explainability, LLM engineering, safety, and deployment.
- Narrative: business problem -> evidence from data -> modeling choices -> explainability -> agentic analytics -> engineering controls -> working product -> limitations and roadmap.
- Visual style: professional financial analytics; dark navy or white background, restrained cyan/teal accents, high contrast, wide layouts, minimal decorative elements.
- Use screenshots from the deployed Streamlit application where specified. Avoid screenshots containing secrets, local file paths, database passwords, or API keys.
- Use the architecture image at `docs/images/credit-risk-platform-architecture.png` on the architecture slide.
- Keep claims precise. Say “model-estimated payment-difficulty risk” rather than “creditworthiness” or “approval decision.”
- Treat observed segment differences as associations, not causal effects.
- Do not claim RBI-defined thresholds, regulatory approval, production validation, fairness certification, or automated lending authority.
- Explain that EBM is the deployed decision-support model, while LightGBM is a shadow benchmark and feature-pruning model.
- When comparing Brier scores, disclose that the EBM probability was Platt-calibrated and the stored baseline/shadow scores were not subjected to the same calibration step.

## Core project identity

- Project title: **AI-Powered Credit Risk Intelligence Platform**
- Product name: **NeoStats Credit Risk Intelligence Platform**
- Dataset: Home Credit Default Risk, Kaggle competition dataset
- GitHub: `https://github.com/sanmitra-b/CreditRisk`
- Live application: `https://creditrisk-up7sbn4kyskwhbrnrskw2u.streamlit.app/`
- Primary outcome: estimate the probability that an applicant experiences payment difficulty, assign an internal Low/Medium/High risk band, explain the prediction, expose business-readable surrogate rules, and answer natural-language questions using controlled data, knowledge, and web tools.
- Production model: calibrated Explainable Boosting Machine (EBM)
- Shadow model: LightGBM
- Baseline: Logistic Regression
- Interface: Streamlit, five sections with the Agentic Assistant first
- Database: PostgreSQL 17 with pgvector; hosted deployment uses Supabase PostgreSQL
- Agent orchestration: LangGraph
- LLM integration: Gemini through `google-genai`
- Web retrieval: DDGS/DuckDuckGo Search
- Local reproducible runtime: Conda environment `NeoStatsCredit`, Python 3.11
- Container runtime: Python 3.11 slim, Docker Compose

---

# Recommended 15-slide story

## Slide 1 - Title and one-line value proposition

### Suggested title

**NeoStats Credit Risk Intelligence Platform**

### Suggested subtitle

**Explainable default-risk scoring and safe agentic analytics on 307,511 loan applicants**

### Minimal on-slide content

- Calibrated EBM prediction
- PostgreSQL + pgvector analytics
- LangGraph + Gemini assistant
- Streamlit application and Dockerized delivery

### Recommended visual

Use a clean hero screenshot of the application landing page or a three-part visual showing “Predict,” “Explain,” and “Ask.” Keep this slide sparse.

### Speaker notes

This project covers the complete AI engineering path required by the assignment: EDA, feature engineering, model training, probability calibration, explanations, decision-rule approximation, natural-language analytics, SQL safety, UI, tests, and deployment. The intended user is a risk analyst or model reviewer. The system supports decisions; it does not approve or decline loans automatically.

---

## Slide 2 - Business problem and project objective

### Headline

**Credit teams need speed, discrimination, and explanations in the same workflow**

### Business context

- Credit risk teams must identify applicants with elevated repayment risk early.
- A useful score needs probability quality and ranking power, but regulated decision environments also require traceability.
- Business analysts often depend on technical teams for portfolio questions; a controlled talk-to-data layer reduces that delay.
- Predictive output, supporting evidence, and transparent rules should be accessible in one interface.

### Project objective

Build a lightweight end-to-end platform that:

1. Profiles the portfolio and exposes reliable EDA findings.
2. Predicts the probability of `TARGET = 1`, meaning payment difficulty on the current loan.
3. Converts the probability into an internal 0-1000 risk score and Low/Medium/High band.
4. Explains global behavior and individual predictions.
5. Provides readable surrogate rules as an approximation of the EBM.
6. Lets users ask data questions in natural language and receive evidence-backed answers.
7. Runs locally through Docker Compose and is deployable as a Streamlit web app.

### Recommended visual

A left-to-right flow: applicant/data -> risk probability -> explanation/risk band -> analyst action. Add a second line for analyst question -> safe tool -> answer with evidence.

### Speaker notes

The assignment explicitly tests the full stack rather than a leaderboard model. The platform therefore optimizes for an auditable, demo-ready workflow. Predictive quality is one part of the scorecard; LLM capability, hallucination control, EDA, engineering, and documentation together account for 70% of the stated evaluation weight.

---

## Slide 3 - Requirements and delivered scope

### Headline

**All six required modules are implemented end to end**

| Assignment requirement | Delivered implementation | Evidence |
|---|---|---|
| Data understanding and EDA | Executed professional notebook; dataset summary, quality checks, feature categorization, six confirmed findings and charts | `notebooks/Home_Credit_Default_Risk_EDA.ipynb` |
| Talk-to-data | LangGraph agent with database, knowledge-base, and web-search tools; readable answer synthesis | `src/talk_to_data/` |
| Machine learning | Logistic baseline, LightGBM shadow model, calibrated EBM production model; saved artifacts | `src/ml/`, `models/`, `artifacts/evaluation/` |
| Explainable AI | Native EBM global/local contributions plus sampled LightGBM SHAP | Explainability tab and evaluation artifacts |
| Business-readable rules | Depth-4 surrogate decision tree trained to approximate EBM probabilities | `src/rules/derive_rules.py`, Rules tab |
| User interface | Five-section Streamlit app; Assistant is the landing section | `app.py`, `src/ui/` |
| Dockerized deployment | PostgreSQL, initialization job, and Streamlit app orchestrated through Compose | `Dockerfile`, `docker-compose.yml`, `.env.example` |
| Documentation | README, architecture image, setup instructions, limitations, presentation source | `README.md`, `docs/images/`, this file |

### Evaluation-weight framing

| Evaluation area | Weight | Where the presentation proves it |
|---|---:|---|
| ML solution and model quality | 30% | Slides 6-10 |
| LLM integration and talk-to-data | 25% | Slides 11-13 |
| Dockerization and engineering | 10% | Slide 14 |
| EDA | 15% | Slides 4-5 |
| Prompt optimization and hallucination control | 15% | Slide 13 |
| Documentation and architecture clarity | 5% | Slides 3 and 11 |

### Recommended visual

Use a requirement-to-delivery checklist with six large checkmarks. Move the detailed evidence table to speaker notes if the slide becomes dense.

---

## Slide 4 - Data landscape, grain, and quality

### Headline

**The pipeline preserves one applicant per row while adding behavioral history**

### Dataset facts

- Main training table: 307,511 applicants and 122 original columns.
- Target counts: 282,686 non-default/payment-normal records and 24,825 payment-difficulty records.
- Target rate: 8.0729%.
- Class ratio: approximately 11.4 non-defaults for every default.
- Final applicant-level modeling artifact: 307,511 rows and 149 columns after joins and engineered fields.
- Primary key: `SK_ID_CURR`.
- Quality checks: zero duplicate applicant IDs, zero missing targets, zero fully duplicated rows, and zero overlap between training/test applicant IDs when the Kaggle test file is present.

### Data sources used in the production feature pipeline

- `application_train.csv`: application, demographic, financial, contact, regional, and external-score fields.
- `bureau.csv`: prior external credit counts, active credit counts, overdue share, maximum days overdue.
- `previous_application.csv`: prior application counts, refusal counts, refusal rate.
- `installments_payments.csv`: late payments, underpayments, scheduled amount, paid amount, and payment ratios; processed in chunks to avoid loading the full history table into memory.

### History coverage in the final applicant artifact

- Bureau history: 263,491 applicants, 85.7% coverage.
- Previous-application history: 291,057 applicants, 94.6% coverage.
- Installment history: 291,643 applicants, 94.8% coverage.
- Explicit `HAS_*_HISTORY` flags distinguish “no matched history” from genuine zero events.

### Data-quality findings

- 67 of the original 122 columns contain missing values.
- 49 columns exceed 40% missingness, concentrated mainly in building/property variables and external-score fields.
- `DAYS_EMPLOYED = 365243` appears in 55,374 rows, 18.0% of applicants; it is treated as a placeholder, flagged, and replaced with missing for duration calculations.
- `CODE_GENDER = XNA` occurs in four rows.
- No non-positive income and no age outside 18-100 were found in the notebook checks.
- Missing external scores are retained and imputed because they carry strong predictive signal.

### Recommended visual

Use a grain diagram: four source tables -> applicant-level aggregate functions -> one `SK_ID_CURR` row. Alongside it, show three data-quality cards: 8.07% target rate, 67 columns with missingness, and 18.0% employment placeholder.

### Speaker notes

One-to-many joins are a common source of silent duplicate counting. The loader aggregates each history table before the join and uses pandas `validate="one_to_one"`; it asserts that row count and ID uniqueness remain unchanged. Installments are chunked in batches of one million input rows. This protects both statistical grain and 16 GB local memory.

---

## Slide 5 - EDA: six decision-relevant findings

### Headline

**Risk is concentrated in affordability, external scores, and repayment behavior**

### Finding 1 - severe target imbalance

- Overall default/payment-difficulty rate: 8.07%.
- A classifier predicting “no default” for every record would appear about 91.93% accurate.
- Therefore the project prioritizes ROC-AUC, PR-AUC, Brier score, calibration, and risk-band separation rather than raw accuracy.

### Finding 2 - age association

| Age band | Applicants | Defaults | Observed default rate |
|---|---:|---:|---:|
| 20-29 | 45,186 | 5,171 | 11.44% |
| 30-39 | 82,331 | 7,897 | 9.59% |
| 40-49 | 76,599 | 5,853 | 7.64% |
| 50-59 | 68,094 | 4,168 | 6.12% |
| 60+ | 35,301 | 1,736 | 4.92% |

This is an association for portfolio monitoring. Age-derived fields are excluded from surrogate policy rules.

### Finding 3 - housing segment lift

- Rented apartment: 4,881 applicants, 12.31% observed default, 1.53x portfolio lift.
- Living with parents: 14,840 applicants, 11.70% default, 1.45x lift.
- House/apartment: 272,868 applicants, 7.80% default.
- Office apartment: 2,617 applicants, 6.57% default.

### Finding 4 - affordability ratios add context

- Raw income and raw credit distributions overlap heavily across target groups.
- Credit-to-income and annuity-to-income represent debt burden relative to repayment capacity.
- Defaults show higher mean annuity-to-income burden: 0.1693 versus 0.1623 for non-defaults in the notebook output.
- The credit-to-goods ratio is also higher for defaults: 1.1452 versus 1.1188.

### Finding 5 - external scores dominate model signal

- All three `EXT_SOURCE_*` variables have a monotonic inverse relationship with default risk.
- In the model-free correlation scan, `EXT_SOURCE_3` has the strongest association with `TARGET` among the three: -0.1789, versus -0.1605 for `EXT_SOURCE_2` and -0.1553 for `EXT_SOURCE_1`.
- In the fitted EBM, `EXT_SOURCE_2` and `EXT_SOURCE_3` are the two highest-importance terms; their ordering differs slightly by model. The defensible conclusion is that external scores collectively dominate, rather than that one field wins universally.

### Finding 6 - repayment behavior is powerful

Using the production applicant-grain definitions:

- Any bureau overdue history: 15.78% default versus 7.98% when not observed.
- Any previous refusal history: 10.32% versus 6.98%.
- Any late-installment history: 9.41% versus 6.72%.

### Recommended visual

Use no more than three charts: age-band default-rate line, housing lift bar chart, and repayment-history observed-versus-not-observed bars. Put external-score deciles or affordability ratios in a smaller inset.

### Speaker notes

The numerical repayment rates above use the final one-row-per-applicant pipeline definitions and match the curated SQL views. Earlier narrative material may contain rates from alternative record-level or cohort definitions; do not mix those figures on the same slide. The presentation should state the grain whenever a behavior rate is shown.

---

## Slide 6 - Feature engineering and leakage controls

### Headline

**Features convert raw amounts and histories into comparable applicant-level risk signals**

### Engineered application features

| Feature | Formula / treatment | Business interpretation |
|---|---|---|
| `AGE_YEARS` | `-DAYS_BIRTH / 365.25` | Applicant age for diagnostics/modeling |
| `DAYS_EMPLOYED_ANOM` | 1 when `DAYS_EMPLOYED == 365243` | Explicit placeholder indicator |
| `DAYS_EMPLOYED_CLEAN` | Placeholder replaced with missing | Usable employment duration |
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Loan size relative to annual income |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Annual payment burden |
| `CREDIT_TERM` | `AMT_ANNUITY / AMT_CREDIT` | Relative annuity/credit structure; a proxy rather than literal duration |
| `CREDIT_GOODS_RATIO` | `AMT_CREDIT / AMT_GOODS_PRICE` | Financing relative to goods price |
| `EMPLOYED_AGE_RATIO` | employment duration / age in days | Employment stability relative to lifetime |

All divisions use a safe-ratio helper: zero denominators become missing and infinities are removed.

### Aggregated history features

- Bureau: `PRIOR_CREDIT_COUNT`, `ACTIVE_CREDIT_COUNT`, `OVERDUE_SHARE`, `MAX_DAYS_OVERDUE`.
- Previous applications: `PREV_APP_COUNT`, `PREV_REFUSED_COUNT`, `PREV_REFUSAL_RATE`.
- Installments: `INSTALLMENT_ROWS`, observed/late/underpayment counts, `SCHEDULED_AMOUNT`, `PAID_AMOUNT`, `INST_LATE_RATE`, `INST_UNDERPAY_RATE`, `INST_PAYMENT_RATIO`.

### Split and leakage controls

- Stratified 70/15/15 split with random seed 42.
- Training: 215,257 applicants.
- Validation: 46,127 applicants.
- Untouched test: 46,127 applicants.
- Imputation, categorical encoding, scaling, feature selection, calibration, and thresholds are fitted using the appropriate training/validation partitions, never the test target.
- The test set is used only for final model comparison and risk-band confirmation.
- Target and identifiers (`SK_ID_CURR`, `SK_ID_PREV`, `SK_ID_BUREAU`) are excluded from model features.

### Model-specific preprocessing

- Logistic Regression: median numeric imputation with missing indicators, standard scaling, categorical constant fill, one-hot encoding with `min_frequency=100`, unknown-category handling.
- LightGBM: median numeric imputation and ordinal encoding; unseen/missing categories map to -1.
- EBM: top selected mixed-type fields; train-fitted numeric medians and explicit `__MISSING__` categories; numeric fields remain continuous and categorical fields are marked nominal.

### Recommended visual

Show a pipeline: raw application + three history tables -> aggregate -> engineer ratios/anomaly flags -> stratified split -> train-fitted transformations -> models. Highlight “one row per applicant” and “test set untouched.”

---

## Slide 7 - Three-model strategy

### Headline

**A model hierarchy separates baseline sanity, predictive ceiling, and governed deployment**

### Logistic Regression - interpretable baseline floor

- `class_weight="balanced"` to address the 8.07% minority event.
- SAGA solver, maximum 300 iterations, tolerance 0.01, `C=0.5`.
- Purpose: detect pipeline errors, establish an interpretable linear baseline, and prove that the feature set has signal.
- Saved as `models/logistic_pipeline.joblib`.

### LightGBM - shadow performance ceiling and feature pruner

- Balanced binary classifier with 1,200 maximum estimators, learning rate 0.03, 31 leaves, minimum child sample 50, 85% row and feature sampling, L2 regularization 1.0.
- Validation AUC early stopping after 60 rounds without improvement.
- Fits all engineered features and ranks them by split importance.
- Top 30 raw features are passed to the EBM.
- Remains a reference model; it is not the deployed decision model.
- Saved as `models/lightgbm_shadow.joblib`.

### EBM - production glass-box model

- ExplainableBoostingClassifier, a generalized additive model with boosted shape functions and selected pairwise interactions.
- Trained on a stratified 120,000-row sample from the training partition to fit the 16 GB RAM/time budget while preserving target proportions.
- Top 30 LightGBM-selected features.
- Balanced sample weights.
- `outer_bags=4`, `inner_bags=0`, `max_bins=128`, `max_interaction_bins=32`, `interactions=8`.
- Internal 15% validation within EBM, up to 15,000 rounds, early stopping after 75 rounds.
- Saved with its transformer, calibrator, feature list, feature types, thresholds, and metadata as `models/ebm_bundle.joblib`.

### Why EBM is production even though LightGBM has higher ROC-AUC

- LightGBM provides the highest ranking performance but requires post-hoc explanations.
- EBM trails LightGBM by only 0.014 ROC-AUC while exposing exact additive term contributions and learned feature-shape functions.
- Native global and local explanations support model review without using a separate explanation model.
- Platt scaling provides calibrated probabilities for thresholds and risk bands.
- This is a deliberate performance/governance trade-off, not a claim that EBM wins every metric.

### Recommended visual

Use three columns: Baseline -> Shadow/Prune -> Production/Explain. Add a thin arrow from LightGBM top-30 importance into EBM.

---

## Slide 8 - Model results on the untouched test set

### Headline

**EBM retains strong discrimination and delivers the best stored probability calibration**

| Model | ROC-AUC | PR-AUC | Brier score | Training detail | Governance role |
|---|---:|---:|---:|---|---|
| Logistic Regression | 0.7589 | 0.2378 | 0.1995 | 16.77 seconds | Interpretable baseline |
| LightGBM shadow | 0.7783 | 0.2652 | 0.1741 | 23.40 seconds | Ranking ceiling and feature pruning |
| Calibrated EBM | 0.7643 | 0.2503 | 0.0675 | 120,000 training rows | Production glass-box model |

### Interpretation

- All models outperform the 8.07% positive-class prevalence substantially on PR-AUC.
- EBM PR-AUC is about 3.1x the random-prevalence baseline.
- LightGBM leads EBM by 0.0140 ROC-AUC and 0.0149 PR-AUC.
- EBM Brier score is 66% lower than the stored Logistic result and 61% lower than the stored LightGBM result.
- Calibration caveat: the EBM received Platt calibration on the validation split; the saved baseline/shadow figures are uncalibrated. A perfectly controlled model comparison would calibrate all three before treating Brier differences as purely model-driven.
- Ranking metrics are evaluated using raw EBM scores so monotonic calibration does not alter rank.

### Recommended visual

Use a grouped bar chart for ROC-AUC and PR-AUC, plus a separate “lower is better” Brier bar. Visually highlight EBM as the deployed choice and LightGBM as the benchmark rather than declaring one overall winner.

### Speaker notes

ROC-AUC measures ranking across thresholds, PR-AUC is more informative under class imbalance, and Brier score measures squared probability error. The EBM decision balances all three with auditability. Accuracy is deliberately omitted because it is misleading at this class ratio.

---

## Slide 9 - Calibration, risk score, and risk bands

### Headline

**Validation-derived thresholds create monotonic, operationally readable test segments**

### Calibration and threshold method

1. Fit EBM on the training sample.
2. Generate raw probabilities on validation and test.
3. Fit a Platt calibrator on validation probabilities and labels.
4. Transform probabilities monotonically, preserving ranking.
5. Set thresholds from calibrated validation-score percentiles: P60 and P85.
6. Freeze thresholds and evaluate the bands on the untouched test set.

### Risk threshold definitions

- Low: probability below 0.064908, approximately below 6.49%, validation P60.
- Medium: 0.064908 to below 0.157852, approximately 6.49%-15.79%, validation P60-P85.
- High: probability at or above 0.157852, approximately 15.79%+, validation P85+.
- Internal risk score: rounded `probability x 1000`; higher score means greater estimated payment-difficulty risk.
- Score cutoffs are therefore approximately 65 and 158.

### Untouched-test performance by band

| Risk band | Applicants | Population share | Defaults | Observed default rate | Mean predicted risk | Share of all defaults captured |
|---|---:|---:|---:|---:|---:|---:|
| Low | 27,810 | 60.29% | 922 | 3.32% | 3.11% | 24.76% |
| Medium | 11,389 | 24.69% | 1,120 | 9.83% | 10.15% | 30.08% |
| High | 6,928 | 15.02% | 1,682 | 24.28% | 24.51% | 45.17% |

### Key messages

- Observed default rates rise monotonically from 3.3% to 9.8% to 24.3%.
- The highest 15% of scores capture 45.2% of test-set defaults.
- High-risk observed default is about 3.0x the overall portfolio default rate.
- Mean predicted and observed rates are close within every band, supporting probability usability at this coarse segmentation level.
- Bands are internal portfolio percentiles because no lender cost matrix or approved policy thresholds were provided.
- They are not RBI risk grades, bureau credit ratings, approval cutoffs, or customer-facing adverse-action policy.

### Recommended visual

Use a three-band horizontal risk strip with population share and observed default rate. Add a small bar showing 15% of applicants capturing 45.2% of defaults.

---

## Slide 10 - Explainability and surrogate rules

### Headline

**The system explains both model behavior and individual scores, with clear fidelity boundaries**

### Global EBM explanation

Top fitted EBM terms by stored importance:

1. `EXT_SOURCE_2` - 0.349
2. `EXT_SOURCE_3` - 0.331
3. `EXT_SOURCE_1` - 0.182
4. `AMT_GOODS_PRICE` - 0.172
5. `CREDIT_TERM` - 0.151
6. `INST_LATE_RATE` - 0.142
7. `ORGANIZATION_TYPE` - 0.127
8. `CREDIT_GOODS_RATIO` - 0.123
9. `AMT_ANNUITY` - 0.123
10. `PRIOR_CREDIT_COUNT` - 0.109

The EBM also learned eight pairwise interactions. An example stored interaction is `EXT_SOURCE_1 & DAYS_BIRTH`.

### Local explanation

- `model.explain_local()` returns the exact additive score contributions for a selected applicant.
- Positive contribution increases EBM log-odds of payment difficulty; negative contribution reduces it.
- The UI plots the top 12 absolute contributions and lists the largest escalators and mitigators.
- Contributions are model-score/log-odds terms, not direct percentage-point changes in probability.

Example high-risk case from the saved explanation artifact:

- Estimated probability: 33.93%.
- Main upward contributions: very low `EXT_SOURCE_2` (+0.674), `CREDIT_TERM` (+0.336), `CREDIT_GOODS_RATIO` (+0.309), and `INST_LATE_RATE` (+0.171).
- This example makes the explanation intuitive: weak external score, financing structure, financed amount relative to goods price, and late-payment history combine to elevate risk.

### LightGBM SHAP cross-check

- A sampled TreeExplainer pass provides post-hoc importance for the shadow model.
- Top mean absolute SHAP fields: `EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, `CREDIT_TERM`, `CODE_GENDER`, `CREDIT_GOODS_RATIO`, `INST_LATE_RATE`.
- Agreement on major external and affordability signals increases confidence that findings are not unique to one model family.
- Protected/policy-sensitive fields appearing in predictive analysis require fairness review and are not converted into business rules.

### Surrogate-tree rule extraction

- A `DecisionTreeRegressor` approximates EBM probabilities.
- Maximum depth: 4; minimum leaf size: 2% of training rows.
- Candidate rules use numeric features and remove `CODE_GENDER`, `AGE_YEARS`, `DAYS_BIRTH`, `NAME_FAMILY_STATUS`, and `NAME_EDUCATION_TYPE`.
- Validation R-squared: 0.5251.
- Risk-band agreement with EBM: 67.96%.
- Example terminal rule outputs span approximately 2.44% to 28.73% estimated probability.
- Dominant split fields include `EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_1`, and `INST_LATE_RATE`.

### Governance statement

The surrogate tree is an explanatory approximation, not the production scoring logic and not an approval/decline policy. Its 68% band agreement is useful for communication but insufficient to replace the EBM.

### Recommended visual

Use a split slide: left, a local waterfall/bar explanation screenshot; right, three shortened surrogate paths with a fidelity badge “68% band agreement.” Avoid displaying the full tree text.

---

## Slide 11 - System architecture

### Headline

**Prediction, explanation, and conversational analytics share one controlled platform**

### Required visual

Use `docs/images/credit-risk-platform-architecture.png` full width.

### Architecture walkthrough

1. **Streamlit presentation layer** exposes Agentic Assistant, Portfolio EDA, Risk Prediction, Explainability, and Surrogate Rules.
2. **Model-serving path** loads the saved EBM bundle and preprocessor from read-only artifacts, produces calibrated probability, score, risk band, and local explanation.
3. **LangGraph agent path** routes questions to one of three tools, gathers evidence, and synthesizes a readable answer.
4. **PostgreSQL/Supabase data layer** stores applicant-grain analytics, curated summary views, knowledge chunks with pgvector embeddings, and LangGraph checkpoints.
5. **External retrieval path** uses DDGS only for general concepts; it is blocked from answering internal portfolio questions.

### Why PostgreSQL instead of loading CSVs on each Streamlit rerun

- Stable applicant grain and curated views prevent accidental duplicate counting.
- Indexed SQL is faster and easier to audit than repeated in-memory joins.
- A restricted read-only login creates defense in depth beneath application SQL validation.
- The same database supports pgvector retrieval and persistent conversation checkpoints.
- Streamlit caches the database engine, EBM predictor, agent, and static artifacts.

### Database objects

- Schemas: `raw`, `analytics`, `platform`.
- Main table: `analytics.applicants`, primary key `sk_id_curr`.
- Views: `applicant_risk_features`, `portfolio_summary`, `age_band_summary`, `education_summary`, `housing_summary`, `repayment_history_summary`.
- Knowledge table: `platform.knowledge_chunks`, with JSONB metadata and `VECTOR(768)`.
- Indexes: target, education, housing, age, history coverage; GIN metadata; HNSW cosine-vector index.
- Verified database invariant: 307,511 rows, 307,511 distinct applicants, one row per applicant, one portfolio-summary row, five age segments.

---

## Slide 12 - Agentic chatbot and three-tool routing

### Headline

**The agent chooses the narrowest evidence source for each question**

### Tool 1 - `query_database`

- Used for counts, rates, averages, comparisons, repayment behavior, and applicant-level analytics.
- Gemini generates one PostgreSQL query from a schema-constrained prompt.
- Summary questions preferentially use pre-aggregated views.
- SQL is parsed and validated locally, executed through the read-only role, and returned with up to 200 rows.
- One repair attempt is allowed for a non-security validation or execution error.

Example: “Which age band has the highest default rate?” -> `analytics.age_band_summary` -> answer includes 20-29 and approximately 11.44%.

### Tool 2 - `search_knowledge_base`

- Used for feature definitions, EDA findings, EBM rationale, model metrics, risk-band definitions, and surrogate-rule explanations.
- Content includes official Home Credit column descriptions plus curated, confirmed project findings.
- No applicant records are placed in the knowledge base.
- Retrieval uses a deterministic local 768-dimensional hashed lexical embedding stored in pgvector and cosine similarity, returning up to five chunks.
- HNSW index supports retrieval performance.

Example: “What does EXT_SOURCE_3 mean?” -> feature-definition/EDA chunks -> answer explains normalized external score and risk direction.

### Tool 3 - `search_web`

- Used for external, general concepts such as probability of default or Basel terminology.
- DDGS returns a maximum of five titles, snippets, and URLs.
- Requests about the internal portfolio are rejected by this tool.
- If external search is unavailable, the tool fails transparently and can return a narrow cached definition for known concepts.

Example: “What is probability of default in general? Use external sources.” -> DDGS -> sourced general definition.

### Routing flow

1. Early security check blocks obvious mutating/destructive verbs before LLM routing.
2. Deterministic regex rules handle common feature, model, portfolio, and external-definition questions quickly.
3. Ambiguous questions use Gemini JSON classification into database, knowledge, web, or direct response.
4. The selected tool runs.
5. Gemini synthesizes the answer strictly from tool evidence.
6. LangGraph saves the turn to the thread checkpoint.

### Recommended visual

Show three example question bubbles flowing to the corresponding tools, then converging into “Answer + evidence + SQL/source.”

---

## Slide 13 - Prompt optimization, hallucination control, SQL safety, and memory

### Headline

**Reliability comes from bounded prompts and deterministic enforcement around the LLM**

### Prompt and token optimization

- Deterministic routing avoids an LLM classification call for obvious questions.
- Router requests strict JSON with only four allowed route labels.
- NL-to-SQL prompt includes only the approved analytics schema, field descriptions, constraints, and five short few-shot examples.
- Summary-view preference improves accuracy and reduces generated SQL complexity.
- SQL validation runs locally; it does not spend another model call.
- Normal deterministic database flow uses SQL generation plus final synthesis; knowledge/web flows usually need synthesis only.
- An ambiguous question can add one router call; a failed SQL query can add only one repair call.
- Tool output is bounded: 200 SQL rows, five KB chunks, five web results.
- Cached Streamlit resources avoid reloading the model, database pool, and agent on every rerun.

### SQL defense layers

1. Pre-routing block for `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, `CREATE`, `GRANT`, and `REVOKE` requests.
2. `sqlglot` PostgreSQL AST parsing.
3. Exactly one statement.
4. Only `SELECT` or `WITH ... SELECT`.
5. No inline or block comments.
6. No nested DDL/DML expressions.
7. Query must read an approved analytics object.
8. Only the `analytics` schema; `raw`, `platform`, and system catalogs are unavailable.
9. Table/view whitelist of seven approved objects.
10. Positive allowlist of aggregate/string/date functions; system introspection and functions such as `pg_sleep` are rejected.
11. LIMIT is added or clamped to 200.
12. Database role is read-only with an eight-second statement timeout.

### Hallucination controls

- Final prompt instructs Gemini to answer strictly from provided evidence.
- Empty evidence produces an explicit “no matching records/evidence” response.
- Tool failures are returned as errors rather than hidden.
- Web results preserve URLs; internal knowledge preserves headings and source names.
- UI exposes the tool used, generated SQL, result table/chart, assumptions/evidence, and sources.
- Security-invalid SQL is refused immediately and never passed to the repair loop.
- LLM/API failure degrades to direct evidence formatting instead of fabricating an answer.

### Conversation memory

- Every browser conversation has a stable UUID thread ID, also accepted through the URL `?thread=` parameter.
- LangGraph `PostgresSaver` persists checkpoints in PostgreSQL.
- On refresh, Streamlit reconstructs visible messages from saved conversation turns.
- Follow-up markers such as “previous question” or “summarize that” route to conversation memory.
- If PostgreSQL checkpointing is unavailable, the agent falls back to session-only behavior and makes that state visible.

### Security demo

Prompt: `DROP TABLE analytics.applicants;`

Expected response: tool `SQL Safety Guard`; destructive request is rejected with a prohibited-operation message; no SQL is executed.

### Recommended visual

Use a defense-in-depth stack: Prompt constraint -> AST validator -> allowlists/limits -> read-only role/timeout -> transparent UI. Add a small refresh icon linking browser thread ID to PostgreSQL checkpoints.

---

## Slide 14 - Product experience, engineering, deployment, and tests

### Headline

**The result is a deployable five-section application, not a notebook-only prototype**

### Five Streamlit sections

1. **Agentic Assistant** - landing section; suggested questions, chat, tool badge, generated SQL, result table, citations, and persistent thread.
2. **Portfolio EDA** - cached SQL summaries and Plotly views for cohort, housing, education, and repayment behavior.
3. **Risk Prediction** - realistic applicant archetypes, adjustable fields, calibrated probability, 0-1000 score, Low/Medium/High band, and risk gauge.
4. **Explainability** - Logistic/LightGBM/EBM comparison, EBM global importance, LightGBM SHAP summary, and applicant-level EBM contributions.
5. **Surrogate Rules** - simplified rule cards, fidelity metrics, governance disclaimer, and expandable raw tree rules.

### UX choices

- Wide institutional dashboard layout influenced by the Streamlit Stock Peer Analysis gallery example.
- Dark navy theme with restrained cyan, teal, amber, and coral accents.
- Compact metric cards, bordered containers, explanatory tooltips, and interactive Plotly charts.
- Expensive resources use Streamlit caching.
- The full raw dataset is not aggregated during each rerun.

### Docker Compose topology

- `db`: PostgreSQL 17 with pgvector, persistent volume, and health check.
- `init-db`: one-shot schema/data/knowledge initialization after database health succeeds.
- `app`: Streamlit waits for both database health and successful initialization.
- Local ports: Streamlit 8501, PostgreSQL 5432.
- Models, artifacts, and data are mounted into containers; app-facing mounts are read-only.
- Container health endpoint: `/_stcore/health`.
- Runtime environment and required secrets are documented in `.env.example`.

### Data-loader resilience

- Applicant rows upload in 500-row batches.
- Inserts use `ON CONFLICT (sk_id_curr) DO NOTHING`.
- Existing applicant IDs are retained and skipped, so an interrupted managed-database upload resumes rather than restarting from row one.
- Connection drops trigger bounded retries and engine disposal/reconnection.
- This behavior was exercised during the Supabase upload on an unstable connection before all 307,511 applicants were verified.

### Automated verification

- Current result: **33 tests passed** in the `NeoStatsCredit` environment.
- Preprocessor/evaluation tests cover feature engineering, safe ratios, split behavior, risk-band monotonicity, and metrics.
- Database tests cover applicant-grain integrity and access behavior.
- SQL safety tests cover valid SELECT/CTE queries, DDL/DML rejection, comments, schema/table whitelist, disallowed functions, system introspection, approved aggregates, and limit clamping.
- Nine chatbot gold paths cover overall default rate, age, education, repayment behavior, feature definition, EBM rationale, web search, conversational follow-up, and destructive-query rejection.

### Recommended screenshots

- One composite of the five tab names and app header.
- Assistant showing a successful database answer with expandable SQL.
- Prediction showing probability, score, band, and gauge.
- Explainability showing local contributions.
- Use at most three screenshots on this slide; move additional screenshots to an appendix or demo.

---

## Slide 15 - Outcome, limitations, roadmap, and closing

### Headline

**A working decision-support platform with explicit boundaries and a clear production path**

### Delivered value

- One governed workflow connects portfolio analytics, model inference, explanations, rules, and natural-language access.
- The model separates risk meaningfully: 3.3%, 9.8%, and 24.3% observed default across the three test bands.
- The highest-risk 15% captures 45.2% of defaults.
- EBM provides native global and applicant-level explanations with only a small ranking gap to LightGBM.
- Analysts can answer common portfolio questions without direct SQL while the application still exposes SQL and sources for review.
- SQL controls and a read-only database role reduce the impact of unsafe or incorrect model output.
- Conversation continuity survives browser refresh through PostgreSQL checkpoints.
- Local and cloud deployment paths are both demonstrated.

### Known limitations

- Home Credit is a historical competition dataset; it does not represent a live lender portfolio or Indian regulatory population.
- No temporal/out-of-time validation is possible from the current setup; the split is stratified and random.
- Risk bands are percentile-based because no approved business cost matrix, risk appetite, or capacity constraint was provided.
- No fairness audit, protected-class impact assessment, reject-inference analysis, or customer-notice validation has been completed.
- Demographic associations can reflect structural factors and must not be interpreted causally.
- The EBM was trained on a 120,000-row stratified sample for time/RAM efficiency rather than the entire training partition.
- LightGBM feature selection may bias the EBM feature set toward the shadow model’s preferred representations.
- Model baselines were not calibrated through the identical procedure; Brier comparison should be treated accordingly.
- The deterministic knowledge embedding is lightweight and reproducible but less semantically expressive than a modern embedding model.
- DDGS quality and availability depend on an external service; web snippets are not equivalent to curated authoritative research.
- Surrogate fidelity is moderate at 68% risk-band agreement; rules cannot replace EBM inference.
- Authentication, role-based UI permissions, production monitoring, drift detection, and audit-event storage are outside the demo scope.
- A fresh Docker clone needs the mounted Home Credit data and generated applicant artifact before database initialization; the dataset is intentionally excluded from Git.
- Supabase/Streamlit free-tier cold starts, connection limits, and network latency may affect demo response time.

### Prioritized roadmap

1. Define lender-specific costs and decision capacity; optimize thresholds using expected loss and business constraints.
2. Add temporal validation, stability monitoring, drift detection, and scheduled recalibration.
3. Run fairness diagnostics, subgroup calibration, adverse-impact analysis, and governance review.
4. Calibrate all benchmark models consistently and add confidence intervals/bootstrap comparisons.
5. Replace lexical hashed vectors with a managed semantic embedding model and evaluate retrieval precision/recall.
6. Add authoritative-domain controls and content caching for web search.
7. Add authentication, user roles, audit logging, secret rotation, and monitoring.
8. Automate raw-data-to-feature generation inside the Docker initialization path.
9. Add model registry/versioning, data lineage, and reproducible experiment tracking.
10. Add portfolio batch scoring and controlled CSV export after governance requirements are defined.

### Closing line

**The project demonstrates that useful credit-risk AI can combine predictive signal, transparent reasoning, safe analytics, and reproducible engineering in one compact platform.**

### Final demo transition

“I’ll now demonstrate the highest-priority path: ask a portfolio question, inspect its safe SQL and result, score an applicant with the EBM, inspect the local explanation, then refresh the page and continue the same conversation.”

---

# Appendix A - Exact metrics and calculations

## Portfolio statistics

| Statistic | Value |
|---|---:|
| Applicants | 307,511 |
| Non-default / normal repayment (`TARGET=0`) | 282,686 |
| Payment difficulty (`TARGET=1`) | 24,825 |
| Default rate | 8.0729% |
| Non-default share | 91.9271% |
| Majority-to-minority ratio | 11.39:1 |
| Mean credit amount | 599,026 |
| Median credit amount | 513,531 |
| Mean annual income | 168,798 |
| Median annual income | 147,150 |
| Mean annuity | 27,109 |
| Median annuity | 24,903 |

## Model comparison

| Metric | Logistic Regression | LightGBM shadow | Calibrated EBM |
|---|---:|---:|---:|
| ROC-AUC | 0.7589446 | 0.7783166 | 0.7643144 |
| PR-AUC | 0.2378246 | 0.2651998 | 0.2503479 |
| Brier score | 0.1994624 | 0.1740637 | 0.0675448 |
| Stored training time | 16.77 sec | 23.40 sec | Not retained after calibration update |
| EBM training rows | - | - | 120,000 |
| Risk bands monotonic | - | - | Yes |

## Risk-band validation

| Band | Probability interval | Score interval (approx.) | Test applicants | Defaults | Observed rate | Mean prediction | Default capture |
|---|---|---|---:|---:|---:|---:|---:|
| Low | `< 0.064908` | `< 65` | 27,810 | 922 | 3.315% | 3.113% | 24.758% |
| Medium | `0.064908 to <0.157852` | `65 to <158` | 11,389 | 1,120 | 9.834% | 10.153% | 30.075% |
| High | `>= 0.157852` | `>= 158` | 6,928 | 1,682 | 24.278% | 24.505% | 45.166% |

---

# Appendix B - Final 30 EBM input features

Selected by LightGBM split importance and stored in `models/feature_manifest.json`, in selection order:

1. `CREDIT_TERM`
2. `EXT_SOURCE_3`
3. `EXT_SOURCE_1`
4. `EXT_SOURCE_2`
5. `DAYS_ID_PUBLISH`
6. `DAYS_REGISTRATION`
7. `DAYS_LAST_PHONE_CHANGE`
8. `AMT_ANNUITY`
9. `DAYS_BIRTH`
10. `ANNUITY_INCOME_RATIO`
11. `PAID_AMOUNT`
12. `CREDIT_GOODS_RATIO`
13. `INST_LATE_RATE`
14. `AMT_GOODS_PRICE`
15. `REGION_POPULATION_RELATIVE`
16. `EMPLOYED_AGE_RATIO`
17. `SCHEDULED_AMOUNT`
18. `CREDIT_INCOME_RATIO`
19. `AGE_YEARS`
20. `AMT_CREDIT`
21. `INSTALLMENT_ROWS`
22. `INST_PAYMENT_RATIO`
23. `ACTIVE_CREDIT_COUNT`
24. `DAYS_EMPLOYED_CLEAN`
25. `AMT_INCOME_TOTAL`
26. `ORGANIZATION_TYPE`
27. `PREV_REFUSAL_RATE`
28. `PRIOR_CREDIT_COUNT`
29. `OWN_CAR_AGE`
30. `DAYS_EMPLOYED`

Governance nuance: age variables remain in the predictive model but are excluded from surrogate business rules. A real deployment would require legal/policy review and fairness testing before deciding whether they remain in production inference.

---

# Appendix C - Sample surrogate rule paths

These are approximations of EBM probability and should be shortened for slides.

### Highest-risk branch

- `EXT_SOURCE_3 <= 0.3163`
- `EXT_SOURCE_2 <= 0.4172`
- `EXT_SOURCE_2 <= 0.2246`
- Surrogate probability: approximately 28.73%

### Elevated-risk branch

- `EXT_SOURCE_3 > 0.3163`
- `EXT_SOURCE_2 <= 0.3992`
- `EXT_SOURCE_3 <= 0.5451`
- `EXT_SOURCE_2 <= 0.1856`
- Surrogate probability: approximately 19.28%

### Repayment-behavior branch

- `EXT_SOURCE_3 > 0.3163`
- `EXT_SOURCE_2 <= 0.3992`
- `EXT_SOURCE_3 > 0.5451`
- `INST_LATE_RATE <= 0.0754` -> approximately 6.09%
- `INST_LATE_RATE > 0.0754` -> approximately 10.19%

### Lowest-risk branch

- `EXT_SOURCE_3 > 0.3163`
- `EXT_SOURCE_2 > 0.3992`
- `EXT_SOURCE_3 > 0.5362`
- `EXT_SOURCE_2 > 0.6178`
- Surrogate probability: approximately 2.44%

---

# Appendix D - Verified chatbot demo questions

| # | Question | Expected tool | Expected evidence |
|---:|---|---|---|
| 1 | What is the overall default rate? | Database Query | 307,511 applicants; 8.07% |
| 2 | Which age band has the highest default rate? | Database Query | 20-29; 11.44% |
| 3 | Compare default rates across education levels. | Database Query | `education_summary` rows |
| 4 | How does previous late-payment behaviour relate to default? | Database Query | Observed vs not observed history |
| 5 | What does EXT_SOURCE_3 mean? | Knowledge Base | External/normalized credit score definition |
| 6 | Why was EBM selected? | Knowledge Base | Glass-box, additive contributions, performance/governance trade-off |
| 7 | What is probability of default in general? Use external sources. | Web Search (DDGS) | General sourced definition |
| 8 | Can you summarize the findings from the previous question? | Conversation Memory | Prior response recovered from thread |
| 9 | DROP TABLE analytics.applicants; | SQL Safety Guard | Refusal; prohibited operation; nothing executed |

### Best live-demo sequence

1. Ask “Which age band has the highest default rate?”
2. Expand the generated SQL and show that it queries a curated summary view.
3. Ask “What does EXT_SOURCE_3 mean?” and point out the tool changes to Knowledge Base.
4. Ask “What is probability of default in general? Use external sources.” and show DDGS citations.
5. Ask a follow-up using “previous question.”
6. Refresh and show that the thread remains.
7. Run the destructive SQL prompt and show the guardrail.
8. Move to Risk Prediction, select a high-risk archetype, and score it.
9. Move to Explainability and show its risk escalators and mitigators.

---

# Appendix E - Repository and artifact map

```text
CreditRisk/
|-- app.py                              # Streamlit entry point and five sections
|-- notebooks/
|   `-- Home_Credit_Default_Risk_EDA.ipynb
|-- src/
|   |-- data/
|   |   |-- loader.py                  # applicant-grain history aggregation
|   |   `-- preprocessor.py            # engineered features, split, transformations
|   |-- ml/
|   |   |-- train.py                   # Logistic -> LightGBM -> EBM
|   |   |-- predict.py                 # probability, score, band, local explanation
|   |   `-- evaluate.py                # metrics, calibration, band evaluation
|   |-- rules/
|   |   `-- derive_rules.py            # policy-sensitive-filtered surrogate tree
|   |-- db/
|   |   |-- connection.py              # pooled and transaction-pooler connections
|   |   |-- load_data.py               # resilient database initialization
|   |   |-- knowledge_base.py          # curated chunks and vector retrieval
|   |   `-- verify.py                  # row-grain and view checks
|   |-- talk_to_data/
|   |   |-- graph.py                   # LangGraph state machine and persistence
|   |   |-- router.py                  # deterministic/Gemini routing
|   |   |-- nl_to_sql.py               # SQL generation and one repair
|   |   |-- sql_validator.py           # AST safety controls
|   |   |-- query_runner.py            # bounded execution
|   |   |-- knowledge_search.py        # pgvector tool wrapper
|   |   |-- web_search.py              # DDGS tool and fallback
|   |   `-- prompt_templates.py        # routing, SQL, repair, synthesis prompts
|   `-- ui/                            # five Streamlit section modules and styles
|-- sql/                               # extensions, schema, views, indexes, grants
|-- models/                            # reloadable model bundles and thresholds
|-- artifacts/evaluation/              # metrics, importance, SHAP, local examples
|-- artifacts/rules/                   # surrogate fidelity and raw rules
|-- tests/                              # unit, integration, gold questions
|-- docs/images/                       # README architecture diagram
|-- documents/project_presentation.pdf # required submitted PDF
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt                   # lean application runtime
|-- requirements-dev.txt               # training/notebook/test dependencies
|-- .env.example
|-- .gitignore                         # excludes raw data, secrets, local artifact
`-- README.md
```

### Saved artifact sizes

- EBM bundle: approximately 0.49 MB.
- LightGBM shadow bundle: approximately 1.07 MB.
- Logistic pipeline: approximately 0.01 MB.
- Surrogate tree: approximately 0.002 MB.
- Small artifacts help Streamlit load quickly and keep GitHub deployment practical.

---

# Appendix F - Screenshot capture checklist

The assignment explicitly asks for output screenshots in the final presentation PDF.

1. **Landing page:** platform header, PostgreSQL Online status, and five section tabs.
2. **Database chatbot result:** question, readable answer, Database Query badge, result table, expandable SQL.
3. **Knowledge chatbot result:** `EXT_SOURCE_3` definition with Knowledge Base badge.
4. **Web chatbot result:** probability-of-default definition with DDGS badge and source URLs.
5. **Memory:** same conversation visible after page refresh; keep thread ID visible but do not expose secrets.
6. **Safety:** destructive query refusal.
7. **Portfolio EDA:** age or housing chart with metric cards.
8. **Risk Prediction:** probability, score, risk band, and gauge for a selected profile.
9. **Explainability:** applicant-level contribution chart with escalators and mitigators.
10. **Rules:** surrogate rule cards and fidelity metrics.
11. **Docker/engineering evidence:** optional terminal screenshot showing healthy services or `33 passed`; use only if it remains legible.

For a 15-slide deck, use screenshots 1, 2, 8, and 9 in the main story. Put the remaining evidence in an appendix or a single demo-evidence collage.

---

# Appendix G - Likely interview questions and strong answers

### Why not deploy LightGBM if it has the highest AUC?

LightGBM is the shadow performance ceiling and feature selector. Its ROC-AUC lead over EBM is 0.014. EBM provides exact additive global and local explanations and learned feature shapes while maintaining strong PR-AUC. For this assignment’s explainability and audit objective, that trade-off justified EBM as the decision-support model. In a real lender, the selection would be governed by an approved performance, fairness, stability, and explainability standard.

### Why use PR-AUC?

Only 8.07% of applicants have the positive target. PR-AUC focuses on precision and recall for that minority event and is more informative than accuracy. ROC-AUC remains useful for overall ranking; calibration/Brier score matters because the UI displays probabilities and bands.

### Why Platt scaling?

Balanced class weighting changes the relationship between raw classifier score and real event prevalence. A monotonic logistic calibration fitted on held-out validation predictions converts raw EBM output into more useful probabilities without changing rank. Isotonic calibration could be compared later, but Platt scaling is compact and stable.

### Why percentile risk bands?

No approved cost matrix, risk appetite, approval capacity, or RBI-defined mapping was supplied. Validation percentiles create stable operational segments for the demo. They are evaluated out of sample and clearly labeled internal model-estimated bands.

### Is the 0-1000 score a credit score?

It is an internal risk score equal to calibrated payment-difficulty probability times 1000. Higher means higher risk. It is not a bureau score, regulatory rating, or approval cutoff.

### How do you prevent data leakage?

The project separates train, validation, and final test by a stratified 70/15/15 split. Preprocessors and feature selection are fit on training data; Platt calibration and percentile thresholds use validation data; final metrics and band behavior use the untouched test. Applicant IDs and target are excluded from features. History tables are aggregated to applicant level before joining.

### How does the chatbot avoid duplicate-counting errors?

It prefers curated applicant-level summary views. The base analytics table has a primary key on `SK_ID_CURR` and was verified to have 307,511 distinct IDs across 307,511 rows. The chatbot cannot freely join raw one-to-many tables.

### Can prompt instructions alone make SQL safe?

No. The system treats the LLM output as untrusted. A local AST validator, schema/table/function allowlists, row clamping, a read-only role, and statement timeout enforce safety independently of the prompt.

### Why a deterministic router plus Gemini?

Common demo intents are easy to identify with precise patterns. Deterministic routing reduces latency, cost, and routing variability. Gemini handles ambiguous natural language. Invalid or unavailable model output falls back predictably.

### Why pgvector with local hashed embeddings?

It avoids another API dependency and produces deterministic, fast lexical retrieval for feature names and project terminology. The trade-off is weaker semantic matching; a production roadmap would use a stronger embedding model and retrieval evaluation.

### Are the surrogate rules the model?

No. The EBM is the production model. The depth-4 tree approximates EBM probabilities for communication, with R-squared 0.525 and 68% band agreement. It is explicitly labeled an explanatory approximation.

### How would you productionize this?

Add temporal validation, consistent calibration benchmarks, fairness analysis, monitored data/model drift, model registry and lineage, authentication and role-based permissions, audit logs, approved decision thresholds, and a controlled release process. Replace free external search with an allowlisted authoritative retrieval layer.

---

# Appendix H - Factual sources within the repository

- Requirements: `NeoStats_AI_Use_Case.pdf` supplied with the assignment.
- Canonical EDA: `notebooks/Home_Credit_Default_Risk_EDA.ipynb`.
- Exact split and feature list: `models/feature_manifest.json`.
- Model metrics: `artifacts/evaluation/metrics.json` and `model_comparison.csv`.
- Risk thresholds: `models/risk_thresholds.json`.
- Risk-band outcomes: `artifacts/evaluation/risk_band_summary.csv`.
- Global EBM importance: `artifacts/evaluation/ebm_global_importance.csv`.
- LightGBM feature ranking and SHAP: `lightgbm_feature_importance.csv`, `lightgbm_shap_summary.csv`.
- Local explanation examples: `artifacts/evaluation/ebm_local_examples.json`.
- Surrogate fidelity and rules: `artifacts/rules/business_rules.json`.
- Model configuration: `src/ml/train.py`.
- Preprocessing and aggregation: `src/data/preprocessor.py`, `src/data/loader.py`.
- Agent graph and prompts: `src/talk_to_data/graph.py`, `prompt_templates.py`.
- SQL enforcement: `src/talk_to_data/sql_validator.py`, `query_runner.py`.
- Database design: `sql/001_extensions_roles.sql` through `004_indexes_grants.sql`.
- Deployment: `Dockerfile`, `docker-compose.yml`, `.env.example`.
- Test evidence: `tests/` and `tests/gold_questions.yaml`; latest local run: 33 passed.

