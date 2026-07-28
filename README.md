# 📊 AI Data Analyst

An AI-powered data analyst that lets you upload CSV files and interact with them in
plain English — ask questions, get charts, generate SQL/pandas, detect anomalies,
and get explained reasoning behind every answer. Powered by **Groq** (fast
open-weight LLM inference with tool/function calling).

Built for the Digital Back Office **AI Engineer Assignment**.

> **Live app / demo video:** _add your deployed URL and demo video link here before submitting_
> **Screenshots:** see [`docs/screenshots`](docs/screenshots) _(add screenshots before submitting)_

---

## ✨ What it does

- Upload **one or more CSV files**, with automatic validation, type inference, and
  data-quality checks (missing values, duplicates, suspicious negative values).
- **Chat with your data** in natural language — the model decides which tool to use
  (pandas, SQL, charting, anomaly detection) based on your question, executes it
  against your real data, and explains its reasoning.
- **Generates and runs real pandas code and SQL** (via DuckDB) — not just text —
  so answers are grounded in your actual numbers, not hallucinated.
- **Creates charts** (bar, line, pie, scatter, histogram, box) from your data.
- **Detects anomalies** using combined z-score + IQR statistics, with a
  plain-English explanation of *why* each value was flagged.
- **Maintains conversation context** across turns in a session.
- **One-click quick actions**: executive summary, trends, top performers,
  underperformers, anomaly scan, data quality check.
- **Exports the session** as a Markdown report you can download and share.
- Basic **response caching** and **structured logging** for observability.

---

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph UI["Streamlit UI (app/main.py)"]
        A[CSV Upload] --> B[DataManager]
        C[Chat Input / Quick Actions] --> D[LLMClient.ask]
        D --> E[Rendered Answer + Charts/Tables]
    end

    subgraph Agent["Agentic Reasoning Loop"]
        D --> F[Groq — Llama 3.3 70B (OpenAI-compatible chat completions)]
        F -->|tool_use| G[Tool Dispatcher]
        G -->|tool_result| F
        F -->|final text| E
    end

    subgraph Tools["Execution Layer (sandboxed / declarative)"]
        G --> H[run_pandas_code\nrestricted exec sandbox]
        G --> I[run_sql\nDuckDB, SELECT-only]
        G --> J[create_chart\nPlotly, declarative spec]
        G --> K[detect_anomalies\nz-score + IQR]
        G --> L[get_dataset_info\nschema / profile]
    end

    B --> H
    B --> I
    B --> J
    B --> K
    B --> L

    style Agent fill:#eef6ff
    style Tools fill:#f5f5f5
```

**Design principle:** The LLM is the *reasoner*, never the *executor*. It decides
which tool to call and with what arguments, but every tool call runs through a
safety layer before touching real data:

| Layer | Safety measure |
|---|---|
| `run_pandas_code` | AST-based static check blocks `import`, `open`, `exec`, `eval`, dangerous attributes; restricted builtins; wall-clock timeout |
| `run_sql` | DuckDB, keyword-blocklist for DDL/DML, only `SELECT`/`WITH` statements accepted |
| `create_chart` | Fully declarative spec (chart type/columns/agg) — no code execution at all |
| `detect_anomalies` | Pure statistics, no user-supplied code |

Tool call failures are returned as structured errors (not exceptions that crash
the app) so the model can read the error and retry with a corrected call.

### Project layout

```
app/
├── main.py              # Streamlit UI
├── config.py            # env-driven settings
├── core/
│   ├── data_manager.py  # CSV load/validate/profile, multi-file registry
│   ├── sandbox.py       # restricted pandas exec
│   ├── sql_engine.py    # DuckDB query execution
│   ├── chart_engine.py  # Plotly chart builder
│   ├── anomaly.py       # z-score + IQR anomaly detection
│   ├── tools.py         # tool schemas + dispatcher
│   ├── llm_client.py    # Groq agent loop
│   ├── insights.py      # quick-action prompt templates
│   └── report_export.py # session -> Markdown report
└── utils/logger.py       # structured logging
tests/                     # pytest unit tests (37 tests)
sample_data/sales_data.csv # synthetic dataset with injected anomalies
```

---

## 🚀 Getting started

### Option A — Docker (preferred)

```bash
git clone <this-repo-url>
cd ai-data-analyst
cp .env.example .env        # then edit .env and add your GROQ_API_KEY
docker compose up --build
```

Open **http://localhost:8501**.

### Option B — Local Python

Requires Python 3.11+.

```bash
git clone <this-repo-url>
cd ai-data-analyst
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env and add your GROQ_API_KEY
streamlit run app/main.py
```

Open **http://localhost:8501**. If you don't want to use a `.env` file, you can
also paste your API key directly into the **Settings** panel in the sidebar.

### Try it

1. Upload `sample_data/sales_data.csv` (or your own CSV).
2. Click a quick action like **Executive summary** or **Detect anomalies**, or
   type a question, e.g.:
   - *"Which region generated the highest revenue?"*
   - *"Show monthly sales trends as a line chart."*
   - *"Which products are underperforming?"*
   - *"Generate SQL to find the top 5 customers by revenue."*
   - *"Detect anomalies in the revenue column."*

---

## 🧪 Running tests

```bash
pip install -r requirements.txt
pytest
```

37 unit tests cover: CSV validation/parsing edge cases, sandbox security
(blocked imports/builtins, syntax/runtime error handling), SQL safety
(DDL/DML blocking), anomaly detection correctness, chart building, and the
tool dispatcher.

---

## ⚙️ Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | _(required)_ | Your Groq API key ([console.groq.com/keys](https://console.groq.com/keys)) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model used for reasoning |
| `MAX_TOKENS` | `2048` | Max tokens per LLM response |
| `MAX_AGENT_TURNS` | `6` | Max tool-call round-trips per question (loop guard) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ENABLE_CACHE` | `true` | Cache identical questions per dataset schema |

---

## 📝 Assumptions & implementation notes

- **Model choice**: defaults to `llama-3.3-70b-versatile` on Groq, chosen for its
  strong tool-use support and Groq's very low latency inference. Any tool-use-capable
  model hosted on Groq (e.g. `openai/gpt-oss-120b`, `llama-3.1-8b-instant` for lower
  cost/latency) can be swapped in via `GROQ_MODEL`.
- **Sandbox scope**: `run_pandas_code` uses AST-based static analysis plus a
  restricted builtins/globals environment. This is appropriate defense-in-depth
  for a local analytics tool, but for a multi-tenant production deployment I'd
  additionally run code execution in an isolated subprocess/container with no
  network access and a hard memory/CPU limit (e.g. `nsjail`, gVisor, or a
  short-lived Firecracker microVM) rather than in-process `exec()`.
- **SQL over pandas for safety-critical paths**: where an analysis maps cleanly
  to SQL, DuckDB is generally the safer and equally capable choice since SQL has
  no access to the filesystem, network, or Python runtime — it's used as the
  default recommendation, with pandas exec available for anything SQL can't
  express cleanly (e.g. certain in-place transforms).
- **Anomaly detection** combines z-score (parametric) and IQR (non-parametric,
  robust to skew) so that both roughly-normal metrics and skewed ones (e.g.
  revenue) get sensible outlier flags; a row is flagged if either method fires.
- **Type inference**: columns are auto-detected as datetime/numeric when >80%/90%
  of values parse cleanly, respectively, to avoid CSVs with a few dirty rows
  falling back to `object` dtype entirely.
- **Caching** is a simple in-session dict keyed on `(question, dataset schema)` —
  sufficient to avoid re-billing identical repeated questions in a demo/dev
  setting; a production version would use a shared cache (Redis) with TTL.
- **Chart generation is declarative, not code-generated** — the model returns a
  spec (`chart_type`, `x`, `y`, `agg`, ...) rather than plotting code, which
  removes an entire class of execution risk from the charting path.
- **Multi-file analysis**: each uploaded CSV becomes its own named dataset
  (derived from the filename) simultaneously queryable via pandas or SQL — you
  can ask cross-file questions like *"join orders and customers on customer_id"*
  and the model can write pandas/SQL that references both tables by name.
- **Sample dataset**: `sample_data/sales_data.csv` is synthetic (1,200 rows,
  2023–2024, with 8 intentionally injected anomalies — revenue spikes and
  zero-quantity errors) so anomaly detection and business-insight features have
  something meaningful to surface out of the box.

## 🔭 Possible next steps (not implemented)

- Authentication / per-user data isolation for multi-tenant deployment
- Forecasting (e.g. Prophet/ARIMA) as an additional tool
- Semantic search over historical Q&A / dataset metadata
- A proper eval harness with a labeled set of question → expected-answer pairs
# AI-powered-Data-Analyst
