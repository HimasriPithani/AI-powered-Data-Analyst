"""
Streamlit front-end for the AI-powered Data Analyst.

Run with:  streamlit run app/main.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st

# Allow `streamlit run app/main.py` from repo root without package install
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core.data_manager import DataManager, DataValidationError
from app.core.insights import QUICK_PROMPTS
from app.core.llm_client import LLMClient
from app.core.report_export import build_markdown_report
from app.utils.logger import get_logger

logger = get_logger("main")

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

QUICK_PROMPT_ICONS = {
    "Executive summary": "📋",
    "Revenue/trend over time": "📈",
    "Top performers": "🏆",
    "Underperformers": "📉",
    "Detect anomalies": "🚨",
    "Data quality check": "🧹",
}

SAMPLE_CSV_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "sales_data.csv"


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

        :root {
            --accent: #4F46E5;
            --accent-2: #7C3AED;
            --card-border: #E5E7EB;
            --muted: #6B7280;
        }

        .stApp { background: #F8F9FC; }

        /* Header */
        .app-header { padding: 0.25rem 0 0.5rem 0; }
        .app-header h1 {
            font-size: 2.15rem; font-weight: 800; margin: 0 0 0.1rem 0; line-height: 1.2;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .app-header p { color: var(--muted); font-size: 0.95rem; margin: 0; }

        /* Metric cards */
        .metric-card {
            background: white; border: 1px solid var(--card-border); border-radius: 14px;
            padding: 0.9rem 1rem; text-align: center; box-shadow: 0 1px 3px rgba(16,24,40,0.04);
        }
        .metric-card .value { font-size: 1.55rem; font-weight: 700; color: #111827; }
        .metric-card .label {
            font-size: 0.72rem; color: var(--muted); text-transform: uppercase;
            letter-spacing: 0.04em; margin-top: 2px;
        }

        /* Buttons */
        .stButton button {
            border-radius: 10px; font-weight: 600; transition: all 0.15s ease;
        }
        .stButton button:hover { border-color: var(--accent); color: var(--accent); }

        /* Chat bubbles */
        [data-testid="stChatMessage"] {
            border-radius: 16px; padding: 0.5rem 0.3rem; margin-bottom: 0.35rem;
        }
        [data-testid="stChatMessageContent"] { font-size: 0.95rem; }

        /* Expanders */
        details {
            border-radius: 10px !important; border: 1px solid var(--card-border) !important;
        }

        /* Dataset chips */
        .chip {
            display: inline-block; background: #EEF2FF; color: #4338CA; font-size: 0.72rem;
            font-weight: 600; padding: 3px 10px; border-radius: 999px; margin: 2px 5px 4px 0;
        }
        .chip.warn { background: #FEF3C7; color: #92400E; }
        .chip.ok { background: #D1FAE5; color: #065F46; }

        /* Empty state */
        .empty-state {
            background: white; border: 1px dashed var(--card-border); border-radius: 16px;
            padding: 2.5rem 1.5rem; text-align: center; color: var(--muted);
        }
        .empty-state .big-emoji { font-size: 2.4rem; margin-bottom: 0.5rem; }

        section[data-testid="stSidebar"] .stButton button { width: 100%; }
        
        /* ==========================
        Sidebar text fixes
        ========================== */

        section[data-testid="stSidebar"] {
            color: #E5E7EB !important;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div:not([data-testid]) {
            color: #E5E7EB !important;
        }

        /* Success / Warning / Info messages */
        section[data-testid="stSidebar"] div[data-testid="stAlert"] {
            color: #111827 !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stAlert"] * {
            color: #111827 !important;
        }
        
        /* Only assistant/user chat messages */
        [data-testid="stChatMessageContent"] {
            color: #111827 !important;
        }

        [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessageContent"] li,
        [data-testid="stChatMessageContent"] span,
        [data-testid="stChatMessageContent"] div {
            color: #111827 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(col, label: str, value: str):
    col.markdown(
        f"""<div class="metric-card"><div class="value">{value}</div>
        <div class="label">{label}</div></div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #
def init_state():
    if "dm" not in st.session_state:
        st.session_state.dm = DataManager()
    if "chat_history" not in st.session_state:
        # Each turn: {"role": "user"/"assistant", "content": str, "steps": [ {tool_name, tool_input, tool_result} ]}
        st.session_state.chat_history = []
    if "llm_messages" not in st.session_state:
        st.session_state.llm_messages = []  # raw Groq (OpenAI-style) message history
    if "tool_cache" not in st.session_state:
        st.session_state.tool_cache = {}
    if "api_key_override" not in st.session_state:
        st.session_state.api_key_override = ""


inject_css()
init_state()
dm: DataManager = st.session_state.dm


# --------------------------------------------------------------------------- #
# Sidebar: upload, settings, data quality
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 📊 AI Data Analyst")
    st.caption("Upload CSVs, then ask questions in plain English.")

    with st.expander("⚙️ Settings", expanded=not settings.groq_api_key):

        if settings.groq_api_key:
            st.success("✅ Groq API key loaded from .env")
        else:
            st.warning("⚠️ No Groq API key found in the environment.")

            st.session_state.api_key_override = st.text_input(
                "Groq API key",
                value=st.session_state.api_key_override,
                type="password",
                help=(
                    "Only needed if GROQ_API_KEY isn't set in the .env file. "
                    "Get a free key at https://console.groq.com/keys"
                ),
            )

        st.caption(f"Model: `{settings.model_name}`")

    st.divider()
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files", type=["csv"], accept_multiple_files=True
    )

    if SAMPLE_CSV_PATH.exists():
        if st.button("✨ Try it with a sample dataset", use_container_width=True):
            key_guess = DataManager._safe_key(SAMPLE_CSV_PATH.name)
            if key_guess not in dm.frames:
                try:
                    profile = dm.load_csv(SAMPLE_CSV_PATH.name, SAMPLE_CSV_PATH.read_bytes())
                    st.success(f"Loaded **{SAMPLE_CSV_PATH.name}** → `{profile.name}` ({profile.n_rows} rows)")
                except DataValidationError as e:
                    st.error(str(e))
            st.rerun()

    if uploaded_files:
        for f in uploaded_files:
            key_guess = DataManager._safe_key(f.name)
            if key_guess in dm.frames:
                continue
            try:
                profile = dm.load_csv(f.name, f.read())
                st.success(f"Loaded **{f.name}** → `{profile.name}` ({profile.n_rows} rows)")
                if profile.quality_warnings:
                    for w in profile.quality_warnings:
                        st.warning(w)
            except DataValidationError as e:
                st.error(str(e))

    if dm.frames:
        st.divider()
        st.subheader("Loaded datasets")
        for name, profile in dm.profiles.items():
            with st.expander(f"`{name}` — {profile.n_rows} rows × {profile.n_cols} cols"):
                chips = f'<span class="chip">{profile.n_rows} rows</span><span class="chip">{profile.n_cols} cols</span>'
                if profile.quality_warnings:
                    chips += f'<span class="chip warn">{len(profile.quality_warnings)} warning(s)</span>'
                else:
                    chips += '<span class="chip ok">clean</span>'
                st.markdown(chips, unsafe_allow_html=True)
                st.write(", ".join(profile.columns))
                st.dataframe(pd.DataFrame(profile.sample), use_container_width=True, height=150)

        if st.button("🗑️ Clear all data & chat", use_container_width=True):
            for key in ["dm", "chat_history", "llm_messages", "tool_cache"]:
                del st.session_state[key]
            st.rerun()

    st.divider()
    if st.session_state.chat_history:
        report_md = build_markdown_report(list(dm.frames.keys()), st.session_state.chat_history)
        st.download_button(
            "⬇️ Export session report (.md)",
            data=report_md,
            file_name="ai_data_analyst_report.md",
            mime="text/markdown",
            use_container_width=True,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_llm_client() -> LLMClient | None:
    api_key = st.session_state.api_key_override or settings.groq_api_key
    if not api_key:
        return None
    try:
        return LLMClient(api_key=api_key)
    except ValueError:
        return None


def cache_key(question: str) -> str:
    schema_fingerprint = "|".join(sorted(dm.frames.keys()))
    raw = f"{question}::{schema_fingerprint}"
    return hashlib.sha256(raw.encode()).hexdigest()


def render_step(step: dict):
    """Render a single executed tool call inline (table/chart/anomaly panel)."""
    tool_name = step["tool_name"]
    tool_input = step["tool_input"]
    result = step["tool_result"]

    if tool_name == "create_chart" and result.get("success"):
        fig = pio.from_json(result["figure_json"])
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{step.get('_id')}")
        return

    if tool_name in {"run_pandas_code", "run_sql"} and result.get("success"):
        payload = result.get("result", result)
        rows = None
        if isinstance(payload, dict) and payload.get("type") == "dataframe":
            rows = payload["rows"]
        elif isinstance(payload, dict) and "rows" in payload:
            rows = payload["rows"]
        if rows:
            with st.expander(f"🔎 {tool_name} result", expanded=False):
                if tool_name == "run_sql":
                    st.code(tool_input.get("query", ""), language="sql")
                else:
                    st.code(tool_input.get("code", ""), language="python")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        return

    if tool_name == "detect_anomalies" and result.get("success"):
        with st.expander(
            f"🚨 Anomaly scan on `{result['column']}` — {result['n_anomalies']} flagged", expanded=True
        ):
            st.caption(f"Method: {result['method']}")
            if result["anomalies"]:
                st.dataframe(pd.DataFrame(result["anomalies"]), use_container_width=True)
            else:
                st.info("No anomalies found with the current threshold.")
        return

    if not result.get("success"):
        st.error(f"Tool `{tool_name}` failed: {result.get('error')}")


def render_turn(turn: dict, idx: int):
    avatar = "🧑" if turn["role"] == "user" else "🤖"
    with st.chat_message(turn["role"], avatar=avatar):
        st.markdown(turn["content"])
        for j, step in enumerate(turn.get("steps", [])):
            step = {**step, "_id": f"{idx}_{j}"}
            render_step(step)
        if turn.get("steps"):
            with st.expander("🧠 Reasoning trace (tools used)", expanded=False):
                for s in turn["steps"]:
                    status = "✅" if s["tool_result"].get("success") else "❌"
                    st.markdown(
                        f"{status} **{s['tool_name']}** — "
                        f"`{json.dumps(s['tool_input'], default=str)[:300]}`"
                    )


def run_question(question: str):
    client = get_llm_client()
    if client is None:
        st.error("Please set a Groq API key in the sidebar Settings panel.")
        return
    if dm.is_empty():
        st.error("Please upload at least one CSV file first.")
        return

    st.session_state.chat_history.append({"role": "user", "content": question, "steps": []})

    ck = cache_key(question)
    if settings.enable_cache and ck in st.session_state.tool_cache:
        response = st.session_state.tool_cache[ck]
    else:
        with st.spinner("Analyzing your data..."):
            response = client.ask(
                question=question,
                dm=dm,
                conversation_history=st.session_state.llm_messages,
            )
        if settings.enable_cache and response.final_text:
            st.session_state.tool_cache[ck] = response

    st.session_state.llm_messages = response.raw_messages

    steps = [
        {"tool_name": s.tool_name, "tool_input": s.tool_input, "tool_result": s.tool_result}
        for s in response.steps
    ]
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response.final_text,
        "steps": steps,
    })
    st.rerun()


# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #
st.markdown(
    """<div class="app-header"><h1>Chat with your data</h1>
    <p>Upload CSVs and ask questions in plain English — powered by Groq + tool calling.</p></div>""",
    unsafe_allow_html=True,
)

if dm.is_empty():
    st.markdown(
        """
        <div class="empty-state">
            <div class="big-emoji">👈📁</div>
            <b>Upload one or more CSV files in the sidebar to get started.</b><br/>
            No file yet? Try the bundled sample dataset from the sidebar button.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    total_rows = sum(p.n_rows for p in dm.profiles.values())
    total_warnings = sum(len(p.quality_warnings) for p in dm.profiles.values())
    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, "Datasets loaded", str(len(dm.profiles)))
    metric_card(m2, "Total rows", f"{total_rows:,}")
    metric_card(m3, "Quality warnings", str(total_warnings))
    metric_card(m4, "Chat turns", str(len(st.session_state.chat_history) // 2))

    st.write("")
    st.caption("Quick actions")
    cols = st.columns(len(QUICK_PROMPTS))
    for col, (label, prompt) in zip(cols, QUICK_PROMPTS.items()):
        icon = QUICK_PROMPT_ICONS.get(label, "▶️")
        if col.button(f"{icon} {label}", use_container_width=True):
            run_question(prompt)

    st.write("")

# Replay full chat history (single source of truth for rendering)
for i, turn in enumerate(st.session_state.chat_history):
    render_turn(turn, i)

user_question = st.chat_input("Ask a question about your data...", disabled=dm.is_empty())
if user_question:
    run_question(user_question)
