from __future__ import annotations

import uuid
import pandas as pd
import streamlit as st

from src.talk_to_data.graph import CreditRiskAgent, create_persistent_agent
from src.ui.styles import get_svg_icon, render_executive_briefing


@st.cache_resource
def load_agent() -> CreditRiskAgent:
    return create_persistent_agent()


def init_chat_session(agent: CreditRiskAgent):
    if "chat_thread_id" not in st.session_state:
        requested_thread = st.query_params.get("thread", "")
        if requested_thread and len(requested_thread) <= 64:
            st.session_state["chat_thread_id"] = requested_thread
        else:
            st.session_state["chat_thread_id"] = str(uuid.uuid4())
            st.query_params["thread"] = st.session_state["chat_thread_id"]
    if "messages" not in st.session_state:
        persisted = agent.get_history(st.session_state["chat_thread_id"])
        st.session_state["messages"] = []
        for turn in persisted:
            st.session_state["messages"].append({"role": "user", "content": turn.get("question", "")})
            result = turn.get("query_result") or {}
            st.session_state["messages"].append({
                "role": "assistant",
                "content": turn.get("answer", ""),
                "tool_used": turn.get("tool_used", "Platform"),
                "sql": turn.get("sql"),
                "query_meta": result,
                "data_preview": result.get("preview"),
                "sources": turn.get("sources") or [],
            })
        if not st.session_state["messages"]:
            st.session_state["messages"] = [{
                "role": "assistant",
                "content": (
                    "Hello, I am the **NeoStats Credit Risk Intelligence Agent**.\n\n"
                    "I can execute validated read-only SQL queries on our **PostgreSQL Analytics Database**, "
                    "search the **Curated Credit Risk Knowledge Base**, or perform external searches via **DDGS**.\n\n"
                    "How can I assist your portfolio analysis or underwriting inquiries today?"
                ),
                "tool_used": "Platform",
            }]


SUGGESTED_QUESTIONS = [
    "What is the overall default rate across the 307,511 portfolio?",
    "Which age band has the highest default rate?",
    "Compare default rates across education levels.",
    "How does previous late-payment behaviour relate to default risk?",
    "What does EXT_SOURCE_3 mean?",
    "Why was EBM selected over LightGBM for production?",
    "What is probability of default in general? Use external sources.",
    "DROP TABLE analytics.applicants; (Security Injection Test)",
]


def render_chatbot_tab():
    # Executive Briefing Callout
    render_executive_briefing(
        title="Multi-Tool Autonomous Agent & Read-Only Governance",
        description=(
            "This conversational intelligence assistant is powered by a LangGraph StateGraph engine using Gemini 3.5 Flash-Lite by default. "
            "The agent dynamically routes natural language queries across three sandboxed tools: (1) PostgreSQL Analytics "
            "for verified database queries, (2) Curated Knowledge Base for credit policy retrieval, and (3) DuckDuckGo for live "
            "external financial research. An Abstract Syntax Tree (AST) SQL validator enforces strict read-only execution."
        ),
        takeaways=[
            "Engine: LangGraph + Gemini 3.5 Flash-Lite",
            "Tools: PostgreSQL, Policy KB, Web Search",
            "Security: AST Read-Only SQL Parser",
            "State Persistence: PostgreSQL Checkpoints",
        ],
        icon="robot",
    )

    agent = load_agent()
    init_chat_session(agent)

    col_info, col_reset = st.columns([0.75, 0.25])
    with col_info:
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: #94a3b8; padding-top: 6px; display:flex; align-items:center; gap:6px;">
                {get_svg_icon("server", "#34d399", 14)} <span>Thread Session ID: <code style="color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 3px 8px; border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px;">{st.session_state['chat_thread_id'][:8]}</code> ({'Persisted in PostgreSQL' if agent.persistence_enabled else 'Session-only fallback'})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_reset:
        if st.button("Reset Conversation", width="stretch"):
            st.session_state["chat_thread_id"] = str(uuid.uuid4())
            st.query_params["thread"] = st.session_state["chat_thread_id"]
            st.session_state["messages"] = []
            st.rerun()

    # Quick Suggestion Chips inside neat container
    selected_prompt = None
    with st.expander("Suggested Exploratory Queries (Click to Auto-Run)", expanded=True):
        col_left, col_right = st.columns(2)
        for idx, q in enumerate(SUGGESTED_QUESTIONS):
            target_col = col_left if idx % 2 == 0 else col_right
            with target_col:
                if st.button(q, key=f"sq_{idx}", width="stretch"):
                    selected_prompt = q

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Render Chat History
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                if msg.get("tool_used") and msg["role"] == "assistant":
                    tool_name = msg["tool_used"]
                    if "Database" in tool_name:
                        icon_svg = get_svg_icon("database", "#38bdf8", 13, "margin-right:5px;")
                    elif "Knowledge" in tool_name:
                        icon_svg = get_svg_icon("book", "#10b981", 13, "margin-right:5px;")
                    elif "Platform" in tool_name:
                        icon_svg = get_svg_icon("shield-check", "#34d399", 13, "margin-right:5px;")
                    else:
                        icon_svg = get_svg_icon("globe", "#f59e0b", 13, "margin-right:5px;")
                    st.markdown(
                        f'<span class="chat-tool-badge" style="display:inline-flex; align-items:center;">{icon_svg} Tool Used: {tool_name}</span>',
                        unsafe_allow_html=True,
                    )

                st.markdown(msg["content"])

                # Expandable details if SQL query was run
                if msg.get("sql"):
                    with st.expander("Executed SQL & Query Latency", expanded=False):
                        st.code(msg["sql"], language="sql")
                        if msg.get("query_meta"):
                            meta = msg["query_meta"]
                            st.caption(f"Rows Returned: {meta.get('row_count', 0)} | Execution Latency: {meta.get('execution_ms', 0)} ms")

                        # Data table preview if available
                        if msg.get("data_preview"):
                            df_preview = pd.DataFrame(msg["data_preview"])
                            st.dataframe(df_preview, width="stretch")

                # Expandable sources if web or knowledge search
                if msg.get("sources"):
                    with st.expander("Source Citations & Documentation", expanded=False):
                        for s in msg["sources"]:
                            if str(s).startswith("http"):
                                st.markdown(f"- {get_svg_icon('globe', '#38bdf8', 12, 'margin-right:4px;')} [{s}]({s})", unsafe_allow_html=True)
                            else:
                                st.markdown(f"- {get_svg_icon('book', '#94a3b8', 12, 'margin-right:4px;')} {s}", unsafe_allow_html=True)

    # Chat Input
    user_input = st.chat_input("Enter a portfolio, underwriting, or credit-risk inquiry...")
    prompt_to_run = selected_prompt or user_input

    if prompt_to_run:
        st.session_state["messages"].append({"role": "user", "content": prompt_to_run})

        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing inquiry with agentic state machine..."):
                thread_id = st.session_state["chat_thread_id"]
                response = agent.ask(prompt_to_run, thread_id=thread_id)

                answer = response.get("final_answer", "No answer generated.")
                tool_used = response.get("tool_used", "Database Query")
                sql = response.get("sql")
                query_res = response.get("query_result") or {}
                sources = response.get("sources") or []

                if "Database" in tool_used:
                    icon_svg = get_svg_icon("database", "#38bdf8", 13, "margin-right:5px;")
                elif "Knowledge" in tool_used:
                    icon_svg = get_svg_icon("book", "#10b981", 13, "margin-right:5px;")
                else:
                    icon_svg = get_svg_icon("globe", "#f59e0b", 13, "margin-right:5px;")

                st.markdown(
                    f'<span class="chat-tool-badge" style="display:inline-flex; align-items:center;">{icon_svg} Tool Used: {tool_used}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(answer)

                if sql:
                    with st.expander("Executed SQL & Query Latency", expanded=False):
                        st.code(sql, language="sql")
                        if query_res:
                            st.caption(f"Rows Returned: {query_res.get('row_count', 0)} | Execution Latency: {query_res.get('execution_ms', 0)} ms")
                            if query_res.get("preview"):
                                st.dataframe(pd.DataFrame(query_res["preview"]), width="stretch")

                if sources:
                    with st.expander("Source Citations & Documentation", expanded=False):
                        for s in sources:
                            if str(s).startswith("http"):
                                st.markdown(f"- {get_svg_icon('globe', '#38bdf8', 12, 'margin-right:4px;')} [{s}]({s})", unsafe_allow_html=True)
                            else:
                                st.markdown(f"- {get_svg_icon('book', '#94a3b8', 12, 'margin-right:4px;')} {s}", unsafe_allow_html=True)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "tool_used": tool_used,
                    "sql": sql,
                    "query_meta": query_res,
                    "data_preview": query_res.get("preview") if query_res else None,
                    "sources": sources,
                })
