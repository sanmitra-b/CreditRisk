from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Any, List, Optional, TypedDict

from google import genai
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.db.connection import is_transaction_pooler_url
from src.talk_to_data.knowledge_search import KnowledgeBaseSearcher
from src.talk_to_data.nl_to_sql import NLToSQLGenerator
from src.talk_to_data.prompt_templates import SYNTHESIS_SYSTEM_PROMPT
from src.talk_to_data.query_runner import DatabaseQueryRunner
from src.talk_to_data.router import QueryRouter
from src.talk_to_data.web_search import WebSearchTool
from src.utils.config import get_settings
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class AgentState(TypedDict, total=False):
    messages: List[dict[str, Any]]
    query: str
    route: str
    route_reasoning: str
    sql: Optional[str]
    sql_valid: bool
    sql_error: Optional[str]
    sql_retry_count: int
    query_result: Optional[dict[str, Any]]
    evidence: str
    tool_used: str
    final_answer: str
    sources: List[str]
    conversation: Annotated[List[dict[str, Any]], operator.add]


class CreditRiskAgent:
    def __init__(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        settings = get_settings()
        self.settings = settings
        self.client = None
        if settings.gemini_api_key:
            self.client = genai.Client(api_key=settings.gemini_api_key)

        self.router = QueryRouter(client=self.client)
        self.nl_to_sql = NLToSQLGenerator(client=self.client)
        self.query_runner = DatabaseQueryRunner()
        self.kb_searcher = KnowledgeBaseSearcher()
        self.web_searcher = WebSearchTool()
        self.checkpointer = checkpointer
        self.persistence_enabled = checkpointer is not None

        self.app = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("router", self._router_node)
        builder.add_node("generate_sql", self._generate_sql_node)
        builder.add_node("execute_sql", self._execute_sql_node)
        builder.add_node("repair_sql", self._repair_sql_node)
        builder.add_node("search_kb", self._search_kb_node)
        builder.add_node("search_web", self._search_web_node)
        builder.add_node("synthesize", self._synthesize_node)

        builder.add_edge(START, "router")

        def route_decision(state: AgentState) -> str:
            route = state.get("route", "query_database")
            if route == "search_knowledge_base":
                return "search_kb"
            elif route == "search_web":
                return "search_web"
            elif route == "direct_response":
                return "synthesize"
            return "generate_sql"

        builder.add_conditional_edges(
            "router",
            route_decision,
            {
                "generate_sql": "generate_sql",
                "search_kb": "search_kb",
                "search_web": "search_web",
                "synthesize": "synthesize",
            },
        )

        builder.add_edge("generate_sql", "execute_sql")

        def sql_execution_decision(state: AgentState) -> str:
            sql_error = state.get("sql_error")
            retry_count = state.get("sql_retry_count", 0)

            # Prohibited operations or comment injection must be aborted immediately without repair
            if sql_error and ("Prohibited operation" in sql_error or "comments are strictly prohibited" in sql_error or "is not permitted" in sql_error or "Only SELECT" in sql_error):
                return "synthesize"

            if sql_error and retry_count < 1:
                return "repair_sql"
            return "synthesize"

        builder.add_conditional_edges(
            "execute_sql",
            sql_execution_decision,
            {
                "repair_sql": "repair_sql",
                "synthesize": "synthesize",
            },
        )

        builder.add_edge("repair_sql", "execute_sql")
        builder.add_edge("search_kb", "synthesize")
        builder.add_edge("search_web", "synthesize")
        builder.add_edge("synthesize", END)

        return builder.compile(checkpointer=self.checkpointer)

    def _router_node(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        normalized = query.lower()
        prohibited_request = re.search(
            r"\b(drop|delete|update|insert|alter|truncate|create|grant|revoke)\b",
            normalized,
        )
        if prohibited_request:
            operation = prohibited_request.group(1).upper()
            return {
                "route": "direct_response",
                "route_reasoning": "Destructive or mutating request blocked before routing",
                "sql_retry_count": 0,
                "sql": None,
                "sql_valid": False,
                "sql_error": (
                    f"Prohibited operation {operation}. The analytics assistant accepts "
                    "read-only questions and never executes DDL or DML."
                ),
                "query_result": None,
                "evidence": "",
                "tool_used": "SQL Safety Guard",
                "sources": [],
            }
        follow_up_markers = (
            "previous question",
            "previous response",
            "previous answer",
            "earlier question",
            "earlier response",
            "above response",
            "summarize that",
            "summarise that",
        )
        if state.get("conversation") and any(marker in normalized for marker in follow_up_markers):
            return {
                "route": "direct_response",
                "route_reasoning": "Follow-up resolved from persisted conversation history",
                "sql_retry_count": 0,
                "sql": None,
                "sql_valid": False,
                "sql_error": None,
                "query_result": None,
                "evidence": "",
                "tool_used": "Conversation Memory",
                "sources": [],
            }
        decision = self.router.route(query)
        LOGGER.info("Router selected: %s (%s)", decision.tool, decision.reasoning)
        return {
            "route": decision.tool,
            "route_reasoning": decision.reasoning,
            "sql_retry_count": 0,
            "sql": None,
            "sql_valid": False,
            "sql_error": None,
            "query_result": None,
            "evidence": "",
            "tool_used": "",
            "sources": [],
        }

    def _generate_sql_node(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "").strip()
        first_word = query.split()[0].upper() if query else ""
        if first_word in ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "SELECT"):
            sql = query
        else:
            sql = self.nl_to_sql.generate_sql(query)
        return {"sql": sql, "tool_used": "Database Query"}

    def _execute_sql_node(self, state: AgentState) -> dict[str, Any]:
        sql = state.get("sql", "")
        res = self.query_runner.execute_query(sql)
        if res.success:
            evidence = f"Executed SQL: {res.sql}\nRows: {res.row_count}\nExecution time: {res.execution_ms} ms\n\nResults:\n"
            if not res.data.empty:
                evidence += res.data.to_string(index=False)
            else:
                evidence += "No records matched the criteria."
            return {
                "sql": res.sql,
                "sql_valid": True,
                "sql_error": None,
                "query_result": res.to_dict(),
                "evidence": evidence,
                "tool_used": "Database Query",
                "sources": ["analytics database"],
            }
        else:
            return {
                "sql": res.sql,
                "sql_valid": False,
                "sql_error": res.error,
                "evidence": f"SQL Error: {res.error}",
                "tool_used": "Database Query",
            }

    def _repair_sql_node(self, state: AgentState) -> dict[str, Any]:
        current_sql = state.get("sql", "")
        error_msg = state.get("sql_error", "Query execution failed")
        retries = state.get("sql_retry_count", 0) + 1
        LOGGER.info("Attempting SQL repair (attempt %d) for error: %s", retries, error_msg)
        repaired_sql = self.nl_to_sql.repair_sql(current_sql, error_msg)
        return {"sql": repaired_sql, "sql_retry_count": retries}

    def _search_kb_node(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        kb_res = self.kb_searcher.search(query)
        sources = list({c.get("source_name", "Knowledge Base") for c in kb_res.chunks})
        return {
            "evidence": kb_res.formatted_evidence,
            "tool_used": "Knowledge Base",
            "sources": sources or ["Internal Credit Risk Knowledge Base"],
        }

    def _search_web_node(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        web_res = self.web_searcher.search(query)
        sources = [r["url"] for r in web_res.results if "url" in r]
        return {
            "evidence": web_res.formatted_evidence,
            "tool_used": "Web Search (DDGS)",
            "sources": sources,
        }

    def _synthesize_node(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        evidence = state.get("evidence", "")
        tool_used = state.get("tool_used", "Platform")
        sql_error = state.get("sql_error")

        if sql_error:
            answer = f"I cannot execute that query safely: {sql_error}"
            return self._final_response(state, answer, tool_used)

        if state.get("route") == "direct_response" and not evidence:
            previous_turns = state.get("conversation", [])[-3:]
            if previous_turns:
                evidence = "\n\n".join(
                    f"Question: {turn.get('question', '')}\nAnswer: {turn.get('answer', '')}"
                    for turn in previous_turns
                )
                tool_used = "Conversation Memory"

        if self.client is not None and evidence and evidence != "No records matched the criteria.":
            try:
                prompt = (
                    f"{SYNTHESIS_SYSTEM_PROMPT}\n\n"
                    f"User Question: {query}\n"
                    f"Tool Used: {tool_used}\n"
                    f"Evidence:\n{evidence}\n\n"
                    "Final Answer:"
                )
                response = self.client.models.generate_content(
                    model=self.settings.gemini_default_model,
                    contents=prompt,
                )
                final_text = response.text.strip()
                return self._final_response(state, final_text, tool_used)
            except Exception as e:
                LOGGER.warning("Gemini synthesis failed (%s), using direct evidence formatting", e)

        if not evidence or evidence == "No records matched the criteria.":
            answer = "No matching records or evidence could be found to answer this inquiry."
        else:
            answer = f"Based on {tool_used}:\n\n{evidence}"

        return self._final_response(state, answer, tool_used)

    @staticmethod
    def _final_response(state: AgentState, answer: str, tool_used: str) -> dict[str, Any]:
        record = {
            "question": state.get("query", ""),
            "answer": answer,
            "tool_used": tool_used,
            "sql": state.get("sql"),
            "query_result": state.get("query_result") or {},
            "sources": state.get("sources") or [],
        }
        return {
            "final_answer": answer,
            "tool_used": tool_used,
            "conversation": [record],
        }

    def ask(self, query: str, thread_id: str = "default_session") -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        initial_state: AgentState = {
            "query": query,
            "messages": [{"role": "user", "content": query}],
        }
        output = self.app.invoke(initial_state, config=config)
        return output

    def get_history(self, thread_id: str) -> list[dict[str, Any]]:
        if not self.persistence_enabled:
            return []
        snapshot = self.app.get_state({"configurable": {"thread_id": thread_id}})
        return list(snapshot.values.get("conversation", [])) if snapshot.values else []


def create_persistent_agent() -> CreditRiskAgent:
    """Create an agent backed by durable PostgreSQL LangGraph checkpoints."""
    settings = get_settings()
    if not settings.database_url:
        LOGGER.warning("DATABASE_URL is unavailable; chatbot persistence is disabled")
        return CreditRiskAgent()

    transaction_pooler = is_transaction_pooler_url(settings.database_url)
    conninfo = settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    try:
        pool = ConnectionPool(
            conninfo=conninfo,
            min_size=1,
            max_size=5,
            kwargs={
                "autocommit": True,
                "prepare_threshold": None if transaction_pooler else 0,
                "row_factory": dict_row,
            },
        )
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        agent = CreditRiskAgent(checkpointer=checkpointer)
        agent._checkpoint_pool = pool
        return agent
    except Exception as exc:
        LOGGER.warning("PostgreSQL checkpoint setup failed; using non-persistent chat: %s", exc)
        return CreditRiskAgent()
