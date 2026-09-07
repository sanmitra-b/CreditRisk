from __future__ import annotations

import json
import re
from typing import Literal

from google import genai
from pydantic import BaseModel, Field

from src.talk_to_data.prompt_templates import ROUTER_SYSTEM_PROMPT
from src.utils.config import get_settings
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

ToolType = Literal["query_database", "search_knowledge_base", "search_web", "direct_response"]


class RouterDecision(BaseModel):
    tool: ToolType = Field(description="The chosen tool")
    reasoning: str = Field(default="", description="Reasoning for tool selection")


DETERMINISTIC_RULES = [
    # Knowledge Base patterns
    (r"\b(ext_source_\d|amt_annuity|amt_credit|cnt_children|days_birth|days_employed)\b", "search_knowledge_base"),
    (r"\b(why was ebm|ebm rationale|why ebm|ebm selected|production model)\b", "search_knowledge_base"),
    (r"\b(what does .* mean|definition of .*|explain the feature)\b", "search_knowledge_base"),
    (r"\b(surrogate (tree|rules)|business rules|rule agreement)\b", "search_knowledge_base"),
    
    # Web search patterns
    (r"\b(in general|general.*definition|external sources|industry standard|basel)\b", "search_web"),
    (r"\b(what is probability of default in general)\b", "search_web"),
    
    # Database query patterns
    (r"\b(overall default rate|default rate|portfolio|highest.*default|compare.*default)\b", "query_database"),
    (r"\b(age band|education level|housing type|late.*payment|overdue.*history)\b", "query_database"),
    (r"\b(how many applicants|average income|average credit|count.*applicants)\b", "query_database"),
    (r"\b(drop table|delete from|update analytics|select .* from)\b", "query_database"),
]


class QueryRouter:
    def __init__(self, client: genai.Client | None = None):
        settings = get_settings()
        self.client = client
        if self.client is None and settings.gemini_api_key:
            self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_default_model

    def route(self, query: str) -> RouterDecision:
        normalized = query.lower().strip()

        # 1. Deterministic pattern check
        for pattern, tool in DETERMINISTIC_RULES:
            if re.search(pattern, normalized):
                LOGGER.info("Deterministic route match: '%s' -> %s", query, tool)
                return RouterDecision(tool=tool, reasoning=f"Matched deterministic pattern: {pattern}")

        # 2. LLM classification if Gemini client is available
        if self.client is not None:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"{ROUTER_SYSTEM_PROMPT}\n\nUser Question: {query}",
                    config={"response_mime_type": "application/json"},
                )
                text_response = response.text.strip()
                parsed = json.loads(text_response)
                tool = parsed.get("tool", "query_database")
                if tool not in ("query_database", "search_knowledge_base", "search_web", "direct_response"):
                    tool = "query_database"
                return RouterDecision(tool=tool, reasoning=parsed.get("reasoning", "LLM router classification"))
            except Exception as e:
                LOGGER.warning("LLM routing failed (%s), defaulting to query_database", e)

        # Fallback default
        return RouterDecision(tool="query_database", reasoning="Default fallback routing")
