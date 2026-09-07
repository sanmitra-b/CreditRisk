from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine

from src.db.connection import create_database_engine
from src.db.knowledge_base import search_knowledge_base
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class KnowledgeSearchResult:
    success: bool
    query: str
    chunks: list[dict[str, Any]]
    formatted_evidence: str
    error: str | None = None


class KnowledgeBaseSearcher:
    def __init__(self, engine: Engine | None = None):
        self.engine = engine or create_database_engine(read_only=True)

    def search(self, query: str, limit: int = 5) -> KnowledgeSearchResult:
        try:
            chunks = search_knowledge_base(self.engine, query, limit=limit)
            if not chunks:
                return KnowledgeSearchResult(
                    success=True,
                    query=query,
                    chunks=[],
                    formatted_evidence="No matching knowledge base documents found.",
                )

            evidence_parts = []
            for idx, c in enumerate(chunks, 1):
                heading = c.get("heading") or "Knowledge Document"
                source = c.get("source_name", "Curated")
                content = c.get("content", "").strip()
                evidence_parts.append(
                    f"[{idx}] {heading} (Source: {source})\n{content}"
                )

            return KnowledgeSearchResult(
                success=True,
                query=query,
                chunks=chunks,
                formatted_evidence="\n\n".join(evidence_parts),
            )
        except Exception as e:
            LOGGER.error("Knowledge search failed: %s", e)
            return KnowledgeSearchResult(
                success=False,
                query=query,
                chunks=[],
                formatted_evidence="",
                error=str(e),
            )
