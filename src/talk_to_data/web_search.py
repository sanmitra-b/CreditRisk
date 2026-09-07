from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ddgs import DDGS

from src.utils.config import get_settings
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

GENERAL_FALLBACKS = {
    "probability of default": (
        "Probability of default (PD) is the estimated likelihood that a borrower will "
        "default during a defined time horizon. It is a forward-looking credit-risk "
        "measure and should always be interpreted together with its horizon, default "
        "definition, model population, and calibration date."
    ),
}


@dataclass
class WebSearchResult:
    success: bool
    query: str
    results: list[dict[str, Any]]
    formatted_evidence: str
    error: str | None = None


class WebSearchTool:
    def __init__(self, max_results: int | None = None):
        settings = get_settings()
        self.max_results = max_results or settings.max_web_results

    def search(self, query: str) -> WebSearchResult:
        # Guardrail: Check if asking about internal portfolio
        internal_keywords = [
            "portfolio", "analytics.applicants", "applicants in our data",
            "internal dataset", "home credit default risk dataset", "our default rate"
        ]
        if any(kw in query.lower() for kw in internal_keywords):
            return WebSearchResult(
                success=False,
                query=query,
                results=[],
                formatted_evidence="",
                error="Web search is restricted to general credit risk industry concepts and cannot query internal portfolio data.",
            )

        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=self.max_results))

            results = []
            evidence_parts = []
            for idx, item in enumerate(raw_results, 1):
                title = item.get("title", "").strip()
                snippet = item.get("body", "").strip()
                url = item.get("href", "").strip()
                results.append({"title": title, "snippet": snippet, "url": url})
                evidence_parts.append(f"[{idx}] {title}\nURL: {url}\n{snippet}")

            formatted_evidence = "\n\n".join(evidence_parts) if evidence_parts else "No web results found."
            return WebSearchResult(
                success=True,
                query=query,
                results=results,
                formatted_evidence=formatted_evidence,
            )
        except Exception as e:
            LOGGER.warning("DDGS web search encountered an error: %s", e)
            fallback = next(
                (text for term, text in GENERAL_FALLBACKS.items() if term in query.lower()),
                None,
            )
            return WebSearchResult(
                success=False,
                query=query,
                results=[],
                formatted_evidence=(
                    f"External web search is temporarily unavailable. Cached general definition: {fallback}"
                    if fallback else "External web search is temporarily unavailable; no cached definition matched this question."
                ),
                error=str(e),
            )
