from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


OUTPUT_PATH = Path("documents/project_presentation.pdf")
BACKGROUND = "#0b0f19"
CARD = "#1e293b"
TEXT = "#f8fafc"
MUTED = "#94a3b8"


def draw_card(ax, x: float, y: float, width: float, height: float, border: str = "#334155") -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=CARD, edgecolor=border, linewidth=1.2,
        )
    )


def create_slide(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=150)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.axis("off")
    ax.text(0.06, 0.92, title, fontsize=22, fontweight="bold", color=TEXT, va="top")
    ax.text(0.06, 0.86, subtitle, fontsize=12, color=MUTED, va="top")
    ax.plot([0.06, 0.94], [0.82, 0.82], color="#334155", lw=1.5)
    ax.text(0.06, 0.04, "NeoStats Credit Risk Intelligence Platform", fontsize=9, color="#64748b")
    return fig, ax


def add_columns(ax, cards: list[tuple[str, str, str]]) -> None:
    width = 0.27 if len(cards) == 3 else 0.42
    gap = 0.03 if len(cards) == 3 else 0.04
    line_width = 34 if len(cards) == 3 else 52
    for index, (title, body, accent) in enumerate(cards):
        x = 0.06 + index * (width + gap)
        draw_card(ax, x, 0.18, width, 0.56, accent)
        ax.text(x + 0.02, 0.69, title, fontsize=13, fontweight="bold", color=accent)
        wrapped_body = "\n".join(
            textwrap.fill(line, width=line_width) if line else ""
            for line in body.splitlines()
        )
        ax.text(x + 0.02, 0.62, wrapped_body, fontsize=9.6, color="#cbd5e1", va="top", linespacing=1.45)


def generate_deck() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PATH) as pdf:
        fig, ax = create_slide(
            "NeoStats Credit Risk Intelligence Platform",
            "Calibrated EBM | PostgreSQL analytics | Gemini agentic assistant",
        )
        draw_card(ax, 0.08, 0.23, 0.84, 0.46, "#38bdf8")
        ax.text(0.11, 0.60, "Demo-ready credit-risk decision support", fontsize=28, fontweight="bold", color=TEXT)
        ax.text(
            0.11, 0.50,
            "Transparent prediction, model explanations, curated analytics,\nand durable conversational intelligence.",
            fontsize=16, color=MUTED, va="top",
        )
        ax.text(0.11, 0.30, "Internal risk segmentation - not an automated lending policy or regulatory rating.", fontsize=11, color="#fbbf24")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_slide("1. Confirmed EDA", "Associations measured across 307,511 applicants")
        add_columns(ax, [
            ("Portfolio", "8.07% default rate\n11.4:1 class imbalance\nEvaluate with ROC-AUC, PR-AUC, and calibration", "#38bdf8"),
            ("Segments", "20-29 has the highest observed age-band rate\nRented housing shows 1.53x observed lift\nAssociations are not causal policy rules", "#f59e0b"),
            ("Behaviour", "External scores dominate importance\nOverdues, refusals, and late instalments provide strong historical signals", "#10b981"),
        ])
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_slide("2. Model hierarchy", "Untouched stratified test-set results")
        add_columns(ax, [
            ("Logistic baseline", "ROC-AUC  0.759\nPR-AUC   0.238\nBrier     0.199\nInterpretable performance floor", MUTED),
            ("LightGBM shadow", "ROC-AUC  0.778\nPR-AUC   0.265\nBrier     0.174\nFeature pruning and reference ceiling", "#38bdf8"),
            ("Calibrated EBM", "ROC-AUC  0.764\nPR-AUC   0.250\nBrier     0.068\nProduction glass-box model", "#10b981"),
        ])
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_slide("3. Explainability and rules", "Transparent evidence with explicit governance boundaries")
        add_columns(ax, [
            ("EBM explanations", "Exact additive global and applicant-level score contributions support model review. Customer-facing reasons require separate validation.", "#38bdf8"),
            ("Surrogate tree", "Depth 4\nValidation R2 0.525\nRisk-band agreement 68.0%\nExplanatory approximation only - not approval policy.", "#f59e0b"),
        ])
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_slide("4. Agentic assistant", "Fast routing, bounded execution, and transparent evidence")
        add_columns(ax, [
            ("Database", "Curated PostgreSQL views\nRead-only login\n8-second timeout\n200-row maximum", "#38bdf8"),
            ("Knowledge", "206 curated chunks\nLocal 768-dimensional embeddings\npgvector cosine retrieval\nConfirmed internal evidence", "#10b981"),
            ("Web", "DDGS external search\nMaximum five results\nInternal-data guardrail\nTransparent offline fallback", "#f59e0b"),
        ])
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_slide("5. Deployment and verification", "One-command Docker Compose demo")
        add_columns(ax, [
            ("Runtime", "PostgreSQL 17 + pgvector\nOne-shot deterministic initialization\nStreamlit waits for initialization\nModels and artifacts mounted read-only", "#38bdf8"),
            ("Verification", "33 automated tests\nPersistent refresh-safe chat threads\nPositive SQL function allowlist\nHealthy Streamlit endpoint", "#10b981"),
        ])
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Presentation deck generated at {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    generate_deck()
