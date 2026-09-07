from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

SVG_ICONS = {
    "bank": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21h18M4 18h16M4 10l8-6 8 6M6 10v8M10 10v8M14 10v8M18 10v8"/></svg>',
    "building-columns": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21h18M4 18h16M4 10l8-6 8 6M6 10v8M10 10v8M14 10v8M18 10v8"/></svg>',
    "shield": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "shield-halved": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M12 2v20" fill="{c}"/></svg>',
    "shield-check": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/></svg>',
    "chart-line": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M19 9l-5 5-4-4-5 5"/></svg>',
    "chart-column": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>',
    "layer-group": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>',
    "compass": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" fill="{c}"/></svg>',
    "database": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "robot": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="8.5" cy="16" r="1.5" fill="{c}"/><circle cx="15.5" cy="16" r="1.5" fill="{c}"/><path d="M9 2v3M15 2v3M12 5h.01M2 15h1M21 15h1"/></svg>',
    "server": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>',
    "users": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "percent": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/></svg>',
    "trend-up": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "trend-down": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>',
    "table-cells": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>',
    "bookmark": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>',
    "scale": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v18M6 8l6-5 6 5M6 8l-4 7h8l-4-7zM18 8l-4 7h8l-4-7z"/></svg>',
    "user-group": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "star": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
    "house": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    "clock": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "bullseye": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2" fill="{c}"/></svg>',
    "sliders": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
    "gauge": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 12l3-3"/></svg>',
    "receipt": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 2 4 22 7 20 10 22 13 20 16 22 19 20 19 2"/><line x1="8" y1="6" x2="15" y2="6"/><line x1="8" y1="10" x2="15" y2="10"/><line x1="8" y1="14" x2="12" y2="14"/></svg>',
    "network-wired": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="4" rx="1"/><rect x="2" y="16" width="6" height="4" rx="1"/><rect x="16" y="16" width="6" height="4" rx="1"/><path d="M12 6v4M5 16v-3a3 3 0 0 1 3-3h8a3 3 0 0 1 3 3v3"/></svg>',
    "list-check": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="10" y1="6" x2="21" y2="6"/><line x1="10" y1="12" x2="21" y2="12"/><line x1="10" y1="18" x2="21" y2="18"/><polyline points="3 6 4.5 7.5 7 4.5"/><polyline points="3 12 4.5 13.5 7 10.5"/><polyline points="3 18 4.5 19.5 7 16.5"/></svg>',
    "sitemap": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="5" rx="1"/><rect x="2" y="16" width="6" height="5" rx="1"/><rect x="16" y="16" width="6" height="5" rx="1"/><path d="M12 8v4M5 16v-2a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v2"/></svg>',
    "handshake": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 15h2a2 2 0 1 0 0-4h-3c-.6 0-1.1.2-1.4.6L4 16l4 4 4.5-4.5"/><path d="M14 9l1.4-1.4c.4-.4.9-.6 1.4-.6h3a2 2 0 1 1 0 4h-2"/></svg>',
    "math": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="12" x2="20" y2="12"/><line x1="12" y1="4" x2="12" y2="20"/></svg>',
    "warning": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "lightbulb": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z"/></svg>',
    "code": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
    "check": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    "xmark": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
    "circle": '<svg viewBox="0 0 24 24" width="{s}" height="{s}"><circle cx="12" cy="12" r="7" fill="{c}"/></svg>',
    "question": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3M12 17h.01"/></svg>',
    "book": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20M4 4.5A2.5 2.5 0 0 1 6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15z"/></svg>',
    "globe": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    "user-tag": '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>',
}


def get_svg_icon(name: str, color: str = "#38bdf8", size: int = 16, extra_style: str = "") -> str:
    """Returns an inline SVG icon string for guaranteed rendering."""
    clean_name = name.replace("fa-solid fa-", "").replace("fa-", "").strip()
    # Map common aliases
    alias_map = {
        "shield-halved": "shield-halved",
        "shield-check": "shield-check",
        "arrow-trend-up": "trend-up",
        "arrow-trend-down": "trend-down",
        "table-cells-large": "table-cells",
        "scale-unbalanced": "scale",
        "scale-balanced": "scale",
        "clock-rotate-left": "clock",
        "gauge-high": "gauge",
        "magnifying-glass-chart": "chart-line",
        "square-root-variable": "math",
        "code-branch": "code",
        "triangle-exclamation": "warning",
        "circle-exclamation": "warning",
        "circle-question": "question",
        "book-bookmark": "book",
    }
    icon_key = alias_map.get(clean_name, clean_name)
    template = SVG_ICONS.get(icon_key, SVG_ICONS.get("chart-line"))
    svg_str = template.format(c=color, s=size)
    style_attr = f' style="display:inline-flex; align-items:center; vertical-align:middle; {extra_style}"' if extra_style else ' style="display:inline-flex; align-items:center; vertical-align:middle;"'
    return f'<span class="inline-svg-icon"{style_attr}>{svg_str}</span>'


NAVY_THEME_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap" rel="stylesheet">

<style>
/* -------------------------------------------------------------
   GLOBAL BACKGROUND, TYPOGRAPHY & AMBIENT NEBULA GLOW
------------------------------------------------------------- */
html, body, .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #f1f5f9 !important;
}

.stApp {
    background-color: #070913 !important;
    background-image: 
        radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 25%, rgba(139, 92, 246, 0.12) 0%, transparent 45%),
        radial-gradient(circle at 50% 85%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
        linear-gradient(135deg, #070913 0%, #0d1127 50%, #151833 100%) !important;
    background-attachment: fixed !important;
    background-size: cover !important;
}

/* Make Streamlit header transparent so sidebar collapse/expand works cleanly */
header[data-testid="stHeader"] {
    background: transparent !important;
    color: #f1f5f9 !important;
}

#MainMenu, footer {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 96% !important;
}

/* -------------------------------------------------------------
   GLASSMORPHIC SIDEBAR
------------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background: rgba(10, 14, 28, 0.75) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 6px 0 30px rgba(0, 0, 0, 0.45) !important;
}

section[data-testid="stSidebar"] > div {
    background: transparent !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
}

/* -------------------------------------------------------------
   PILL-SHAPED FLOATING TABS (GLASSMORPHISM & BREATHING ROOM)
------------------------------------------------------------- */
div[data-testid="stTabs"] {
    margin-top: 10px !important;
    margin-bottom: 34px !important;
}

div[data-testid="stTabs"] [role="tablist"],
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    padding: 8px 12px !important;
    border-radius: 9999px !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    gap: 12px !important;
    display: inline-flex !important;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.45), inset 0 1px 1px rgba(255, 255, 255, 0.15) !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-border"],
div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

div[data-testid="stTabs"] button[role="tab"],
.stTabs [data-baseweb="tab"] {
    height: 44px !important;
    background: transparent !important;
    border: none !important;
    border-radius: 9999px !important;
    color: #94a3b8 !important;
    font-size: 0.94rem !important;
    font-weight: 700 !important;
    padding: 0 24px !important;
    letter-spacing: 0.01em !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div[data-testid="stTabs"] button[role="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover {
    color: #f8fafc !important;
    background: rgba(255, 255, 255, 0.08) !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.28) 0%, rgba(139, 92, 246, 0.28) 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.65) !important;
    color: #38bdf8 !important;
    box-shadow: 0 0 22px rgba(56, 189, 248, 0.35), inset 0 0 12px rgba(56, 189, 248, 0.15) !important;
    transform: translateY(-1px) !important;
}

/* -------------------------------------------------------------
   EXECUTIVE BRIEFING CALLOUT BANNERS
------------------------------------------------------------- */
.executive-briefing {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(20, 29, 58, 0.7) 100%) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-left: 5px solid #38bdf8 !important;
    border-radius: 14px !important;
    padding: 18px 24px !important;
    margin-bottom: 28px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35), 0 0 16px rgba(56, 189, 248, 0.08) !important;
}

.executive-briefing-title {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    margin-bottom: 6px !important;
}

.executive-briefing-desc {
    font-size: 0.88rem !important;
    color: #cbd5e1 !important;
    line-height: 1.65 !important;
}

.executive-briefing-pills {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 12px !important;
}

.briefing-pill {
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    padding: 4px 11px !important;
    border-radius: 6px !important;
    background: rgba(56, 189, 248, 0.1) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    color: #38bdf8 !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
}

/* -------------------------------------------------------------
   VISUAL RULE FLOW CARDS (SURROGATE HEURISTICS)
------------------------------------------------------------- */
.rule-card {
    background: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(18px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    margin-bottom: 16px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
}

.rule-card:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.4) !important;
}

.rule-step {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: rgba(30, 41, 75, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    padding: 5px 12px !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    color: #f1f5f9 !important;
    margin-right: 6px !important;
    margin-bottom: 6px !important;
}

.rule-step code {
    color: #38bdf8 !important;
    font-weight: 700 !important;
    background: transparent !important;
    padding: 0 !important;
}

/* -------------------------------------------------------------
   ANTI-GRAVITY METRIC CARDS & CONTAINERS
------------------------------------------------------------- */
.metric-card {
    background: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.1) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.5), transparent);
}

.metric-card:hover {
    transform: translateY(-3px) scale(1.005) !important;
    border-color: rgba(56, 189, 248, 0.45) !important;
    box-shadow: 0 14px 35px rgba(0, 0, 0, 0.45), 0 0 20px rgba(56, 189, 248, 0.2) !important;
}

.metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.metric-icon {
    display: flex;
    align-items: center;
    justify-content: center;
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.4);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.2;
}

.metric-delta {
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.delta-positive { color: #34d399; }
.delta-neutral { color: #38bdf8; }
.delta-warning { color: #fbbf24; }
.delta-danger { color: #f43f5e; }

/* Streamlit Container border styling matching cards */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(18px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
}

/* -------------------------------------------------------------
   GLOWING PILL BUTTONS & INPUTS
------------------------------------------------------------- */
div.stButton > button {
    background: linear-gradient(135deg, rgba(30, 41, 75, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%) !important;
    backdrop-filter: blur(12px) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 9999px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7 0%, #3b82f6 100%) !important;
    color: #ffffff !important;
    border-color: rgba(56, 189, 248, 0.8) !important;
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.45) !important;
}

/* Translucent Inputs and Selectboxes */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background: rgba(10, 15, 30, 0.85) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    transition: all 0.25s ease !important;
}

div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"] > div:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.35) !important;
}

/* Dropdown popover list styling (dark theme, legible text) */
div[data-baseweb="popover"], div[data-baseweb="popover"] > div, ul[role="listbox"] {
    background-color: #0b0f1e !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.6) !important;
}

li[role="option"] {
    color: #f1f5f9 !important;
    background-color: transparent !important;
    transition: all 0.2s ease !important;
    font-size: 0.85rem !important;
}

li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background-color: rgba(56, 189, 248, 0.16) !important;
    color: #38bdf8 !important;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    font-size: 0.76rem;
    font-weight: 700;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    backdrop-filter: blur(10px);
}
.badge-low { 
    background: rgba(16, 185, 129, 0.16); 
    color: #34d399; 
    border: 1px solid rgba(16, 185, 129, 0.4); 
    box-shadow: 0 0 14px rgba(16, 185, 129, 0.2);
}
.badge-medium { 
    background: rgba(245, 158, 11, 0.16); 
    color: #fbbf24; 
    border: 1px solid rgba(245, 158, 11, 0.4); 
    box-shadow: 0 0 14px rgba(245, 158, 11, 0.2);
}
.badge-high { 
    background: rgba(244, 63, 94, 0.16); 
    color: #fb7185; 
    border: 1px solid rgba(244, 63, 94, 0.4); 
    box-shadow: 0 0 14px rgba(244, 63, 94, 0.25);
}

/* Insight Cards */
.insight-card {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-left: 4px solid #38bdf8 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    font-size: 0.88rem !important;
    line-height: 1.55 !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
    transition: all 0.25s ease !important;
}

.insight-card:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.3) !important;
}

/* Chat tool badges */
.chat-tool-badge {
    font-size: 0.76rem;
    padding: 4px 12px;
    border-radius: 9999px;
    background: rgba(30, 41, 75, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #38bdf8;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
    font-weight: 600;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
}

/* Chat messages & Expanders with glassmorphism */
div[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
    margin-bottom: 12px !important;
}

div[data-testid="stExpander"] {
    background: rgba(12, 17, 34, 0.6) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    margin-top: 10px !important;
}

/* Code block containment */
div[data-testid="stCodeBlock"] {
    max-height: 480px !important;
    overflow-y: auto !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Plotly chart container styling */
.stPlotlyChart {
    border-radius: 12px !important;
    overflow: hidden !important;
}
</style>
"""


def apply_theme():
    if hasattr(st, "html"):
        st.html(NAVY_THEME_CSS)
    else:
        st.markdown(NAVY_THEME_CSS, unsafe_allow_html=True)


def create_metric_card(
    label: str,
    value: str,
    delta: str = "",
    delta_type: str = "neutral",
    icon: str = "",
    custom_class: str = "",
):
    delta_class = f"delta-{delta_type}"
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    icon_html = ""
    if icon:
        color = "#34d399" if delta_type == "positive" else ("#f43f5e" if delta_type == "danger" else ("#fbbf24" if delta_type == "warning" else "#38bdf8"))
        icon_html = f'<div class="metric-icon">{get_svg_icon(icon, color=color, size=18)}</div>'
    card_html = f"""
    <div class="metric-card {custom_class}">
        <div class="metric-header">
            <span class="metric-label">{label}</span>
            {icon_html}
        </div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    if hasattr(st, "html"):
        st.html(card_html)
    else:
        st.markdown(card_html, unsafe_allow_html=True)


def configure_plotly_chart(
    fig: go.Figure,
    title: str = "",
    height: int = 380,
    margin_l: int = 55,
    margin_r: int = 25,
    margin_t: int | None = None,
    margin_b: int = 45,
) -> go.Figure:
    """
    Applies the Anti-Gravity Deep Space Glassmorphic Chart Styling:
    - Translucent container background matching glass cards
    - Glowing neon accents (#38bdf8, #818cf8, #f43f5e, #34d399)
    - Clean margin configuration to prevent label truncation
    - Zero vertical gridlines; ultra-faint horizontal gridlines
    - Refined tooltip styling with glow
    """
    top_margin = margin_t if margin_t is not None else (35 if title else 15)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#94a3b8"),
        margin=dict(l=margin_l, r=margin_r, t=top_margin, b=margin_b),
        height=height,
        hoverlabel=dict(
            bgcolor="rgba(10, 14, 28, 0.95)",
            bordercolor="rgba(56, 189, 248, 0.5)",
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            font_color="#f8fafc",
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="rgba(255, 255, 255, 0.08)",
            tickcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(size=10, color="#94a3b8"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
            gridwidth=1,
            zeroline=False,
            showline=False,
            tickfont=dict(size=10, color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#94a3b8"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=13, color="#f8fafc", family="Plus Jakarta Sans, sans-serif"),
                x=0.02,
                y=0.96,
            )
        )
    else:
        fig.layout.pop("title", None)
    return fig


FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "EXT_SOURCE_3": "Bureau Score C (Top Factor)",
    "EXT_SOURCE_2": "Bureau Score B",
    "EXT_SOURCE_1": "Bureau Score A",
    "AMT_CREDIT": "Loan Credit Amount",
    "AMT_INCOME_TOTAL": "Annual Total Income",
    "AMT_ANNUITY": "Loan Annual Annuity",
    "AMT_GOODS_PRICE": "Goods Purchase Price",
    "AGE_YEARS": "Applicant Age (Years)",
    "DAYS_BIRTH": "Applicant Age (Days)",
    "DAYS_EMPLOYED_CLEAN": "Days in Current Job",
    "DAYS_EMPLOYED": "Employment Duration",
    "CREDIT_INCOME_RATIO": "Credit-to-Income Multiple (DTI)",
    "ANNUITY_INCOME_RATIO": "Annuity-to-Income Debt Burden",
    "CREDIT_TERM": "Implied Credit Term",
    "CREDIT_GOODS_RATIO": "Credit-to-Goods Price Ratio",
    "EMPLOYED_AGE_RATIO": "Career Stability Ratio",
    "INST_LATE_RATE": "Late Installment Rate",
    "INST_PAYMENT_RATIO": "Installment Payment Ratio",
    "PAID_AMOUNT": "Historical Total Repaid",
    "SCHEDULED_AMOUNT": "Historical Total Billed",
    "INSTALLMENT_ROWS": "Credit Installment Count",
    "PREV_REFUSAL_RATE": "Past Loan Rejection Rate",
    "ACTIVE_CREDIT_COUNT": "Active Credit Bureau Lines",
    "PRIOR_CREDIT_COUNT": "Historical Credit Inquiries",
    "DAYS_ID_PUBLISH": "ID Document Age",
    "DAYS_REGISTRATION": "Registration Address Age",
    "DAYS_LAST_PHONE_CHANGE": "Phone Change Recency",
    "REGION_POPULATION_RELATIVE": "Region Urban Density",
    "OWN_CAR_AGE": "Vehicle Age (Years)",
    "ORGANIZATION_TYPE": "Employer Industry",
}


def get_feature_label(col_name: str) -> str:
    """Returns human-readable name for a feature column, falling back to clean title."""
    if col_name in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[col_name]
    clean = col_name.replace("AMT_", "").replace("DAYS_", "").replace("_CLEAN", "").replace("_", " ").title()
    return clean


def render_executive_briefing(
    title: str,
    description: str,
    takeaways: list[str] | None = None,
    icon: str = "lightbulb",
):
    """Renders a prominent, high-fidelity Executive Briefing banner for the page."""
    icon_svg = get_svg_icon(icon, "#38bdf8", 20)
    pills_html = ""
    if takeaways:
        pill_items = "".join([f'<span class="briefing-pill">{get_svg_icon("check", "#34d399", 11)} {t}</span>' for t in takeaways])
        pills_html = f'<div class="executive-briefing-pills">{pill_items}</div>'

    html = f"""
    <div class="executive-briefing">
        <div class="executive-briefing-title">
            {icon_svg}
            <span>{title}</span>
        </div>
        <div class="executive-briefing-desc">{description}</div>
        {pills_html}
    </div>
    """
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)
