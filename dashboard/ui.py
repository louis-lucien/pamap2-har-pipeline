"""Composants d'interface communs à toutes les pages du dashboard."""
from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

ACCENT = "#1D5FA8"

_CSS = f"""
<style>
[data-testid="stMainBlockContainer"], .block-container {{ padding-top: 3rem !important; max-width: 1240px; }}
.page-header {{ border-bottom: 1px solid #E3E6EA; padding: 1.2rem 0 .9rem 0; margin-bottom: 1.4rem; }}
.page-header .eyebrow {{ color: {ACCENT}; font-size: .74rem; font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase; }}
.page-header h1 {{ font-size: 1.75rem; font-weight: 700; margin: .2rem 0 .3rem 0; padding: 0; }}
.page-header p {{ color: #5F6B7A; font-size: .98rem; margin: 0; max-width: 860px; }}
.kpi {{ border: 1px solid #E3E6EA; border-radius: 8px; padding: 14px 16px; background: #FFFFFF; height: 100%; }}
.kpi .lab {{ color: #5F6B7A; font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }}
.kpi .val {{ color: #1F2933; font-size: 1.55rem; font-weight: 700; margin-top: 4px;
  font-variant-numeric: tabular-nums; line-height: 1.15; }}
.kpi .sub {{ color: #7B8794; font-size: .8rem; margin-top: 2px; }}
.kpi.accent {{ border-left: 3px solid {ACCENT}; }}
@media (max-width: 640px) {{ .kpi {{ margin-bottom: .6rem; }} .flow .sep {{ display: none; }} }}
.section {{ font-size: 1.08rem; font-weight: 700; color: #1F2933; margin: 1.6rem 0 .15rem 0; }}
.section-caption {{ color: #5F6B7A; font-size: .9rem; margin-bottom: .7rem; }}
.flow {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: stretch; }}
.flow .node {{ flex: 1; min-width: 140px; border: 1px solid #E3E6EA; border-radius: 8px;
  padding: 12px 14px; background: #FFFFFF; }}
.flow .node .nm {{ font-weight: 700; color: #1F2933; font-size: .9rem; }}
.flow .node .big {{ font-weight: 700; color: #1F2933; font-size: 1.35rem; font-variant-numeric: tabular-nums; }}
.flow .node .ds {{ color: #7B8794; font-size: .78rem; margin-top: 2px; line-height: 1.35; }}
.flow .sep {{ display: flex; align-items: center; color: #AAB2BD; font-size: 1.2rem; }}
.note {{ border-left: 3px solid {ACCENT}; background: #F6F7F9; padding: 10px 14px;
  border-radius: 4px; color: #3E4C59; font-size: .93rem; }}
</style>
"""


def fr(text) -> str:
    """Typographie française des nombres : 0.926 -> 0,926 et 45.5% -> 45,5 %."""
    text = re.sub(r"(?<=\d)\.(?=\d)", ",", str(text))
    return re.sub(r"(?<=\d)%", " %", text)


def apply_style() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-header"><div class="eyebrow">{eyebrow}</div>'
                f'<h1>{title}</h1><p>{fr(subtitle)}</p></div>', unsafe_allow_html=True)


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def kpi_row(items: list[tuple]) -> None:
    """items : (libellé, valeur[, sous-titre[, accent]])."""
    for col, item in zip(st.columns(len(items)), items):
        label, value, sub, accent = (*item, "", False)[:4]
        cls = "kpi accent" if accent else "kpi"
        col.markdown(f'<div class="{cls}"><div class="lab">{label}</div><div class="val">{fr(value)}</div>'
                     f'<div class="sub">{fr(sub)}</div></div>', unsafe_allow_html=True)


def flow(nodes: list[tuple[str, str, str]]) -> None:
    """Schéma horizontal : (titre, valeur mise en avant ou "", description)."""
    html = '<div class="flow">'
    for i, (name, big, desc) in enumerate(nodes):
        big_html = f'<div class="big">{fr(big)}</div>' if big else ""
        html += f'<div class="node"><div class="nm">{name}</div>{big_html}<div class="ds">{fr(desc)}</div></div>'
        if i < len(nodes) - 1:
            html += '<div class="sep">→</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{fr(text)}</div>', unsafe_allow_html=True)


def figure(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure absente : `{path.name}`. Relancez le pipeline pour la générer.")


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")
