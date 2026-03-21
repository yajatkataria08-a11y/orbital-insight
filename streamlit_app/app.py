"""
Orbital Insight — Streamlit Analytics Dashboard v7.4
NSH 2026 · Team BroCODE
CSS / JS perfectly mirrors index.html: same fonts, variables, panel-hdr, panel-badge,
cdm-item, sat-item, contact-win, uptime bars, telem-grid, cursor ring + tooltip.
Best of v7.2 (login overlay, full pages, cursor) + v7.3 (panel_header, cdm-item, scan sweep).
"""

import streamlit as st
import requests
import pandas as pd
import altair as alt
import time
import os as _os
from datetime import datetime

st.set_page_config(
    page_title="Orbital Insight v7.4 — Analytics",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API = _os.environ.get("BACKEND_URL", "http://localhost:8000/api")

# ═══════════════════════════════════════════════════════════════════════════════
#  CSS — exact index.html variable names, font stack, and component classes
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

/* ── Root vars (identical to index.html :root) ── */
:root {
  --bg:#010508; --bg2:#030b10; --bg3:#061420;
  --panel:#05111a; --panel2:#071822;
  --border:rgba(0,210,180,0.10); --border2:rgba(0,210,180,0.22); --border3:rgba(0,210,180,0.40);
  --cyan:#00c8b4; --cyan2:#00e8d0; --cyan3:#00fff0;
  --blue:#0090ff; --blue2:#00b4ff;
  --yellow:#ffd700; --orange:#ff7b00; --red:#ff2244; --red2:#ff4466;
  --green:#00ff88; --green2:#00cc66; --purple:#aa44ff;
  --muted:rgba(0,220,190,0.35); --muted2:rgba(0,220,190,0.18);
  --text:rgba(220,255,248,0.88); --text2:rgba(180,230,220,0.50); --text3:rgba(140,200,185,0.30);
  --font-mono:'Share Tech Mono',monospace;
  --font-body:'Rajdhani',sans-serif;
  --font-display:'Orbitron',sans-serif;
  --glow-cyan:0 0 12px rgba(0,210,180,0.5);
  --glow-red:0 0 12px rgba(255,34,68,0.5);
  --glow-green:0 0 12px rgba(0,255,136,0.4);
}

/* ── Global overrides ── */
html,body,[class*="css"],.stApp {
  font-family:var(--font-body) !important;
  background:var(--bg) !important;
  color:var(--text) !important;
}
.stApp { background:var(--bg) !important; }
.main .block-container { padding-top:0.75rem !important; max-width:100% !important; position:relative; z-index:1; }

/* scanlines */
.stApp::before {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.025) 3px,rgba(0,0,0,0.025) 4px);
}
/* top vignette */
.stApp::after {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background:radial-gradient(ellipse at 50% 0%,rgba(0,180,160,0.04) 0%,transparent 70%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background:var(--bg2) !important;
  border-right:1px solid var(--border2) !important;
}
[data-testid="stSidebar"]::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,transparent,var(--cyan2),transparent);
}
[data-testid="stSidebar"] > div { padding-top:0.75rem; }
[data-testid="stSidebar"] .stRadio > div { gap:1px !important; }
[data-testid="stSidebar"] .stRadio label {
  font-family:var(--font-mono) !important;
  background:transparent !important; border:1px solid transparent !important;
  border-radius:3px !important; padding:7px 10px !important; margin:1px 0 !important;
  cursor:pointer !important; transition:all 0.12s !important;
  color:var(--text2) !important; font-size:10px !important; letter-spacing:0.08em !important;
  display:block !important; width:100% !important; position:relative !important; overflow:hidden !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background:rgba(0,200,180,0.04) !important; border-color:var(--border) !important;
  color:var(--cyan2) !important; padding-left:14px !important; text-shadow:var(--glow-cyan) !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child { display:none !important; }

/* ── Buttons ── */
.stButton > button {
  background:rgba(0,200,180,0.06) !important; border:1px solid var(--border2) !important;
  color:var(--cyan2) !important; font-family:var(--font-mono) !important;
  font-size:10px !important; letter-spacing:0.14em !important;
  border-radius:3px !important; transition:all 0.15s !important; text-transform:uppercase !important;
  position:relative !important; overflow:hidden !important;
}
.stButton > button::after {
  content:''; position:absolute; inset:0;
  background:linear-gradient(135deg,var(--cyan),var(--cyan2));
  opacity:0; transition:opacity 0.15s;
}
.stButton > button:hover {
  background:rgba(0,200,180,0.12) !important; border-color:var(--border3) !important;
  box-shadow:0 0 18px rgba(0,200,180,0.2) !important; transform:translateY(-1px) !important;
}
.stButton > button:active { transform:translateY(1px) !important; box-shadow:0 0 8px rgba(0,200,180,0.3) !important; }

/* ── Inputs & selects ── */
.stSelectbox div[data-baseweb="select"] > div {
  background:var(--bg3) !important; border-color:var(--border2) !important;
  color:var(--text) !important; font-family:var(--font-mono) !important; font-size:10px !important;
  transition:border-color 0.15s,box-shadow 0.15s !important;
}
.stSelectbox div[data-baseweb="select"]:focus-within > div {
  border-color:var(--cyan2) !important; box-shadow:0 0 10px rgba(0,200,180,0.15) !important;
}
[data-baseweb="popover"] { background:var(--bg3) !important; border:1px solid var(--border2) !important; }
[data-baseweb="menu"] li { color:var(--text2) !important; font-family:var(--font-mono) !important; font-size:10px !important; transition:all 0.1s !important; }
[data-baseweb="menu"] li:hover { background:rgba(0,200,180,0.08) !important; color:var(--cyan2) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:var(--cyan2) !important; box-shadow:0 0 8px var(--cyan2) !important; will-change:box-shadow; }
.stCheckbox label { color:var(--text2) !important; font-family:var(--font-mono) !important; font-size:10px !important; letter-spacing:0.1em !important; transition:color 0.15s !important; }
.stCheckbox label:hover { color:var(--cyan2) !important; }

/* ── DataFrames ── */
.stDataFrame,[data-testid="stDataFrame"] {
  background:var(--bg3) !important; border:1px solid var(--border) !important; border-radius:3px !important;
}
[data-testid="stDataFrame"] th {
  background:rgba(0,200,180,0.06) !important; color:var(--cyan) !important;
  font-family:var(--font-mono) !important; font-size:9px !important;
  letter-spacing:0.12em !important; text-transform:uppercase !important;
  border-bottom:1px solid var(--border2) !important;
}
[data-testid="stDataFrame"] td {
  color:var(--text) !important; font-family:var(--font-mono) !important;
  font-size:10px !important; border-bottom:1px solid var(--border) !important;
}
[data-testid="stDataFrame"] tr:hover td { background:rgba(0,200,180,0.04) !important; }

/* ── Alerts ── */
.stSuccess { background:rgba(0,255,136,0.07) !important; border:1px solid rgba(0,255,136,0.25) !important; color:var(--green) !important; font-family:var(--font-mono) !important; font-size:10px !important; border-radius:3px !important; }
.stError   { background:rgba(255,34,68,0.07) !important;  border:1px solid rgba(255,34,68,0.25) !important;  color:var(--red) !important;   font-family:var(--font-mono) !important; font-size:10px !important; border-radius:3px !important; }
.stInfo    { background:rgba(0,144,255,0.07) !important;  border:1px solid rgba(0,144,255,0.25) !important;  color:var(--blue2) !important; font-family:var(--font-mono) !important; font-size:10px !important; border-radius:3px !important; }
hr { border:none !important; border-top:1px solid var(--border) !important; margin:10px 0 !important; }

/* ─────────────────── index.html PANEL classes ─────────────────── */
.panel-hdr {
  display:flex; align-items:center; justify-content:space-between;
  padding:7px 11px; border-bottom:1px solid var(--border);
  background:rgba(0,0,0,0.15); margin-bottom:10px; margin-top:16px;
  transition:background 0.2s;
}
.panel-hdr:hover { background:rgba(0,200,180,0.04); }
.panel-title {
  font-family:var(--font-mono); font-size:9px; letter-spacing:0.12em;
  color:var(--cyan); display:flex; align-items:center; gap:6px;
}
.panel-title::before {
  content:''; width:2px; height:9px; background:var(--cyan);
  display:inline-block; box-shadow:var(--glow-cyan);
}
.panel-badge {
  font-family:var(--font-mono); font-size:8px; padding:2px 6px;
  border-radius:2px; border:1px solid; letter-spacing:0.05em;
  transition:all 0.2s; cursor:default;
}
.panel-badge:hover { letter-spacing:0.08em; }
.badge-green  { color:var(--green);  border-color:rgba(0,255,136,0.3);  background:rgba(0,255,136,0.06); }
.badge-yellow { color:var(--yellow); border-color:rgba(255,215,0,0.3);  background:rgba(255,215,0,0.06); }
.badge-red    { color:var(--red);    border-color:rgba(255,34,68,0.3);  background:rgba(255,34,68,0.06); box-shadow:var(--glow-red); }
.badge-blue   { color:var(--blue2);  border-color:rgba(0,180,255,0.3);  background:rgba(0,180,255,0.06); }
.badge-purple { color:var(--purple); border-color:rgba(170,68,255,0.3); background:rgba(170,68,255,0.06); }
.badge-cyan   { color:var(--cyan2);  border-color:var(--border2);       background:rgba(0,232,208,0.06); box-shadow:var(--glow-cyan); }
.badge-orange { color:var(--orange); border-color:rgba(255,123,0,0.3);  background:rgba(255,123,0,0.06); }

/* ── Metric card ── */
.metric-card {
  background:rgba(0,0,0,0.28); border:1px solid var(--border); border-radius:3px;
  padding:12px 14px; position:relative; overflow:hidden;
  transition:border-color 0.18s, background 0.18s, transform 0.15s; cursor:default;
}
.metric-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent); opacity:0.4;
}
.metric-card::after {
  content:''; position:absolute; top:0; left:-100%; width:100%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(0,210,180,0.06),transparent);
  transition:none; pointer-events:none;
}
.metric-card:hover {
  border-color:var(--border2); background:rgba(0,200,180,0.05);
  box-shadow:0 0 14px rgba(0,200,180,0.08) inset; transform:translateY(-1px);
}
.metric-card:hover::after { animation:scan-sweep 0.5s ease-out forwards; }
@keyframes scan-sweep { from{left:-100%} to{left:100%} }
.metric-label { font-family:var(--font-mono); font-size:8px; letter-spacing:0.15em; color:var(--text2); text-transform:uppercase; margin-bottom:5px; }
.metric-value { font-family:var(--font-display); font-size:20px; font-weight:700; color:var(--text); line-height:1; }
.metric-value.red    { color:var(--red);    text-shadow:var(--glow-red);   }
.metric-value.green  { color:var(--green);  text-shadow:var(--glow-green); }
.metric-value.cyan   { color:var(--cyan2);  text-shadow:var(--glow-cyan);  }
.metric-value.yellow { color:var(--yellow); }
.metric-value.purple { color:var(--purple); }
.metric-value.blue   { color:var(--blue2);  }
.metric-value.orange { color:var(--orange); }
.metric-delta { font-family:var(--font-mono); font-size:8px; margin-top:4px; color:var(--text2); letter-spacing:0.06em; }

/* ── CDM items (exact index.html .cdm-item) ── */
.cdm-item {
  padding:7px 9px; border-left:2px solid; margin-bottom:4px;
  cursor:pointer; background:rgba(0,0,0,0.18); transition:all 0.12s;
  border-radius:0 3px 3px 0;
}
.cdm-item:hover { background:rgba(0,200,180,0.05); transform:translateX(2px); box-shadow:0 0 10px rgba(0,200,180,0.06) inset; }
.cdm-item.safe { border-left-color:var(--green); }
.cdm-item.warn { border-left-color:var(--yellow); }
.cdm-item.crit { border-left-color:var(--red); animation:cdmpulse 1.5s infinite; }
@keyframes cdmpulse { 0%,100%{background:rgba(255,34,68,0.04)} 50%{background:rgba(255,34,68,0.10)} }
.cdm-ids  { font-family:var(--font-mono); font-size:8px; color:var(--text2); margin-bottom:2px; }
.cdm-dist { font-family:var(--font-mono); font-size:11px; font-weight:700; }
.cdm-meta { display:flex; justify-content:space-between; align-items:center; margin-top:2px; }
.cdm-pc   { font-family:var(--font-mono); font-size:8px; color:var(--text2); }
.cdm-tca  { font-family:var(--font-mono); font-size:8px; color:var(--text3); }
.cdm-pruned { font-family:var(--font-mono); font-size:7px; color:var(--purple); margin-top:1px; }

/* ── Contact window (exact index.html .contact-win) ── */
.contact-win {
  padding:8px 10px; border:1px solid var(--border); border-radius:3px;
  margin-bottom:5px; background:rgba(0,0,0,0.2); transition:all 0.15s;
}
.contact-win:hover { border-color:var(--border2); background:rgba(0,200,180,0.04); transform:translateY(-1px); box-shadow:0 0 12px rgba(0,200,180,0.08); }
.contact-win.blackout { border-color:rgba(255,123,0,0.3); }
.contact-win.blackout:hover { border-color:rgba(255,123,0,0.55); box-shadow:0 0 12px rgba(255,123,0,0.1); }
.cw-gs   { font-family:var(--font-mono); font-size:9px; color:var(--cyan); margin-bottom:3px; display:flex; align-items:center; gap:6px; }
.cw-time { font-family:var(--font-mono); font-size:8px; color:var(--text2); }
.cw-dur  { font-family:var(--font-mono); font-size:9px; color:var(--green); }
.cw-el   { font-family:var(--font-mono); font-size:8px; color:var(--text2); }
.cw-blackout-tag { font-family:var(--font-mono); font-size:7px; color:var(--orange); margin-top:3px; padding:2px 5px; border:1px solid rgba(255,123,0,0.3); border-radius:2px; display:inline-block; }

/* ── Uptime bars (exact index.html) ── */
.uptime-bar-row { display:flex; align-items:center; gap:6px; margin-bottom:4px; transition:opacity 0.15s; }
.uptime-bar-row:hover { opacity:0.85; }
.uptime-sat-id { font-family:var(--font-mono); font-size:8px; color:var(--text2); width:72px; flex-shrink:0; overflow:hidden; text-overflow:ellipsis; }
.uptime-bar-bg { flex:1; height:4px; background:rgba(255,255,255,0.06); border-radius:2px; overflow:hidden; }
.uptime-bar-fill { height:100%; border-radius:2px; transition:width 1s,box-shadow 0.2s; }
.uptime-bar-row:hover .uptime-bar-fill { box-shadow:0 0 6px currentColor; }
.uptime-pct { font-family:var(--font-mono); font-size:8px; width:36px; text-align:right; flex-shrink:0; }
.uptime-fleet-card { background:rgba(0,200,180,0.06); border:1px solid var(--border2); border-radius:4px; padding:12px; margin-bottom:10px; text-align:center; }
.uptime-fleet-num { font-family:var(--font-display); font-size:24px; font-weight:700; color:var(--cyan2); }
.uptime-grade { font-family:var(--font-mono); font-size:9px; margin-top:3px; }
.grade-EXCELLENT { color:var(--green);  } .grade-GOOD { color:var(--cyan); }
.grade-ACCEPTABLE { color:var(--yellow); } .grade-POOR { color:var(--red); }

/* ── Telem grid ── */
.telem-grid { display:grid; grid-template-columns:1fr 1fr; gap:5px; margin-top:6px; }
.telem-val  { background:rgba(0,0,0,0.28); border:1px solid var(--border); border-radius:3px; padding:5px 8px; transition:border-color 0.15s; }
.telem-val:hover { border-color:var(--border2); }
.telem-label { font-family:var(--font-mono); font-size:7px; color:var(--text2); letter-spacing:0.08em; margin-bottom:2px; text-transform:uppercase; }
.telem-num   { font-family:var(--font-mono); font-size:12px; color:var(--cyan2); }
.telem-num.warn   { color:var(--yellow); } .telem-num.crit { color:var(--red); text-shadow:var(--glow-red); }
.telem-num.green  { color:var(--green);  } .telem-num.purple { color:var(--purple); }

/* ── GS items ── */
.gs-item { padding:6px 10px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(0,200,180,0.04); transition:background 0.1s; }
.gs-item:hover { background:rgba(0,200,180,0.04); }
.gs-card { background:linear-gradient(135deg,var(--bg3),var(--bg2)); border:1px solid var(--border); border-radius:3px; padding:12px 14px; margin-bottom:8px; position:relative; overflow:hidden; transition:border-color 0.18s,box-shadow 0.18s,transform 0.15s; cursor:default; }
.gs-card:hover { border-color:var(--border2); box-shadow:0 0 14px rgba(0,200,180,0.1); transform:translateY(-1px); }
.gs-card.active::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,var(--green),transparent); }

/* ── Rubric boxes ── */
.rubric-box { text-align:center; padding:10px 8px; border-radius:3px; border:1px solid transparent; transition:border-color 0.2s,box-shadow 0.2s,transform 0.15s; cursor:default; }
.rubric-box:hover { transform:translateY(-2px); }
.rubric-excellent { background:rgba(0,255,136,0.05); border-color:rgba(0,255,136,0.12); }
.rubric-excellent:hover { border-color:rgba(0,255,136,0.4); box-shadow:0 4px 16px rgba(0,255,136,0.1); }
.rubric-good  { background:rgba(0,200,180,0.05); border-color:rgba(0,200,180,0.12); }
.rubric-good:hover  { border-color:rgba(0,200,180,0.4); box-shadow:0 4px 16px rgba(0,200,180,0.1); }
.rubric-ok    { background:rgba(255,215,0,0.05);  border-color:rgba(255,215,0,0.12); }
.rubric-ok:hover    { border-color:rgba(255,215,0,0.4);  box-shadow:0 4px 16px rgba(255,215,0,0.1); }
.rubric-poor  { background:rgba(255,34,68,0.05);  border-color:rgba(255,34,68,0.12); }
.rubric-poor:hover  { border-color:rgba(255,34,68,0.4);  box-shadow:0 4px 16px rgba(255,34,68,0.1); }

/* ── Sidebar logo ── */
.tb-logo { font-family:var(--font-display); font-size:20px; font-weight:900; color:var(--cyan2); letter-spacing:0.15em; text-shadow:var(--glow-cyan); margin-bottom:2px; }
.tb-sub  { font-family:var(--font-mono); font-size:9px; color:var(--text2); letter-spacing:0.2em; margin-bottom:14px; }
.sidebar-uptime { background:rgba(0,200,180,0.06); border:1px solid var(--border2); border-radius:3px; padding:10px 12px; margin-bottom:8px; transition:border-color 0.2s; }
.sidebar-uptime:hover { border-color:var(--border3); }

/* ── Animations ── */
@keyframes blink   { 0%,100%{opacity:1} 50%{opacity:0.3} }
@keyframes pulse   { 0%,100%{opacity:1} 50%{opacity:0.35} }
.live-dot { animation:pulse 1.8s ease-in-out infinite; }
.crit-blink { animation:blink 0.8s infinite; }

/* ── Smooth hover transitions — global ── */
*, *::before, *::after {
  -webkit-font-smoothing: antialiased;
}
/* Sidebar nav items */
[data-testid="stSidebar"] .stRadio label {
  transition: all 0.18s cubic-bezier(0.4,0,0.2,1) !important;
}
/* All buttons */
.stButton > button {
  transition: all 0.18s cubic-bezier(0.4,0,0.2,1) !important;
}
/* Metric cards */
.metric-card {
  transition: border-color 0.22s ease, background 0.22s ease,
              transform 0.18s cubic-bezier(0.34,1.56,0.64,1),
              box-shadow 0.22s ease !important;
}
/* CDM items */
.cdm-item {
  transition: background 0.15s ease, transform 0.15s cubic-bezier(0.34,1.56,0.64,1),
              box-shadow 0.15s ease !important;
}
/* Contact windows */
.contact-win {
  transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
}
/* GS cards */
.gs-card {
  transition: border-color 0.2s ease, box-shadow 0.2s ease,
              transform 0.18s cubic-bezier(0.34,1.56,0.64,1) !important;
}
/* Rubric boxes */
.rubric-box {
  transition: all 0.2s cubic-bezier(0.34,1.56,0.64,1) !important;
}
/* Panel badges */
.panel-badge {
  transition: box-shadow 0.15s ease, letter-spacing 0.15s ease !important;
}
.panel-badge:hover {
  box-shadow: 0 0 8px currentColor !important;
  letter-spacing: 0.08em !important;
}
/* Uptime bars — smooth fill on load */
.uptime-bar-fill {
  transition: width 1.2s cubic-bezier(0.4,0,0.2,1),
              box-shadow 0.2s ease !important;
}
/* Dataframe rows */
[data-testid="stDataFrame"] tr {
  transition: background 0.12s ease !important;
}
/* Slider thumb — glow only, no transform (transform causes Streamlit rerender) */
.stSlider [data-baseweb="slider"] div[role="slider"] {
  transition: box-shadow 0.15s ease !important;
  will-change: box-shadow;
}
.stSlider [data-baseweb="slider"] div[role="slider"]:hover {
  box-shadow: 0 0 16px var(--cyan2), 0 0 4px var(--cyan2) !important;
}
/* Telem vals */
.telem-val {
  transition: border-color 0.18s ease, background 0.18s ease !important;
}
/* Progress bars in dataframes */
[data-testid="stDataFrame"] [data-progress] {
  transition: width 0.8s ease !important;
}
/* Sidebar uptime card */
.sidebar-uptime {
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.sidebar-uptime:hover {
  box-shadow: 0 0 12px rgba(0,200,180,0.15) !important;
}
/* ── stMetric dark theme — used in ML Intelligence tabs ── */
div[data-testid="stMetric"] {
  background:var(--bg3) !important;
  border:1px solid var(--border) !important;
  border-radius:3px !important;
  padding:10px 12px !important;
  transition:border-color 0.2s !important;
}
div[data-testid="stMetric"]:hover {
  border-color:var(--border2) !important;
}
div[data-testid="stMetricLabel"] p {
  font-family:var(--font-mono) !important;
  font-size:8px !important;
  letter-spacing:0.12em !important;
  color:var(--text2) !important;
  text-transform:uppercase !important;
}
div[data-testid="stMetricValue"] {
  font-family:var(--font-display) !important;
  color:var(--cyan2) !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOGIN — Animated card via components.html + postMessage to session_state
# ═══════════════════════════════════════════════════════════════════════════════
import streamlit.components.v1 as _components

VALID_CREDS = {"admin": "orbital2026", "nsh2026": "acm", "brocode": "orbital"}

if "oi_authenticated" not in st.session_state:
    st.session_state.oi_authenticated = False

# Receive credentials from the login iframe via query params
_auth_query = st.query_params.get("oi_auth_user", "")
_auth_pass  = st.query_params.get("oi_auth_pass", "")
if _auth_query and VALID_CREDS.get(_auth_query) == _auth_pass:
    st.session_state.oi_authenticated = True
    st.query_params.clear()
    st.rerun()

if not st.session_state.oi_authenticated:
    st.markdown("""<style>
#MainMenu,header,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"]{display:none!important}
.stApp{background:#000a0f!important}
section[data-testid="stSidebar"]{display:none!important}
.block-container{padding:0!important;margin:0!important;max-width:100%!important}
iframe{position:fixed!important;inset:0!important;
  width:100vw!important;height:100vh!important;
  border:none!important;z-index:9!important}
</style>""", unsafe_allow_html=True)
    _login_html = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#010508;--bg2:#030b10;--bg3:#061420;
      --cyan:#00c8b4;--cyan2:#00e8d0;--cyan3:#00fff0;
      --green:#00ff88;--red:#ff2244;
      --text:rgba(220,255,248,0.88);--text2:rgba(180,230,220,0.50);--text3:rgba(140,200,185,0.30);
      --border:rgba(0,210,180,0.10);--border2:rgba(0,210,180,0.22);
      --font-mono:'Share Tech Mono',monospace;
      --font-display:'Orbitron',sans-serif;
      --glow-cyan:0 0 12px rgba(0,210,180,0.5);
      --glow-green:0 0 12px rgba(0,255,136,0.4);
    }
    html,body{
  width:100%;height:100%;min-height:100vh;
  overflow:hidden;background:var(--bg);
  display:flex;align-items:center;justify-content:center;
}
       LOGIN PAGE
    ══════════════════════════════════════════════ */
    #login-screen{
      position:fixed;inset:0;z-index:99999;
      background:#000a0f;
      display:flex;align-items:center;justify-content:center;
      transition:opacity 0.8s ease, transform 0.8s ease;
      overflow:hidden;
    }
    #login-screen.fade-out{opacity:0;transform:scale(1.04);pointer-events:none}

    /* animated star field behind login */
    #login-canvas{position:absolute;inset:0;pointer-events:none}

    /* orbiting ring decoration */
    .login-ring{
      position:absolute;border-radius:50%;border:1px solid;
      animation:spin linear infinite;
      pointer-events:none;
    }
    .login-ring-1{
      width:440px;height:440px;
      border-color:rgba(0,210,180,0.12);
      top:50%;left:50%;margin:-220px 0 0 -220px;
      animation-duration:22s;
    }
    .login-ring-2{
      width:320px;height:320px;
      border-color:rgba(0,210,180,0.08);
      top:50%;left:50%;margin:-160px 0 0 -160px;
      animation-duration:14s;animation-direction:reverse;
    }
    .login-ring-3{
      width:200px;height:200px;
      border-color:rgba(0,200,180,0.06);
      top:50%;left:50%;margin:-100px 0 0 -100px;
      animation-duration:8s;
    }
    /* small dot on each ring */
    .login-ring::after{
      content:'';position:absolute;top:-3px;left:50%;
      width:5px;height:5px;margin-left:-2px;
      border-radius:50%;background:var(--cyan2);
      box-shadow:var(--glow-cyan);
    }
    @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}

    /* horizontal scan line */
    .login-scan{
      position:absolute;left:0;right:0;height:1px;
      background:linear-gradient(90deg,transparent 0%,rgba(0,210,180,0.4) 30%,rgba(0,210,180,0.8) 50%,rgba(0,210,180,0.4) 70%,transparent 100%);
      animation:scandown 4s linear infinite;pointer-events:none;
      box-shadow:0 0 8px rgba(0,210,180,0.4);
    }
    @keyframes scandown{0%{top:-2px;opacity:0}5%{opacity:1}95%{opacity:1}100%{top:100%;opacity:0}}

    /* ══════════════════════════════════════════════
       LOGIN INTRO SEQUENCE
       1. Title appears center-screen (0s)
       2. Title shifts upward to card-header position (1.4s)
       3. Card body grows downward from beneath title (1.8s)
       4. Fields fade in (2.5s) · Title inside card swaps in (2.4s)
    ══════════════════════════════════════════════ */

    /* Floating title — starts dead center, shifts up to card header position.
       --splash-shift is set at runtime by JS to account for any screen height. */
    #login-splash{
      position:absolute;z-index:4;
      top:50%;left:50%;
      transform:translate(-50%,-50%);
      text-align:center;pointer-events:none;
      --splash-shift:-155px; /* fallback; overridden by JS */
      animation:splashMove 3.0s cubic-bezier(0.4,0,0.2,1) forwards;
    }
    #login-splash-text{
      font-family:var(--font-display);font-size:26px;font-weight:900;
      color:var(--cyan2);letter-spacing:0.16em;white-space:nowrap;
      /* glow stops running once parent opacity hits 0 at 84% of 3s = 2.52s */
      animation:splashGlow 1.2s ease-in-out infinite alternate;
    }
    #login-splash-sub{
      font-family:var(--font-mono);font-size:9px;color:var(--text2);
      letter-spacing:0.22em;margin-top:7px;
    }
    /* Phase 1: fade in · Phase 2: hold · Phase 3: shift up · Phase 4: fade out */
    @keyframes splashMove{
      0%  {opacity:0; transform:translate(-50%,-50%) translateY(12px)}
      14% {opacity:1; transform:translate(-50%,-50%) translateY(0)}
      44% {opacity:1; transform:translate(-50%,-50%) translateY(0)}
      72% {opacity:1; transform:translate(-50%,-50%) translateY(var(--splash-shift))}
      84% {opacity:0; transform:translate(-50%,-50%) translateY(var(--splash-shift))}
      100%{opacity:0; transform:translate(-50%,-50%) translateY(var(--splash-shift))}
    }
    @keyframes splashGlow{
      from{text-shadow:0 0 16px rgba(0,232,208,0.55),0 0 36px rgba(0,232,208,0.18)}
      to  {text-shadow:0 0 28px rgba(0,232,208,1.0), 0 0 70px rgba(0,232,208,0.38),0 0 110px rgba(0,232,208,0.12)}
    }

    /* Card: invisible at first, grows DOWNWARD once title is in position */
    .login-card{
      position:relative;z-index:2;
      width:400px;padding:44px 40px 40px;
      background:rgba(3,12,18,0.95);
      border:1px solid rgba(0,210,180,0.2);
      border-radius:4px;
      box-shadow:0 0 60px rgba(0,0,0,0.8),0 0 30px rgba(0,210,180,0.05) inset;
      overflow:hidden;
      animation:cardGrow 0.75s cubic-bezier(0.16,1,0.3,1) both;
      animation-delay:1.85s;
    }
    /* Clip top stays fixed, bottom expands → card grows downward */
    @keyframes cardGrow{
      from{clip-path:inset(0 0 100% 0 round 4px);opacity:0.8}
      to  {clip-path:inset(0 0 0%   0 round 4px);opacity:1}
    }

    /* Logo/sub inside card: hidden during splash, fades in as splash disappears */
    .login-logo,.login-sub,.login-divider{
      animation:logoReveal 0.35s ease both;
      animation-delay:2.45s;
    }
    @keyframes logoReveal{
      from{opacity:0} to{opacity:1}
    }

    /* Fields + button: fade up after card is open */
    .login-card-body{
      animation:bodyReveal 0.45s ease both;
      animation-delay:2.55s;
    }
    @keyframes bodyReveal{
      from{opacity:0;transform:translateY(10px)}
      to  {opacity:1;transform:translateY(0)}
    }

    /* top glow bar */
    .login-card::before{
      content:'';position:absolute;top:-1px;left:20%;right:20%;height:1px;
      background:linear-gradient(90deg,transparent,var(--cyan2),transparent);
      box-shadow:0 0 12px var(--cyan2);
    }

    .login-logo{
      font-family:var(--font-display);font-size:20px;font-weight:900;
      color:var(--cyan2);letter-spacing:0.14em;text-align:center;
      text-shadow:var(--glow-cyan);margin-bottom:4px;
    }
    .login-sub{
      font-family:var(--font-mono);font-size:9px;color:var(--text2);
      letter-spacing:0.2em;text-align:center;margin-bottom:30px;
    }
    .login-divider{
      width:60px;height:1px;background:var(--border2);
      margin:0 auto 28px;position:relative;
    }
    .login-divider::before,.login-divider::after{
      content:'';position:absolute;top:-2px;
      width:4px;height:4px;border-radius:50%;
      background:var(--cyan2);box-shadow:var(--glow-cyan);
    }
    .login-divider::before{left:-2px}.login-divider::after{right:-2px}

    .login-field-wrap{margin-bottom:16px;position:relative}
    .login-field-label{
      font-family:var(--font-mono);font-size:8px;letter-spacing:0.16em;
      color:var(--text2);margin-bottom:6px;display:block;
      transition:color 0.2s;
    }
    .login-field-wrap:focus-within .login-field-label{color:var(--cyan2)}
    .login-field{
      width:100%;background:rgba(0,20,28,0.8);
      border:1px solid rgba(0,210,180,0.18);
      border-radius:3px;padding:10px 14px;
      font-family:var(--font-mono);font-size:11px;color:var(--text);
      outline:none;transition:border-color 0.2s, box-shadow 0.2s, background 0.2s;
      letter-spacing:0.06em;
    }
    .login-field::placeholder{color:var(--text3)}
    .login-field:focus{
      border-color:var(--cyan2);
      box-shadow:0 0 0 2px rgba(0,210,180,0.12), 0 0 12px rgba(0,210,180,0.08) inset;
      background:rgba(0,25,35,0.9);
    }
    /* field bottom sweep animation on focus */
    .login-field-sweep{
      position:absolute;bottom:0;left:0;right:0;height:2px;overflow:hidden;border-radius:0 0 3px 3px;
    }
    .login-field-sweep::after{
      content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;
      background:linear-gradient(90deg,transparent,var(--cyan2),transparent);
      transition:none;
    }
    .login-field-wrap:focus-within .login-field-sweep::after{
      animation:fieldsweep 0.4s ease-out forwards;
    }
    @keyframes fieldsweep{from{left:-100%}to{left:100%}}

    .login-error{
      font-family:var(--font-mono);font-size:9px;color:var(--red);
      text-align:center;margin-bottom:12px;min-height:14px;
      animation:shake 0.35s ease-out;letter-spacing:0.08em;
    }
    @keyframes shake{0%,100%{transform:translateX(0)}20%{transform:translateX(-6px)}40%{transform:translateX(6px)}60%{transform:translateX(-4px)}80%{transform:translateX(4px)}}

    .login-btn{
      width:100%;padding:13px;margin-top:4px;
      font-family:var(--font-display);font-size:11px;font-weight:700;
      letter-spacing:0.18em;color:var(--bg);
      background:linear-gradient(135deg,var(--cyan),var(--cyan2));
      border:none;border-radius:3px;cursor:pointer;
      position:relative;overflow:hidden;
      transition:transform 0.15s, box-shadow 0.15s;
      box-shadow:0 0 20px rgba(0,210,180,0.3);
    }
    .login-btn::before{
      content:'';position:absolute;inset:0;
      background:linear-gradient(135deg,var(--cyan2),var(--cyan3));
      opacity:0;transition:opacity 0.2s;
    }
    .login-btn:hover::before{opacity:1}
    .login-btn:hover{transform:translateY(-1px);box-shadow:0 4px 24px rgba(0,210,180,0.5)}
    .login-btn:active{transform:translateY(1px);box-shadow:0 0 12px rgba(0,210,180,0.3)}
    /* ripple on click */
    .login-btn::after{
      content:'';position:absolute;inset:0;
      background:radial-gradient(circle at var(--rx,50%) var(--ry,50%),rgba(255,255,255,0.3) 0%,transparent 60%);
      opacity:0;transition:opacity 0.4s;
    }
    .login-btn.ripple::after{opacity:1;animation:ripplefade 0.5s ease-out forwards}
    @keyframes ripplefade{from{opacity:0.6}to{opacity:0}}
    /* scanning line inside button */
    .login-btn-inner{position:relative;z-index:1}

    .login-hint{
      font-family:var(--font-mono);font-size:8px;color:var(--text3);
      text-align:center;margin-top:20px;letter-spacing:0.1em;
    }
    .login-hint span{color:var(--cyan);cursor:pointer;transition:color 0.15s}
    .login-hint span:hover{color:var(--cyan2);text-shadow:var(--glow-cyan)}

    /* loading dots */
    .login-loading{display:none;text-align:center;margin:12px 0}
    .login-loading.active{display:block}
    .login-loading span{
      display:inline-block;width:5px;height:5px;border-radius:50%;
      background:var(--cyan2);margin:0 3px;
      animation:dotpulse 1.2s ease-in-out infinite;
    }
    .login-loading span:nth-child(2){animation-delay:0.2s}
    .login-loading span:nth-child(3){animation-delay:0.4s}
    @keyframes dotpulse{0%,80%,100%{transform:scale(0.6);opacity:0.3}40%{transform:scale(1);opacity:1}}

    /* access granted flash */
    .login-granted{
      position:absolute;inset:0;z-index:10;
      display:none;align-items:center;justify-content:center;
      background:rgba(0,255,136,0.04);border-radius:4px;
    }
    .login-granted.active{display:flex}
    .login-granted-text{
      font-family:var(--font-display);font-size:14px;font-weight:900;
      color:var(--green);letter-spacing:0.25em;
      text-shadow:var(--glow-green);
      animation:grantedpulse 0.5s ease-out;
    }
    @keyframes grantedpulse{from{opacity:0;transform:scale(0.8)}to{opacity:1;transform:scale(1)}}

    /* corner decorators */
    .login-corner{position:absolute;width:14px;height:14px;pointer-events:none}
    .login-corner::before,.login-corner::after{content:'';position:absolute;background:var(--cyan2)}
    .login-corner::before{width:100%;height:1px}
    .login-corner::after{width:1px;height:100%}
    .login-corner.tl{top:10px;left:10px}
    .login-corner.tl::before{top:0;left:0}.login-corner.tl::after{top:0;left:0}
    .login-corner.tr{top:10px;right:10px}
    .login-corner.tr::before{top:0;right:0}.login-corner.tr::after{top:0;right:0}
    .login-corner.bl{bottom:10px;left:10px}
    .login-corner.bl::before{bottom:0;left:0}.login-corner.bl::after{bottom:0;left:0}
    .login-corner.br{bottom:10px;right:10px}
    .login-corner.br::before{bottom:0;right:0}.login-corner.br::after{bottom:0;right:0}
    </style>
    </head>
    <body>
    <div id="login-screen">
      <canvas id="login-canvas"></canvas>
      <div class="login-scan"></div>
      <div class="login-ring login-ring-1"></div>
      <div class="login-ring login-ring-2"></div>
      <div class="login-ring login-ring-3"></div>
      <div id="login-splash">
        <div id="login-splash-text">ORBITAL INSIGHT</div>
        <div id="login-splash-sub">NSH 2026 · ACM v8.0 · ANALYTICS</div>
      </div>
      <div class="login-card">
        <div class="login-corner tl"></div><div class="login-corner tr"></div>
        <div class="login-corner bl"></div><div class="login-corner br"></div>
        <div class="login-granted" id="login-granted">
          <div class="login-granted-text">ACCESS GRANTED</div>
        </div>
        <div class="login-logo">ORBITAL INSIGHT</div>
        <div class="login-sub">NSH 2026 · ACM v8.0 · TEAM BROCODE</div>
        <div class="login-divider"></div>
        <div class="login-card-body">
          <div class="login-field-wrap">
            <label class="login-field-label">OPERATOR ID</label>
            <input class="login-field" id="login-user" type="text" placeholder="Enter operator ID" autocomplete="off" spellcheck="false"/>
            <div class="login-field-sweep"></div>
          </div>
          <div class="login-field-wrap">
            <label class="login-field-label">ACCESS CODE</label>
            <input class="login-field" id="login-pass" type="password" placeholder="Enter access code" autocomplete="off"/>
            <div class="login-field-sweep"></div>
          </div>
          <div class="login-error" id="login-error"></div>
          <div class="login-loading" id="login-loading"><span></span><span></span><span></span></div>
          <button class="login-btn" id="login-btn" onclick="doLogin(event)">
            <span class="login-btn-inner">AUTHENTICATE</span>
          </button>
          <div class="login-hint">Demo credentials: <span onclick="fillDemo()">admin / orbital2026</span></div>
        </div>
      </div>
    </div>
    <script>
    //  LOGIN STAR FIELD
    // ═══════════════════════════════════════════════════════════════════════
    (function initLoginStars(){
      const lc = document.getElementById('login-canvas');
      if(!lc) return;
      const lctx = lc.getContext('2d');
      lc.width = window.innerWidth; lc.height = window.innerHeight;
      const stars = Array.from({length:180}, () => ({
        x: Math.random() * lc.width,
        y: Math.random() * lc.height,
        r: Math.random() * 1.2 + 0.2,
        speed: Math.random() * 0.12 + 0.02,
        twinkle: Math.random() * Math.PI * 2
      }));
      // occasional shooting star
      let shootStar = null, shootTimer = 0;
      function spawnShoot(){
        shootStar = {
          x: Math.random() * lc.width * 0.7, y: Math.random() * lc.height * 0.4,
          len: 80 + Math.random() * 60, speed: 5 + Math.random() * 4,
          angle: Math.PI / 6, life: 1, tailX: 0, tailY: 0
        };
      }
      let _vigCache = null;
      function drawStars(ts){
        // Only resize when dimensions actually change — not every frame
        if(lc.width !== window.innerWidth || lc.height !== window.innerHeight){
          lc.width = window.innerWidth; lc.height = window.innerHeight;
          _vigCache = null; // invalidate cached gradient on resize
        }
        lctx.clearRect(0,0,lc.width,lc.height);
        lctx.fillStyle = '#000a0f'; lctx.fillRect(0,0,lc.width,lc.height);
        // deep blue vignette — cached, recreated only on resize
        if(!_vigCache){
          _vigCache = lctx.createRadialGradient(lc.width/2,lc.height/2,0,lc.width/2,lc.height/2,lc.width*0.6);
          _vigCache.addColorStop(0,'rgba(0,20,40,0.3)'); _vigCache.addColorStop(1,'rgba(0,0,0,0.7)');
        }
        lctx.fillStyle = _vigCache; lctx.fillRect(0,0,lc.width,lc.height);

        stars.forEach(s => {
          s.y -= s.speed; s.twinkle += 0.02;
          if(s.y < 0){ s.y = lc.height; s.x = Math.random() * lc.width; }
          const alpha = 0.3 + 0.4 * Math.abs(Math.sin(s.twinkle));
          lctx.fillStyle = `rgba(180,230,255,${alpha})`;
          lctx.beginPath(); lctx.arc(s.x, s.y, s.r, 0, Math.PI*2); lctx.fill();
        });
        // shooting star
        shootTimer -= 16;
        if(shootTimer <= 0){ spawnShoot(); shootTimer = 4000 + Math.random()*4000; }
        if(shootStar){
          const s = shootStar;
          s.tailX = Math.cos(s.angle) * s.len;
          s.tailY = Math.sin(s.angle) * s.len;
          s.x += Math.cos(s.angle) * s.speed;
          s.y += Math.sin(s.angle) * s.speed;
          s.life -= 0.02;
          if(s.life > 0 && s.x < lc.width && s.y < lc.height){
            const grad = lctx.createLinearGradient(s.x,s.y,s.x-s.tailX,s.y-s.tailY);
            grad.addColorStop(0, `rgba(0,235,215,${s.life})`);
            grad.addColorStop(1, 'transparent');
            lctx.strokeStyle = grad; lctx.lineWidth = 1.5;
            lctx.beginPath(); lctx.moveTo(s.x,s.y); lctx.lineTo(s.x-s.tailX,s.y-s.tailY); lctx.stroke();
          } else { shootStar = null; }
        }
        requestAnimationFrame(drawStars);
      }
      requestAnimationFrame(drawStars);
    })();

    // ═══════════════════════════════════════════════════════════════════════
    //  LOGIN LOGIC
    // ═══════════════════════════════════════════════════════════════════════
    // Valid credentials (demo)
    const VALID_CREDS = [
      {user:'admin', pass:'orbital2026'},
      {user:'nsh2026', pass:'acm'},
      {user:'operator', pass:'insight'},
    ];

    function fillDemo(){
      document.getElementById('login-user').value = 'admin';
      document.getElementById('login-pass').value = 'orbital2026';
      document.getElementById('login-error').textContent = '';
      // trigger sweep animation by briefly unfocusing/focusing
      document.getElementById('login-user').focus();
      setTimeout(()=>document.getElementById('login-pass').focus(), 80);
    }

    function doLogin(e){
      const btn = document.getElementById('login-btn');
      // ripple effect
      const rect = btn.getBoundingClientRect();
      const rx = ((e.clientX - rect.left) / rect.width * 100).toFixed(0);
      const ry = ((e.clientY - rect.top) / rect.height * 100).toFixed(0);
      btn.style.setProperty('--rx', rx + '%');
      btn.style.setProperty('--ry', ry + '%');
      btn.classList.remove('ripple');
      void btn.offsetWidth;
      btn.classList.add('ripple');

      const user = document.getElementById('login-user').value.trim();
      const pass = document.getElementById('login-pass').value.trim();
      const errEl = document.getElementById('login-error');
      const loadEl = document.getElementById('login-loading');

      errEl.textContent = '';
      if(!user || !pass){
        setLoginError('CREDENTIALS REQUIRED');
        return;
      }

      // Show loading
      btn.style.pointerEvents = 'none';
      btn.querySelector('.login-btn-inner').textContent = 'AUTHENTICATING…';
      loadEl.classList.add('active');

      // Simulate auth delay (typewriter style)
      let dots = 0;
      const dotInterval = setInterval(()=>{
        dots = (dots + 1) % 4;
        btn.querySelector('.login-btn-inner').textContent = 'AUTHENTICATING' + '.'.repeat(dots);
      }, 200);

      setTimeout(()=>{
        clearInterval(dotInterval);
        const ok = VALID_CREDS.some(c => c.user === user && c.pass === pass);
        loadEl.classList.remove('active');

        if(ok){
          grantAccess(user);
        } else {
          btn.style.pointerEvents = '';
          btn.querySelector('.login-btn-inner').textContent = 'AUTHENTICATE';
          setLoginError('INVALID CREDENTIALS — ACCESS DENIED');
        }
      }, 1400);
    }

    function setLoginError(msg){
      const el = document.getElementById('login-error');
      el.textContent = '';
      void el.offsetWidth; // force reflow to retrigger animation
      el.textContent = msg;
    }

    function grantAccess(user){
      // Show "ACCESS GRANTED"
      const granted = document.getElementById('login-granted');
      granted.classList.add('active');

      // Green border flash on card
      const card = document.querySelector('.login-card');
      card.style.borderColor = 'rgba(0,255,136,0.5)';
      card.style.boxShadow = '0 0 40px rgba(0,255,136,0.15), 0 0 60px rgba(0,0,0,0.8)';

      // Type operator name into topbar after unlock
      setTimeout(()=>{
        const screen = document.getElementById('login-screen');
        screen.classList.add('fade-out');
        // Start main app
        setTimeout(()=>{
          screen.style.display = 'none';
          // add operator badge to topbar
          const badge = document.createElement('div');
          badge.style.cssText = `font-family:var(--font-mono);font-size:8px;color:var(--text2);
            border:1px solid var(--border);border-radius:2px;padding:2px 8px;
            letter-spacing:0.08em;flex-shrink:0;animation:panelReveal 0.4s ease-out both`;
          badge.textContent = `OPR: ${user.toUpperCase()}`;
          document.getElementById('topbar').appendChild(badge);
        }, 850);
      }, 900);
    }

    // Enter key on password field
    document.addEventListener('DOMContentLoaded', ()=>{
      document.getElementById('login-pass')?.addEventListener('keydown', e => {
        if(e.key === 'Enter') document.getElementById('login-btn').click();
      });
      document.getElementById('login-user')?.addEventListener('keydown', e => {
        if(e.key === 'Enter') document.getElementById('login-pass')?.focus();
      });

      // ── Bug 1 fix: measure actual card position and set --splash-shift dynamically
      // so the title lands exactly on the card header regardless of screen height.
      // We wait one rAF so flexbox has finished layout before measuring.
      requestAnimationFrame(() => {
        const splash = document.getElementById('login-splash');
        const card   = document.querySelector('.login-card');
        if(splash && card){
          const cardRect   = card.getBoundingClientRect();
          const screenMidY = window.innerHeight / 2;
          // Logo sits ~54px from the top of the card (padding-top 44px + half line-height ~10px)
          const logoY  = cardRect.top + 54;
          const shift  = logoY - screenMidY;   // negative = upward
          splash.style.setProperty('--splash-shift', `${shift}px`);
        }
      });

      // ── Bug 2 fix: remove splash from DOM once its animation ends
      // stops the infinite splashGlow from compositing on an invisible element.
      const splash = document.getElementById('login-splash');
      if(splash){
        splash.addEventListener('animationend', () => splash.remove(), { once: true });
      }

      // ── Bug 4 fix: ensure logo/sub/divider are fully visible after their
      // animation completes — guards against opacity:0 if animation never fires.
      setTimeout(() => {
        document.querySelectorAll('.login-logo,.login-sub,.login-divider')
          .forEach(el => el.style.opacity = '1');
      }, 3000); // well after the 2.45s + 0.35s animation window
    });

    // Override grantAccess — redirect Streamlit parent with auth params
    function grantAccess(user){
      const granted = document.getElementById("login-granted");
      granted.classList.add("active");
      const card = document.querySelector(".login-card");
      card.style.borderColor = "rgba(0,255,136,0.5)";
      card.style.boxShadow = "0 0 40px rgba(0,255,136,0.15),0 0 60px rgba(0,0,0,0.8)";
      setTimeout(()=>{
        const screen = document.getElementById("login-screen");
        screen.classList.add("fade-out");
        setTimeout(()=>{
          const pass = document.getElementById("login-pass").value.trim();
          const params = "?oi_auth_user=" + encodeURIComponent(user)
                       + "&oi_auth_pass=" + encodeURIComponent(pass);
          window.location.href = "/" + params;
        }, 850);
      }, 900);
    }
    </script>
    </body>
    </html>
"""

    _components.html(_login_html, height=900, scrolling=False)
    st.stop()

# ── Cursor ring + tooltip (shown after login) ─────────────────────────────────
st.markdown("""
<div id="oi-cur"></div>
<div id="oi-tip"></div>
<script>
(function(){
  var ring=document.getElementById('oi-cur'),tip=document.getElementById('oi-tip');
  if(ring){
    document.addEventListener('mousemove',function(e){ring.style.left=e.clientX+'px';ring.style.top=e.clientY+'px';});
    document.addEventListener('mousedown',function(){ring.classList.add('c');});
    document.addEventListener('mouseup',function(){ring.classList.remove('c');});
    document.addEventListener('mouseover',function(e){
      if(e.target.closest('button,a,.metric-card,.cdm-item,.contact-win,.gs-card,.gs-item,.sat-row,.rubric-box,.panel-badge,[data-tip]'))ring.classList.add('h');
      else ring.classList.remove('h');
    });
  }
  if(tip){
    document.addEventListener('mousemove',function(e){tip.style.left=(e.clientX+16)+'px';tip.style.top=(e.clientY-8)+'px';});
    document.addEventListener('mouseover',function(e){var el=e.target.closest('[data-tip]');if(el){tip.innerHTML=el.getAttribute('data-tip').replace(/&#10;/g,'<br>');tip.style.opacity='1';}});
    document.addEventListener('mouseout',function(e){if(e.target.closest('[data-tip]'))tip.style.opacity='0';});
  }
})();
</script>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  API HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=2)
def get_status():
    try: return requests.get(f"{API}/status", timeout=3).json()
    except: return {}

@st.cache_data(ttl=2)
def get_satellites():
    try: return requests.get(f"{API}/satellites", timeout=5).json()
    except: return []

@st.cache_data(ttl=2)
def get_conjunctions():
    try: return requests.get(f"{API}/conjunctions", timeout=3).json()
    except: return []

@st.cache_data(ttl=5)
def get_cdm_registry(limit=100):
    try: return requests.get(f"{API}/cdm/registry?limit={limit}", timeout=3).json()
    except: return []

@st.cache_data(ttl=3)
def get_maneuver_history(limit=300):
    try: return requests.get(f"{API}/maneuver/history?limit={limit}", timeout=3).json()
    except: return []

@st.cache_data(ttl=3)
def get_ground_stations():
    try: return requests.get(f"{API}/ground_stations", timeout=3).json()
    except: return []

@st.cache_data(ttl=3)
def get_heatmap():
    try: return requests.get(f"{API}/fleet/heatmap", timeout=3).json()
    except: return []

@st.cache_data(ttl=4)
def get_fleet_uptime():
    try: return requests.get(f"{API}/fleet/uptime", timeout=3).json()
    except: return {}

@st.cache_data(ttl=4)
def get_fleet_contact_summary():
    try: return requests.get(f"{API}/fleet/contact_summary", timeout=3).json()
    except: return {}

@st.cache_data(ttl=4)
def get_contact_schedule(sat_id):
    try: return requests.get(f"{API}/satellite/{sat_id}/contact_schedule", timeout=3).json()
    except: return {}

@st.cache_data(ttl=3)
def get_metrics():
    try: return requests.get(f"{API}/metrics", timeout=3).json()
    except: return {}

# ── ML v8.0 helpers — ttl=2 to match sat/conjunction refresh rate ─────────────
@st.cache_data(ttl=2)
def get_ml_bandit():
    try:
        r = requests.get(f"{API}/ml/bandit", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=2)
def get_ml_anomalies():
    try:
        r = requests.get(f"{API}/ml/anomalies?top=30", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=2)
def get_ml_fuel_forecast():
    try:
        r = requests.get(f"{API}/ml/fuel_forecast", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=2)
def get_ml_risk_trends():
    try:
        r = requests.get(f"{API}/ml/risk_trends?top=30", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=2)
def get_ml_summary():
    try:
        r = requests.get(f"{API}/ml/summary", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

def api_online():
    try: requests.get(f"{API}/status", timeout=2); return True
    except: return False

# ── Component helpers ──────────────────────────────────────────────────────────
def ph(title, badge="", badge_cls="badge-cyan"):
    """panel_header — exact index.html .panel-hdr"""
    b = f'<span class="panel-badge {badge_cls}">{badge}</span>' if badge else ""
    return f'<div class="panel-hdr"><div class="panel-title">{title}</div>{b}</div>'

def mc(label, val, color="", delta="", tip=""):
    """metric_card with optional data-tip tooltip"""
    tip_attr = f' data-tip="{tip}"' if tip else ""
    dh = f'<div class="metric-delta">{delta}</div>' if delta else ""
    return f'<div class="metric-card"{tip_attr}><div class="metric-label">{label}</div><div class="metric-value {color}">{val}</div>{dh}</div>'

def badge(text, cls="cyan"):
    return f'<span class="panel-badge badge-{cls}">{text}</span>'

def cdm_item(sat, deb, dist_m, pc, risk, tca, pruned=False):
    cls = "crit" if risk=="RED" else "warn" if risk=="YELLOW" else "safe"
    col = "var(--red)" if risk=="RED" else "var(--yellow)" if risk=="YELLOW" else "var(--green)"
    prune_html = '<div class="cdm-pruned">🟣 Pc PRUNED</div>' if pruned else ""
    return f'''<div class="cdm-item {cls}" data-tip="Sat: {sat}&#10;Debris: {deb}&#10;Pc: {pc:.2e}&#10;TCA: {tca}">
      <div class="cdm-ids">{sat.replace("SAT-Alpha-","A-")} ↔ {deb}</div>
      <div class="cdm-dist" style="color:{col}">{dist_m:.0f} m</div>
      <div class="cdm-meta"><span class="cdm-pc">Pc {pc:.1e}</span><span class="cdm-tca">{tca[:16]}</span></div>
      {prune_html}
    </div>'''

def uptime_bar(sat_id, pct):
    if pct >= 99:   col = "var(--green)"
    elif pct >= 95: col = "var(--cyan)"
    elif pct >= 90: col = "var(--yellow)"
    else:           col = "var(--red)"
    return f'''<div class="uptime-bar-row">
      <div class="uptime-sat-id">{sat_id.replace("SAT-Alpha-","A-")}</div>
      <div class="uptime-bar-bg"><div class="uptime-bar-fill" style="width:{pct}%;background:{col};color:{col}"></div></div>
      <div class="uptime-pct" style="color:{col}">{pct:.1f}%</div>
    </div>'''

def altair_chart(df, x_field, y_field, color_field=None, color_scale=None, height=300, tooltips=None):
    # If y_field is already an alt.Y object, use it directly — don't double-wrap
    if isinstance(y_field, alt.Y):
        y_enc = y_field
    else:
        y_enc = alt.Y(y_field, axis=alt.Axis(labelColor="rgba(180,230,220,0.5)", labelFont="Share Tech Mono", labelFontSize=9))
    enc = dict(
        x=alt.X(x_field, axis=alt.Axis(labelColor="rgba(180,230,220,0.5)", gridColor="rgba(0,210,180,0.08)", labelFont="Share Tech Mono", labelFontSize=9)),
        y=y_enc,
    )
    if color_field and color_scale:
        enc["color"] = alt.Color(color_field, scale=color_scale, legend=None)
    if tooltips:
        enc["tooltip"] = tooltips
    return alt.Chart(df).mark_bar(cornerRadiusTopRight=2, cornerRadiusBottomRight=2).encode(
        **enc
    ).properties(height=height, background="transparent").configure_view(
        strokeOpacity=0
    ).configure_axis(gridColor="rgba(0,210,180,0.07)", domainColor="rgba(0,210,180,0.15)")

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    online      = api_online()
    status      = get_status() if online else {}
    uptime_q    = get_fleet_uptime() if online else {}
    metrics     = get_metrics() if online else {}

    st.markdown(f'''
    <div class="tb-logo">🛰 ORBITAL<span style="color:var(--muted);font-weight:400"> INSIGHT</span></div>
    <div class="tb-sub">NSH 2026 · ACM v7.4 · TEAM BROCODE</div>
    ''', unsafe_allow_html=True)

    if online:
        spatial = status.get("spatial_index", "?")
        sim_t_s = int(status.get("sim_time", 0))
        st.markdown(
            f'{badge("⬤ LIVE","green")} {badge(spatial.upper(),"purple")} {badge(f"T+{sim_t_s//3600:03d}H","cyan")}',
            unsafe_allow_html=True)
    else:
        st.markdown(badge("⬤ OFFLINE","red"), unsafe_allow_html=True)
        st.warning("Backend not reachable")

    st.markdown(ph("NAVIGATION"), unsafe_allow_html=True)
    page = st.radio("", [
        "📊 Dashboard",
        "🛰 Fleet Status",
        "📡 Contact Schedule",
        "📈 Uptime Monitor",
        "⚠ CDM Registry",
        "🔥 Maneuver History",
        "🌐 Ground Stations",
        "🤖 ML Intelligence",
        "🗺 Live Visualizer",
    ], label_visibility="collapsed")

    st.markdown(ph("SIM CONTROL"), unsafe_allow_html=True)
    step_hrs = st.slider("Step (hours)", 0.25, 6.0, 1.0, 0.25)
    if st.button("▶ ADVANCE SIM", use_container_width=True):
        try:
            r = requests.post(f"{API}/simulate/step",
                              json={"step_seconds": step_hrs*3600}, timeout=300).json()
            st.session_state["sim_step_target"] = r.get("target_sim_time", 0)
            st.session_state["sim_step_start"]  = r.get("current_sim_time", 0)
            st.session_state["sim_step_hrs"]    = step_hrs
            st.rerun()
        except Exception as e:
            st.error(f"Step failed: {e}")

    # Show live progress if a step is in progress
    if "sim_step_target" in st.session_state and online:
        try:
            target_t = st.session_state["sim_step_target"]
            start_t  = st.session_state["sim_step_start"]
            step_s   = st.session_state.get("sim_step_hrs", 1) * 3600
            cur_t    = requests.get(f"{API}/status", timeout=2).json().get("sim_time", start_t)
            pct      = min(1.0, (cur_t - start_t) / max(step_s, 1))

            if cur_t >= target_t - 30:
                st.success(f"✓ {st.session_state.get('sim_step_hrs',1):.2f}h simulated")
                for k in ["sim_step_target","sim_step_start","sim_step_hrs"]:
                    st.session_state.pop(k, None)
                st.cache_data.clear()
            else:
                st.progress(pct)
                mans = requests.get(f"{API}/status", timeout=2).json().get("maneuvers_executed", 0)
                st.markdown(
                    f'<div style="font-family:var(--font-mono);font-size:9px;'
                    f'color:var(--cyan2);letter-spacing:0.1em;margin-top:4px">'
                    f'⏱ {pct*100:.0f}% · {mans} burns · T+{int(cur_t//3600):04d}H</div>',
                    unsafe_allow_html=True)
        except Exception:
            pass

    if online and uptime_q:
        fp_q    = uptime_q.get("fleet_uptime_pct", 0)
        grade_q = uptime_q.get("grade", "—")
        gcol_q  = {"EXCELLENT":"green","GOOD":"cyan","ACCEPTABLE":"yellow","POOR":"red"}.get(grade_q,"")
        nom_q   = status.get("satellites_nominal", 0)
        eol_q   = status.get("satellites_eol", 0)
        cdm_q   = status.get("active_conjunctions", 0)
        step_ms = metrics.get("step_ms_avg", 0)
        st.markdown(ph("FLEET HEALTH"), unsafe_allow_html=True)
        st.markdown(f'''
        <div class="sidebar-uptime">
          <div class="metric-label">FLEET UPTIME</div>
          <div class="metric-value {gcol_q}" style="font-size:24px">{fp_q:.1f}%</div>
          <div class="metric-delta grade-{grade_q}">{grade_q}</div>
        </div>
        <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px">
          {badge(f"{nom_q} NOM","green")}
          {badge(f"{eol_q} EOL","purple") if eol_q else ""}
          {badge(f"{cdm_q} CDM","red") if cdm_q else badge("0 CDM","green")}
          {badge(f"{step_ms:.0f}ms","cyan") if step_ms else ""}
        </div>''', unsafe_allow_html=True)

    auto_refresh = st.checkbox("Auto-refresh (3s)", value=False)
    if auto_refresh:
        time.sleep(3); st.cache_data.clear(); st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Dashboard ──────────────────────────────────────────────────────────────────
if "Dashboard" in page:
    status      = get_status()
    satellites  = get_satellites()
    conjunctions= get_conjunctions()
    uptime_data = get_fleet_uptime()
    metrics     = get_metrics()

    sim_t   = int(status.get("sim_time", 0))
    sp_idx  = status.get("spatial_index", "—").upper()
    step_ms = metrics.get("step_ms_avg", 0)

    st.markdown(ph("MISSION COMMAND", sp_idx, "badge-purple"), unsafe_allow_html=True)
    st.markdown(f'''<div style="font-family:var(--font-mono);font-size:9px;color:var(--text2);
        letter-spacing:0.12em;margin-bottom:12px;padding-left:2px">
      SIM T+{sim_t//3600:04d}H {(sim_t%3600)//60:02d}M &nbsp;·&nbsp;
      {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC &nbsp;·&nbsp;
      STEP {step_ms:.1f}ms AVG
    </div>''', unsafe_allow_html=True)

    # Telem grid top row
    fp    = uptime_data.get("fleet_uptime_pct")
    grade = uptime_data.get("grade", "—")
    gcol  = {"EXCELLENT":"green","GOOD":"cyan","ACCEPTABLE":"yellow","POOR":"red"}.get(grade, "")
    tp    = sum(s.get("pc_prune_count", 0) for s in satellites)
    af    = sum(s.get("fuel_pct", 0) for s in satellites)/len(satellites) if satellites else 0
    eol   = status.get("satellites_eol", 0)
    cdmc  = status.get("active_conjunctions", 0)
    mans  = status.get("maneuvers_executed", 0)
    nom   = status.get("satellites_nominal", 0)

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    cards = [
        (c1, "NOMINAL SATS", nom,                                           "green",  "In slot",      f"Active satellites in station-keeping box"),
        (c2, "CDM ACTIVE",   cdmc,                                          "red" if cdmc else "green","Critical",  f"Active conjunction data messages"),
        (c3, "FLEET UPTIME", f"{fp:.1f}%" if fp is not None else "—",      gcol,     grade,          f"NSH rubric: ≥99% EXCELLENT · ≥95% GOOD"),
        (c4, "Pc PRUNED",    tp,                                            "purple", "Burns saved",  f"Burns skipped (Pc < 1e-6)&#10;Preserves ΔV budget"),
        (c5, "FUEL AVG",     f"{af:.0f}%",                                  "yellow" if af<50 else "", "Fleet mean", f"Fleet average fuel remaining"),
        (c6, "MANEUVERS",    mans,                                          "cyan",   "Executed",     f"Total burns executed since epoch"),
        (c7, "EOL SATS",     eol,                                           "red" if eol else "green","Graveyard",  f"Satellites in Hohmann graveyard transfer"),
    ]
    for col,(lbl,val,clr,dlt,tip) in zip([c1,c2,c3,c4,c5,c6,c7],[(x[1],x[2],x[3],x[4],x[5]) for x in cards]):
        with col: st.markdown(mc(lbl, val, clr, dlt, tip), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    cl, cr = st.columns([2.2, 1])

    with cl:
        st.markdown(ph("FLEET FUEL & ΔV DISTRIBUTION"), unsafe_allow_html=True)
        if satellites:
            df = pd.DataFrame([{
                "Satellite":  s["id"].replace("SAT-Alpha-","A-"),
                "Fuel %":     round(s.get("fuel_pct", 0), 1),
                "Status":     s.get("status", "?"),
                "Slot km":    round(s.get("slot_distance_km", 0), 2),
                "ΔV m/s":     round((s.get("total_dv_used_kms", 0) or 0)*1000, 1),
                "Avoided":    s.get("collisions_avoided", 0),
                "Pc Pruned":  s.get("pc_prune_count", 0),
            } for s in satellites]).sort_values("Fuel %")

            chart = altair_chart(df,
                x_field="Fuel %:Q",
                y_field=alt.Y("Satellite:N", sort="-x", axis=alt.Axis(labelColor="rgba(180,230,220,0.5)", labelFont="Share Tech Mono", labelFontSize=9)),
                color_field="Fuel %:Q",
                color_scale=alt.Scale(domain=[0,15,35,65,100], range=["#ff2244","#ff7b00","#ffd700","#00c8b4","#00ff88"]),
                height=320,
                tooltips=["Satellite","Fuel %","Status","Slot km","ΔV m/s","Avoided","Pc Pruned"]
            )
            st.altair_chart(chart, use_container_width=True)

    with cr:
        n_conj = len(conjunctions)
        st.markdown(ph("ACTIVE CONJUNCTIONS", str(n_conj), "badge-red" if n_conj else "badge-green"), unsafe_allow_html=True)
        if conjunctions:
            for c in conjunctions[:8]:
                st.markdown(cdm_item(
                    c.get("satellite_id","?"), c.get("debris_id","?"),
                    c.get("miss_distance_m", c.get("miss_distance",0)*1000),
                    c.get("probability", 0), c.get("risk_level","GREEN"),
                    c.get("tca_iso","?"), c.get("pc_pruned", False)
                ), unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="padding:12px 0">{badge("✓ NO ACTIVE CONJUNCTIONS","green")}</div>', unsafe_allow_html=True)

    # Uptime bars
    if uptime_data and uptime_data.get("per_satellite"):
        st.markdown(ph("PER-SATELLITE UPTIME", f"{fp:.1f}%" if fp else "—", f"badge-{gcol}" if gcol else "badge-cyan"), unsafe_allow_html=True)
        bars = "".join(uptime_bar(s["id"], s["uptime_pct"]) for s in sorted(uptime_data["per_satellite"], key=lambda x: x["uptime_pct"]))
        st.markdown(f'<div style="padding:4px 0">{bars}</div>', unsafe_allow_html=True)


# ── Fleet Status ───────────────────────────────────────────────────────────────
elif "Fleet Status" in page:
    satellites = get_satellites()
    conj_ids   = set(c["satellite_id"] for c in get_conjunctions())
    heatmap    = get_heatmap()

    st.markdown(ph("FLEET STATUS ROSTER", f"{len(satellites)} SATS", "badge-cyan"), unsafe_allow_html=True)

    if satellites:
        rows = []
        for s in satellites:
            rows.append({
                "": {"NOMINAL":"🟢","MANEUVERING":"🟠","OUT_OF_SLOT":"🟡","EOL":"🔴"}.get(s["status"],"⚪"),
                "ID": s["id"].replace("SAT-Alpha-","A-"),
                "Status": s["status"],
                "Fuel %": round(s.get("fuel_pct", 0), 1),
                "Alt km": s.get("altitude_km","?"),
                "Slot Δ km": s.get("slot_distance_km","?"),
                "ΔV m/s": round((s.get("total_dv_used_kms",0) or 0)*1000, 1),
                "Avoided": s.get("collisions_avoided",0),
                "Pc Pruned": s.get("pc_prune_count",0),
                "In Slot": "✓" if s.get("in_slot") else "✗",
                "CDM": "⚠" if s["id"] in conj_ids else "",
                "Cooldown s": s.get("cooldown_remaining_s","?"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     column_config={
                         "Fuel %": st.column_config.ProgressColumn("Fuel %", min_value=0, max_value=100, format="%.1f%%"),
                         "Slot Δ km": st.column_config.NumberColumn(format="%.2f km"),
                         "Pc Pruned": st.column_config.NumberColumn("🟣 Pruned"),
                     })

        # Telem grid for selected satellite
        if heatmap:
            st.markdown(ph("FLEET TELEMETRY GRID"), unsafe_allow_html=True)
            total_dv   = sum(s.get("total_dv_kms", 0)*1000 for s in heatmap)
            total_avoid= sum(s.get("collisions_avoided",0) for s in heatmap)
            total_prune= sum(s.get("pc_prune_count",0) for s in heatmap)
            avg_slot   = sum(s.get("slot_distance_km",0) for s in heatmap)/len(heatmap) if heatmap else 0
            st.markdown(f'''<div class="telem-grid">
              <div class="telem-val" data-tip="Total ΔV consumed fleet-wide">
                <div class="telem-label">TOTAL ΔV</div>
                <div class="telem-num {'warn' if total_dv>200 else ''}">{total_dv:.1f} m/s</div>
              </div>
              <div class="telem-val" data-tip="Total collisions avoided">
                <div class="telem-label">AVOIDED</div>
                <div class="telem-num green">{total_avoid}</div>
              </div>
              <div class="telem-val" data-tip="Burns skipped via Pc pruning">
                <div class="telem-label">Pc PRUNED</div>
                <div class="telem-num purple">{total_prune}</div>
              </div>
              <div class="telem-val" data-tip="Average slot distance fleet-wide">
                <div class="telem-label">AVG SLOT Δ</div>
                <div class="telem-num {'warn' if avg_slot>5 else ''}">{avg_slot:.2f} km</div>
              </div>
            </div>''', unsafe_allow_html=True)


# ── Contact Schedule ───────────────────────────────────────────────────────────
elif "Contact Schedule" in page:
    satellites      = get_satellites()
    contact_summary = get_fleet_contact_summary()
    summary_sats    = contact_summary.get("satellites", [])

    st.markdown(ph("PREDICTIVE CONTACT SCHEDULE", "4-HOUR HORIZON", "badge-blue"), unsafe_allow_html=True)

    if summary_sats:
        in_now     = sum(1 for s in summary_sats if s.get("in_contact_now"))
        blackouts  = sum(1 for s in summary_sats if (s.get("next_window") or {}).get("is_last_before_blackout"))
        c1,c2,c3  = st.columns(3)
        with c1: st.markdown(mc("IN CONTACT NOW", in_now, "green", "Satellites"), unsafe_allow_html=True)
        with c2: st.markdown(mc("TOTAL ACTIVE", len(summary_sats), "cyan", "Fleet"), unsafe_allow_html=True)
        with c3: st.markdown(mc("PRE-BLACKOUT", blackouts, "orange" if blackouts else "green", "Last window alerts"), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(ph("FLEET CONTACT OVERVIEW"), unsafe_allow_html=True)
        rows = []
        for s in summary_sats:
            nw = s.get("next_window") or {}
            rows.append({
                "Satellite":   s["id"].replace("SAT-Alpha-","A-"),
                "In Contact":  "📡 YES" if s.get("in_contact_now") else "○ No",
                "Current GS":  s.get("current_gs") or "—",
                "El °":        s.get("current_elevation_deg") or "—",
                "Next GS":     nw.get("gs_id","—"),
                "Window Start":nw.get("start_iso","")[:16] if nw.get("start_iso") else "—",
                "Duration s":  nw.get("duration_s","—"),
                "Peak El °":   nw.get("peak_el_deg","—"),
                "Blackout?":   "⚠ YES" if nw.get("is_last_before_blackout") else "",
                "Pc Pruned":   s.get("pc_prune_count",0),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(ph("PER-SATELLITE DETAIL"), unsafe_allow_html=True)
    sat_ids   = [s["id"] for s in satellites[:20]]
    sat_short = [s.replace("SAT-Alpha-","A-") for s in sat_ids]
    sel       = st.selectbox("Select Satellite", sat_short)
    sel_id    = sat_ids[sat_short.index(sel)] if sel in sat_short else None

    if sel_id:
        cs      = get_contact_schedule(sel_id)
        windows = cs.get("windows", [])
        if windows:
            cols = st.columns(min(len(windows), 3))
            for i, w in enumerate(windows[:3]):
                blackout = w.get("is_last_before_blackout", False)
                with cols[i]:
                    cls = "contact-win blackout" if blackout else "contact-win"
                    tag = '<div class="cw-blackout-tag">⚠ LAST BEFORE BLACKOUT</div>' if blackout else ""
                    st.markdown(f'''<div class="{cls}" data-tip="{w.get('gs_id','?')}&#10;Peak El: {w.get('peak_elevation_deg',0):.1f}°">
                      <div class="cw-gs">
                        <span style="font-family:var(--font-display);font-size:13px;color:var(--cyan2)">{w.get('gs_id','?')}</span>
                        {badge("BLACKOUT","orange") if blackout else ""}
                      </div>
                      <div class="cw-time">START &nbsp;{w.get('start_iso','?')[:19]} UTC</div>
                      <div class="cw-time" style="margin-bottom:6px">END &nbsp;&nbsp;{w.get('end_iso','?')[:19]} UTC</div>
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <span class="cw-dur">⏱ {w.get('duration_s',0):.0f}s</span>
                        <span class="cw-el">⬆ {w.get('peak_elevation_deg',0):.1f}°</span>
                      </div>
                      {tag}
                    </div>''', unsafe_allow_html=True)
        else:
            st.info("No contact windows found for this satellite.")


# ── Uptime Monitor ─────────────────────────────────────────────────────────────
elif "Uptime Monitor" in page:
    uptime_data = get_fleet_uptime()

    st.markdown(ph("FLEET UPTIME MONITOR", "STATION-KEEPING 10km", "badge-green"), unsafe_allow_html=True)

    if uptime_data:
        fp        = uptime_data.get("fleet_uptime_pct", 0)
        grade     = uptime_data.get("grade", "—")
        sim_el    = uptime_data.get("sim_time_elapsed_s", 0)
        active_s  = uptime_data.get("active_satellites", 0)
        gcol      = {"EXCELLENT":"green","GOOD":"cyan","ACCEPTABLE":"yellow","POOR":"red"}.get(grade,"")

        # Fleet uptime card (exact index.html .uptime-fleet-card)
        st.markdown(f'''<div class="uptime-fleet-card">
          <div class="metric-label">FLEET UPTIME</div>
          <div class="uptime-fleet-num grade-{grade}">{fp:.2f}%</div>
          <div class="uptime-grade grade-{grade}">{grade}</div>
        </div>''', unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(mc("ACTIVE SATS",    active_s,              "cyan",  "In constellation"), unsafe_allow_html=True)
        with c2: st.markdown(mc("SIM ELAPSED",    f"{sim_el/3600:.1f}h", "cyan",  "Simulation time"),  unsafe_allow_html=True)
        with c3: st.markdown(mc("SCORE GRADE",    grade,                 gcol,    "NSH 2026 rubric"),  unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(ph("NSH 2026 SCORING RUBRIC"), unsafe_allow_html=True)
        rc1,rc2,rc3,rc4 = st.columns(4)
        for col,pct,lbl,pts,cls,color in [
            (rc1,"≥ 99%","EXCELLENT","15 pts","excellent","#00ff88"),
            (rc2,"≥ 95%","GOOD","~12 pts","good","#00c8b4"),
            (rc3,"≥ 90%","ACCEPTABLE","~9 pts","ok","#ffd700"),
            (rc4,"< 90%","POOR","—","poor","#ff2244"),
        ]:
            with col:
                st.markdown(f'''<div class="rubric-box rubric-{cls}">
                  <div style="font-family:var(--font-display);font-size:16px;color:{color};font-weight:700">{pct}</div>
                  <div style="font-family:var(--font-mono);font-size:9px;color:{color};letter-spacing:0.1em;margin-top:3px">{lbl}</div>
                  <div style="font-family:var(--font-mono);font-size:8px;color:var(--text2);margin-top:2px">{pts}</div>
                </div>''', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(ph("PER-SATELLITE UPTIME"), unsafe_allow_html=True)
        per_sat = uptime_data.get("per_satellite", [])
        if per_sat:
            bars = "".join(uptime_bar(s["id"], s["uptime_pct"]) for s in sorted(per_sat, key=lambda x: x["uptime_pct"]))
            st.markdown(f'<div style="padding:4px 0">{bars}</div>', unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            df_up = pd.DataFrame([{
                "Satellite": s["id"].replace("SAT-Alpha-","A-"),
                "Uptime %":  s["uptime_pct"],
                "Samples In":s["samples_in_slot"],
                "Total":     s["samples_total"],
                "Status":    s["status"],
            } for s in per_sat])
            st.dataframe(df_up, use_container_width=True, hide_index=True,
                         column_config={"Uptime %": st.column_config.ProgressColumn("Uptime %", min_value=0, max_value=100, format="%.2f%%")})
    else:
        st.info("Uptime data not yet available — wait for simulation to initialise.")


# ── CDM Registry ───────────────────────────────────────────────────────────────
elif "CDM Registry" in page:
    cdms = get_cdm_registry(100)

    red    = sum(1 for c in cdms if c.get("risk_level")=="RED")
    yellow = sum(1 for c in cdms if c.get("risk_level")=="YELLOW")
    pruned = sum(1 for c in cdms if c.get("pc_pruned"))
    evade  = sum(1 for c in cdms if c.get("evasion_planned"))

    st.markdown(ph("CONJUNCTION DATA MESSAGES", f"{len(cdms)} CDMs", "badge-red" if red else "badge-cyan"), unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(mc("TOTAL CDMs",      len(cdms), "cyan",   "Registered"), unsafe_allow_html=True)
    with c2: st.markdown(mc("RED RISK",        red,       "red",    "< 1 km"),     unsafe_allow_html=True)
    with c3: st.markdown(mc("YELLOW RISK",     yellow,    "yellow", "1–5 km"),     unsafe_allow_html=True)
    with c4: st.markdown(mc("EVASION PLANNED", evade,     "green",  "Burns queued"),unsafe_allow_html=True)
    with c5: st.markdown(mc("Pc PRUNED",       pruned,    "purple", "Burns saved"), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    if cdms:
        rows = []
        for c in cdms:
            mult   = c.get("anomaly_multiplier", 1.0)
            pc_raw = c.get("probability_raw", c.get("probability_of_collision", 0))
            rows.append({
                "CDM ID":       c.get("cdm_id","?")[-18:],
                "Satellite":    c.get("satellite_id","?").replace("SAT-Alpha-","A-"),
                "Debris":       c.get("debris_id","?"),
                "Miss m":       round(c.get("miss_distance_m",0), 1),
                "Pc (adj)":     f"{c.get('probability_of_collision',0):.2e}",
                "Pc (raw)":     f"{pc_raw:.2e}",
                "Anom ×":       round(mult, 1),
                "Risk":         c.get("risk_level","GREEN"),
                "Rel Vel km/s": round(c.get("relative_velocity_kms",0), 3),
                "TCA":          c.get("tca_iso","?")[:16],
                "TCA s":        round(c.get("time_to_tca_s",0)),
                "Evasion":      "✓" if c.get("evasion_planned") else "—",
                "Pruned":       "🟣" if c.get("pc_pruned") else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     column_config={
                         "Miss m":  st.column_config.NumberColumn("Miss (m)",  format="%.1f m"),
                         "TCA s":   st.column_config.NumberColumn("TCA (s)",   format="%d s"),
                         "Anom ×":  st.column_config.NumberColumn("Anomaly ×",
                             help="ML-2 Isolation Forest Pc multiplier. >1 = anomalous debris flagged"),
                     })


# ── Maneuver History ───────────────────────────────────────────────────────────
elif "Maneuver History" in page:
    history = get_maneuver_history(300)

    st.markdown(ph("MANEUVER HISTORY", f"{len(history)} BURNS", "badge-yellow"), unsafe_allow_html=True)

    if history:
        total_dv   = sum((h.get("dv_mag_kms",0) or 0)*1000 for h in history)
        pre_upload = sum(1 for h in history if h.get("pre_upload"))
        graveyard  = sum(1 for h in history if h.get("burn_type")=="graveyard")
        evasion    = sum(1 for h in history if h.get("burn_type")=="evasion")
        recovery   = sum(1 for h in history if h.get("burn_type")=="recovery")
        stationkeep= sum(1 for h in history if h.get("burn_type")=="stationkeep")

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: st.markdown(mc("TOTAL BURNS",   len(history),       "cyan",   "All types",    "All executed burns since sim epoch"), unsafe_allow_html=True)
        with c2: st.markdown(mc("TOTAL ΔV",      f"{total_dv:.1f}m/s","yellow", "Fleet budget", "Total ΔV consumed fleet-wide"), unsafe_allow_html=True)
        with c3: st.markdown(mc("EVASION",        evasion,            "red",    "CDM burns",    "Collision avoidance burns"), unsafe_allow_html=True)
        with c4: st.markdown(mc("RECOVERY",       recovery,           "cyan",   "Hohmann",      "Hohmann phasing recovery"), unsafe_allow_html=True)
        with c5: st.markdown(mc("PRE-UPLOAD",     pre_upload,         "blue",   "§5.4 blind",   "Burns scheduled before LOS blackout"), unsafe_allow_html=True)
        with c6: st.markdown(mc("GRAVEYARD",      graveyard,          "purple", "EOL",          "Two-burn Hohmann graveyard transfers"), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        burn_counts = {}
        for h in history:
            bt = h.get("burn_type","?"); burn_counts[bt] = burn_counts.get(bt,0)+1
        df_bt = pd.DataFrame(list(burn_counts.items()), columns=["Type","Count"])
        color_map = {"evasion":"#ff2244","recovery":"#00c8b4","stationkeep":"#0090ff","graveyard":"#aa44ff","commanded":"#ffd700"}
        bt_chart = alt.Chart(df_bt).mark_bar(cornerRadiusTopRight=3,cornerRadiusBottomRight=3).encode(
            y=alt.Y("Type:N", axis=alt.Axis(labelColor="rgba(180,230,220,0.5)",labelFont="Share Tech Mono",labelFontSize=10)),
            x=alt.X("Count:Q",axis=alt.Axis(labelColor="rgba(180,230,220,0.5)",labelFont="Share Tech Mono",labelFontSize=9)),
            color=alt.Color("Type:N",scale=alt.Scale(domain=list(color_map.keys()),range=list(color_map.values())),
                legend=alt.Legend(labelColor="rgba(180,230,220,0.5)",titleColor="rgba(0,200,180,0.5)",labelFont="Share Tech Mono")),
            tooltip=["Type","Count"]
        ).properties(height=160,background="transparent").configure_view(strokeOpacity=0).configure_axis(gridColor="rgba(0,210,180,0.07)")
        st.altair_chart(bt_chart, use_container_width=True)

        df = pd.DataFrame([{
            "Time":         h.get("executed_iso","?")[:16],
            "Satellite":    h.get("satellite_id","?").replace("SAT-Alpha-","A-"),
            "Type":         h.get("burn_type","?"),
            "ΔV m/s":       round((h.get("dv_mag_kms",0) or 0)*1000, 2),
            "Fuel kg":      round(h.get("fuel_consumed_kg",0), 3),
            "Remaining kg": round(h.get("fuel_remaining_kg",0), 2),
            "Pre-Upload":   "📡" if h.get("pre_upload") else "",
            "GS Window":    h.get("contact_window","—") or "—",
        } for h in history])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No maneuver history yet — simulation is initialising.")


# ── Ground Stations ─────────────────────────────────────────────────────────────
elif "Ground Stations" in page:
    gs_list = get_ground_stations()

    st.markdown(ph("GROUND STATION NETWORK", "6 STATIONS", "badge-blue"), unsafe_allow_html=True)

    if gs_list:
        total_vis = sum(g.get("visible_count",0) for g in gs_list)
        active_gs = sum(1 for g in gs_list if g.get("visible_count",0)>0)

        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(mc("ACTIVE STATIONS", f"{active_gs}/{len(gs_list)}", "green", "In coverage"), unsafe_allow_html=True)
        with c2: st.markdown(mc("TOTAL VISIBILITY", total_vis,                    "cyan",  "Satellite passes"), unsafe_allow_html=True)
        with c3: st.markdown(mc("COVERAGE",         f"{active_gs/len(gs_list)*100:.0f}%", "cyan", "Stations online"), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, gs in enumerate(gs_list):
            with cols[i % 3]:
                vis        = gs.get("visible_count", 0)
                color      = "green" if vis>10 else "yellow" if vis>3 else ""
                raw_sats   = (gs.get("visible_satellites") or [])[:5]
                sat_list   = ", ".join(s.replace("SAT-Alpha-","A-") for s in raw_sats)
                active_cls = "gs-card active" if vis>0 else "gs-card"
                # Build sat list HTML separately to avoid f-string </div> confusion
                sat_html   = f'<div style="font-family:var(--font-mono);font-size:7px;color:var(--green2);margin-top:4px;letter-spacing:0.04em">{sat_list}</div>' if sat_list else ""
                st.markdown(f'''<div class="{active_cls}" data-tip="{gs.get('name','?')}&#10;Min El: {gs.get('min_el',5)}°&#10;Alt: {gs.get('elev_m',0)} m">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <span style="font-family:var(--font-mono);font-size:8px;color:var(--text2);letter-spacing:0.12em">{gs["id"]}</span>
                    {badge("ACTIVE","green") if vis>0 else badge("IDLE","cyan")}
                  </div>
                  <div style="font-family:var(--font-body);font-size:13px;font-weight:700;color:var(--text);margin-bottom:6px;letter-spacing:0.03em">
                    {gs.get("name","?").replace("_"," ")}
                  </div>
                  <div style="font-family:var(--font-mono);font-size:8px;color:var(--text2);line-height:1.7">
                    Lat {gs["lat"]}° · Lon {gs["lon"]}°<br>Min El {gs.get("min_el",5)}° · Alt {gs.get("elev_m",0)} m
                  </div>
                  <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center">
                    <span class="metric-value {color}" style="font-size:22px">{vis}</span>
                    <span style="font-family:var(--font-mono);font-size:8px;color:var(--text2)">visible sats</span>
                  </div>
                  {sat_html}
                </div>''', unsafe_allow_html=True)


# ── Live Visualizer ─────────────────────────────────────────────────────────────
elif "ML Intelligence" in page:
    st.markdown(ph("ML INTELLIGENCE", "v8.0", "badge-purple"), unsafe_allow_html=True)
    st.markdown('<div class="orbital-sub">ZERO EXTERNAL DEPENDENCIES · PURE NUMPY / PURE-PYTHON FALLBACKS</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎰 ΔV Bandit", "🔍 Anomaly Detector", "⛽ Fuel Forecast", "📉 Risk Trends"
    ])

    with tab1:
        b = get_ml_bandit()
        if b.get("_error"):
            st.warning(f"ML-1 endpoint error: {b['_error']}")
        st.markdown(ph("UCB1 GRADIENT BANDIT — EVASION ΔV OPTIMISER"), unsafe_allow_html=True)
        st.markdown('''<div style="background:linear-gradient(135deg,var(--bg3),var(--bg2));
          border:1px solid rgba(170,68,255,0.2);border-radius:4px;padding:12px 14px;margin:10px 0">
          <div class="metric-label">HOW IT WORKS</div>
          <div style="font-size:9px;color:var(--text2);line-height:1.75">
            6 arms: 0.004–0.015 km/s · UCB1 selection · reward = miss_km − 80×dv_kms<br>
            Converges to minimum safe ΔV → saves 20–40% fuel vs fixed 0.010 km/s default.
          </div></div>''', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.metric("Total Updates", b.get("total_updates", 0))
        with c2: st.metric("Best ΔV Arm", f"{b.get('best_dv_kms','—')} km/s")
        arms = b.get("arms", [])
        if arms:
            best_dv = b.get("best_dv_kms", 0)
            df = pd.DataFrame(arms)
            # Use a string category for reliable Vega-Lite color condition
            df["arm_type"] = df["dv_kms"].apply(lambda v: "best" if v == best_dv else "other")
            bar = alt.Chart(df).mark_bar(cornerRadiusTopRight=3).encode(
                x=alt.X("dv_kms:O", axis=alt.Axis(labelColor="#a8c8e0", title="ΔV (km/s)")),
                y=alt.Y("mean_reward:Q", axis=alt.Axis(labelColor="#a8c8e0", title="Mean Reward")),
                color=alt.Color("arm_type:N",
                    scale=alt.Scale(domain=["best","other"],
                                    range=["#aa44ff","#00c8b4"]),
                    legend=None),
                tooltip=["dv_kms","mean_reward","visits"]
            ).properties(height=200, background="#060f1c",
                         title=alt.TitleParams("Mean Reward per ΔV Arm", color="#a8c8e0"))
            st.altair_chart(bar, use_container_width=True)
            st.dataframe(df[["dv_kms","mean_reward","visits"]], use_container_width=True, hide_index=True)
        else:
            st.info("Bandit data accumulates after first evasion burn (~60s)")

    with tab2:
        an = get_ml_anomalies()
        if an.get("_error"):
            st.warning(f"ML-2 endpoint error: {an['_error']}")
        st.markdown(ph("ISOLATION FOREST — DEBRIS ANOMALY SCORER"), unsafe_allow_html=True)
        st.markdown(f'''<div style="background:linear-gradient(135deg,var(--bg3),var(--bg2));
          border:1px solid rgba(170,68,255,0.2);border-radius:4px;padding:12px 14px;margin:10px 0">
          <div class="metric-label">HOW IT WORKS</div>
          <div style="font-size:9px;color:var(--text2);line-height:1.75">
            {an.get("n_trees","?")} trees · {an.get("subsample","?")} samples ·
            retrains every {int((an.get("retrain_interval_s") or 1800)//60)} min<br>
            Score &gt;0.7 → 3× Pc boost · Score &gt;0.85 → 6× Pc boost → earlier evasion.
          </div></div>''', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Status", "TRAINED ✓" if an.get("trained") else "INIT…")
        with c2: st.metric("Debris Scored", an.get("debris_scored",0))
        with c3: st.metric("Trees", an.get("n_trees",0))
        top = an.get("top_anomalies",[])
        if top:
            df = pd.DataFrame(top[:20])
            bar = alt.Chart(df).mark_bar(cornerRadiusTopRight=2).encode(
                x=alt.X("anomaly_score:Q", scale=alt.Scale(domain=[0,1]),
                        axis=alt.Axis(labelColor="#a8c8e0", title="Anomaly Score")),
                y=alt.Y("debris_id:N", sort="-x",
                        axis=alt.Axis(labelColor="#a8c8e0", labelFontSize=8)),
                color=alt.Color("anomaly_score:Q",
                    scale=alt.Scale(domain=[0,0.5,0.7,0.85,1.0],
                                    range=["#00ff88","#00d2ff","#ffd700","#ff7b00","#ff2244"]),
                    legend=None),
                tooltip=["debris_id","anomaly_score"]
            ).properties(height=300, background="#060f1c",
                         title=alt.TitleParams("Top 20 Anomalous Debris", color="#a8c8e0"))
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("Trained at startup — if empty, backend is still initialising")

    with tab3:
        ff = get_ml_fuel_forecast()
        if ff.get("_error"):
            st.warning(f"ML-3 endpoint error: {ff['_error']}")
        st.markdown(ph("ONLINE RLS — FUEL DEPLETION FORECAST"), unsafe_allow_html=True)
        st.markdown(f'''<div style="background:linear-gradient(135deg,var(--bg3),var(--bg2));
          border:1px solid rgba(170,68,255,0.2);border-radius:4px;padding:12px 14px;margin:10px 0">
          <div class="metric-label">HOW IT WORKS</div>
          <div style="font-size:9px;color:var(--text2);line-height:1.75">
            fuel(t) = w₀ + w₁·t · λ={ff.get("lambda_forgetting",0.98)} · EOL threshold: {ff.get("eol_threshold_kg","?")} kg<br>
            Skips SK burns when EOL &lt;3600s · triggers early graveyard at &lt;2h warning.
          </div></div>''', unsafe_allow_html=True)
        sats_ff = ff.get("satellites",[])
        if sats_ff:
            warnings = [s for s in sats_ff if s.get("eol_warning")]
            if warnings:
                st.error(f"⚠ {len(warnings)} satellite(s) predicted EOL within 2 hours!")
                for w in warnings:
                    st.markdown(f'`{w["id"]}` — {w["fuel_now_kg"]} kg — EOL in {(w.get("t_to_eol_s") or 0)/3600:.1f}h')
            df = pd.DataFrame([{
                "Satellite": s["id"].replace("SAT-Alpha-","A-"),
                "Now kg":  s.get("fuel_now_kg",0),
                "+1h kg":  s.get("fuel_1h_kg",0),
                "+6h kg":  s.get("fuel_6h_kg",0),
                "+24h kg": s.get("fuel_24h_kg",0),
                "EOL ⚠":   "⚠" if s.get("eol_warning") else "",
                "Status":  s.get("status","?"),
            } for s in sats_ff])
            # max_value = highest current fuel across all sats, fallback 50
            fuel_max = max((s.get("fuel_now_kg", 0) for s in sats_ff), default=50)
            fuel_max = max(fuel_max, 1)   # guard against all-zero
            st.dataframe(df, use_container_width=True, hide_index=True,
                         column_config={"Now kg": st.column_config.ProgressColumn(
                             "Now kg", min_value=0,
                             max_value=fuel_max,
                             format="%.1f kg")})
        else:
            st.info("Forecast populates after first burn events")

    with tab4:
        rt = get_ml_risk_trends()
        if rt.get("_error"):
            st.warning(f"ML-4 endpoint error: {rt['_error']}")
        st.markdown(ph("EXPONENTIAL SMOOTHING — CONJUNCTION RISK TRENDS"), unsafe_allow_html=True)
        st.markdown(f'''<div style="background:linear-gradient(135deg,var(--bg3),var(--bg2));
          border:1px solid rgba(170,68,255,0.2);border-radius:4px;padding:12px 14px;margin:10px 0">
          <div class="metric-label">HOW IT WORKS</div>
          <div style="font-size:9px;color:var(--text2);line-height:1.75">
            α={rt.get("alpha",0.35)} smoothing per (sat, debris) pair.<br>
            Trend &gt;{rt.get("skip_threshold_kms",0.05)} km/s → diverging → 24h scan skipped → ~30% CPU saving.<br>
            Converging pairs (trend &lt;0) sorted to front of assessment queue.
          </div></div>''', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1: st.metric("Tracked Pairs", int(rt.get("tracked_pairs") or 0))
        with c2: st.metric("Skip Threshold", f"{rt.get('skip_threshold_kms', 0.05):.2f} km/s")
        pairs = rt.get("converging_pairs",[])
        if pairs:
            df = pd.DataFrame(pairs)
            bar = alt.Chart(df).mark_bar(cornerRadiusTopRight=2).encode(
                x=alt.X("trend_kms:Q",
                        axis=alt.Axis(labelColor="#a8c8e0",
                                      title="Trend km/s (negative = converging)")),
                y=alt.Y("satellite_id:N", sort="x",
                        axis=alt.Axis(labelColor="#a8c8e0", labelFontSize=8)),
                color=alt.condition(alt.datum.trend_kms < 0,
                    alt.value("#ff2244"), alt.value("#00ff88")),
                tooltip=["satellite_id","debris_id","trend_kms","smoothed_miss_km"]
            ).properties(height=280, background="#060f1c",
                         title=alt.TitleParams("Converging (sat, debris) Pairs", color="#a8c8e0"))
            st.altair_chart(bar, use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Risk trends accumulate after first conjunction assessment (~60s)")






elif "Live Visualizer" in page:
    frontend_url = _os.environ.get("FRONTEND_URL", "http://localhost:80")

    st.markdown(ph("LIVE ORBITAL DASHBOARD", "HTML CANVAS", "badge-cyan"), unsafe_allow_html=True)
    st.markdown(f'''
    <div style="border:1px solid var(--border2);border-radius:3px;overflow:hidden;
        background:var(--bg);box-shadow:0 0 40px rgba(0,0,0,0.7),0 0 80px rgba(0,200,180,0.02)">
      <div style="background:rgba(0,0,0,0.25);border-bottom:1px solid var(--border);
        padding:6px 12px;display:flex;align-items:center;gap:8px">
        <div class="panel-title">ORBITAL INSIGHT · HTML FRONTEND</div>
        <span class="panel-badge badge-green live-dot" style="margin-left:auto">⬤ LIVE</span>
      </div>
      <iframe src="{frontend_url}" width="100%" height="760" frameborder="0" style="display:block;border:none;"></iframe>
    </div>
    <div style="font-family:var(--font-mono);font-size:8px;color:var(--text3);margin-top:6px;text-align:center;letter-spacing:0.1em">
      <a href="{frontend_url}" target="_blank" style="color:var(--muted);text-decoration:none">↗ FULLSCREEN</a>
      &nbsp;·&nbsp;
      <a href="http://localhost:8000/docs" target="_blank" style="color:var(--muted2);text-decoration:none">API DOCS</a>
      &nbsp;·&nbsp;
      <a href="http://localhost:8000/api/logs" target="_blank" style="color:var(--muted2);text-decoration:none">AUDIT LOG</a>
    </div>
    ''', unsafe_allow_html=True)
