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

# ── Hide Streamlit chrome + inject matching topbar ────────────────────────────
st.markdown("""<style>
#MainMenu,header,footer,[data-testid="stToolbar"],
[data-testid="stDecoration"],[data-testid="stStatusWidget"],
[data-testid="collapsedControl"]{display:none!important}

/* ── Topbar matching index.html exactly ── */
.oi-topbar {
  position:fixed; top:0; left:0; right:0; z-index:999;
  height:38px;
  background:rgba(1,5,8,0.97);
  border-bottom:1px solid rgba(0,210,180,0.22);
  display:flex; align-items:center; gap:16px;
  padding:0 16px;
  font-family:'Share Tech Mono',monospace;
  backdrop-filter:blur(8px);
}
.oi-topbar::after {
  content:''; position:absolute; bottom:-1px; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(0,210,180,0.5),transparent);
}
.oi-logo {
  font-family:'Orbitron',sans-serif; font-size:13px; font-weight:900;
  color:#00e8d0; letter-spacing:0.14em;
  text-shadow:0 0 12px rgba(0,210,180,0.5);
  white-space:nowrap;
}
.oi-logo span { color:rgba(0,220,190,0.35); font-weight:400; }
.oi-tb-divider {
  width:1px; height:18px;
  background:rgba(0,210,180,0.15);
  flex-shrink:0;
}
.oi-tb-item {
  display:flex; flex-direction:column;
  font-size:7px; letter-spacing:0.12em;
  color:rgba(180,230,220,0.5); line-height:1.2;
  white-space:nowrap;
}
.oi-tb-item .val {
  font-size:11px; color:rgba(220,255,248,0.88);
  letter-spacing:0.06em;
}
.oi-tb-item .val.green  { color:#00ff88; text-shadow:0 0 8px rgba(0,255,136,0.4); }
.oi-tb-item .val.red    { color:#ff2244; text-shadow:0 0 8px rgba(255,34,68,0.4); }
.oi-tb-item .val.yellow { color:#ffd700; }
.oi-tb-item .val.cyan   { color:#00e8d0; text-shadow:0 0 8px rgba(0,210,180,0.4); }
.oi-tb-item .val.purple { color:#aa44ff; }
.oi-tb-spacer { flex:1; }
.oi-live-dot {
  display:inline-block; width:6px; height:6px; border-radius:50%;
  background:#00ff88; box-shadow:0 0 6px #00ff88;
  animation:liveDotPulse 1.8s ease-in-out infinite;
  margin-right:5px;
}

/* Push main content below topbar */
.main .block-container { padding-top:52px !important; }
[data-testid="stSidebar"] { top:38px !important; height:calc(100vh - 38px) !important; }
</style>""", unsafe_allow_html=True)

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
.main .block-container {
  padding-top:0.75rem !important; max-width:100% !important;
  position:relative; z-index:1;
}

/* scanlines overlay */
.stApp::before {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.025) 3px,rgba(0,0,0,0.025) 4px);
}
/* top cyan vignette */
.stApp::after {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background:radial-gradient(ellipse at 50% 0%,rgba(0,180,160,0.06) 0%,transparent 65%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background:rgba(3,11,16,0.96) !important;
  border-right:1px solid var(--border2) !important;
  backdrop-filter:blur(12px);
}
[data-testid="stSidebar"]::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,transparent,var(--cyan2),transparent);
  box-shadow:0 0 8px rgba(0,232,208,0.3);
}
[data-testid="stSidebar"] > div { padding-top:0.75rem; }
[data-testid="stSidebar"] .stRadio > div { gap:1px !important; }
[data-testid="stSidebar"] .stRadio label {
  font-family:var(--font-mono) !important;
  background:transparent !important; border:1px solid transparent !important;
  border-radius:3px !important; padding:7px 10px !important; margin:1px 0 !important;
  cursor:pointer !important; transition:all 0.15s !important;
  color:var(--text2) !important; font-size:10px !important; letter-spacing:0.08em !important;
  display:block !important; width:100% !important; position:relative !important; overflow:hidden !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background:rgba(0,200,180,0.06) !important; border-color:var(--border2) !important;
  color:var(--cyan2) !important; padding-left:14px !important;
  text-shadow:var(--glow-cyan) !important;
  box-shadow:0 0 10px rgba(0,200,180,0.06) inset !important;
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
.stButton > button:hover {
  background:rgba(0,200,180,0.12) !important; border-color:var(--border3) !important;
  box-shadow:0 0 24px rgba(0,200,180,0.28), 0 4px 16px rgba(0,0,0,0.4) !important;
  transform:translateY(-2px) !important;
}
.stButton > button:active { transform:translateY(1px) !important; box-shadow:0 0 12px rgba(0,200,180,0.4) !important; }

/* ── Inputs & selects ── */
.stSelectbox div[data-baseweb="select"] > div {
  background:var(--bg3) !important; border-color:var(--border2) !important;
  color:var(--text) !important; font-family:var(--font-mono) !important; font-size:10px !important;
  transition:border-color 0.15s,box-shadow 0.15s !important;
}
.stSelectbox div[data-baseweb="select"]:focus-within > div {
  border-color:var(--cyan2) !important; box-shadow:0 0 12px rgba(0,200,180,0.18) !important;
}
[data-baseweb="popover"] { background:var(--bg3) !important; border:1px solid var(--border2) !important; }
[data-baseweb="menu"] li { color:var(--text2) !important; font-family:var(--font-mono) !important; font-size:10px !important; transition:all 0.1s !important; }
[data-baseweb="menu"] li:hover { background:rgba(0,200,180,0.08) !important; color:var(--cyan2) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background:var(--cyan2) !important;
  box-shadow:0 0 10px var(--cyan2) !important;
  transition:box-shadow 0.15s !important; will-change:box-shadow;
}
.stSlider [data-baseweb="slider"] div[role="slider"]:hover {
  box-shadow:0 0 18px var(--cyan2), 0 0 4px var(--cyan2) !important;
}
.stCheckbox label { color:var(--text2) !important; font-family:var(--font-mono) !important; font-size:10px !important; letter-spacing:0.1em !important; transition:color 0.15s !important; }
.stCheckbox label:hover { color:var(--cyan2) !important; }

/* ── DataFrames — aggressive dark override ── */
.stDataFrame,[data-testid="stDataFrame"] {
  background:var(--bg3) !important; border:1px solid var(--border) !important; border-radius:3px !important;
}
/* The iframe that contains the actual table */
[data-testid="stDataFrame"] iframe {
  filter: invert(1) hue-rotate(175deg) brightness(0.85) saturate(1.2) !important;
  border-radius:3px !important;
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
  transition:color .15s, background .15s !important;
}
[data-testid="stDataFrame"] tr:hover td {
  background:rgba(0,200,180,0.06) !important;
  color:var(--cyan2) !important;
}

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
  transition:background 0.2s; position:relative; overflow:hidden;
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
  transition:border-color .22s ease, background .22s ease,
             transform .18s cubic-bezier(0.34,1.56,0.64,1),
             box-shadow .22s ease; cursor:default;
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
  border-color:var(--border3); background:rgba(0,200,180,0.07);
  box-shadow:0 0 20px rgba(0,200,180,0.12) inset,
             0 4px 24px rgba(0,0,0,0.4),
             0 0 0 1px rgba(0,210,180,0.08);
  transform:translateY(-3px);
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

/* ── CDM items ── */
.cdm-item {
  padding:7px 9px; border-left:2px solid; margin-bottom:4px;
  cursor:pointer; background:rgba(0,0,0,0.18); transition:all 0.12s;
  border-radius:0 3px 3px 0;
}
.cdm-item:hover { background:rgba(0,200,180,0.05); transform:translateX(2px); }
.cdm-item.safe { border-left-color:var(--green); }
.cdm-item.safe:hover { box-shadow:0 0 14px rgba(0,255,136,0.08) inset; }
.cdm-item.warn { border-left-color:var(--yellow); }
.cdm-item.warn:hover { box-shadow:0 0 14px rgba(255,215,0,0.1) inset; }
.cdm-item.crit { border-left-color:var(--red); animation:cdmpulse 1.5s infinite; box-shadow:0 0 14px rgba(255,34,68,0.08) inset; }
.cdm-item.crit:hover { box-shadow:0 0 20px rgba(255,34,68,0.15) inset !important; border-left-color:var(--red2) !important; }
@keyframes cdmpulse { 0%,100%{background:rgba(255,34,68,0.04)} 50%{background:rgba(255,34,68,0.10)} }
.cdm-ids  { font-family:var(--font-mono); font-size:8px; color:var(--text2); margin-bottom:2px; }
.cdm-dist { font-family:var(--font-mono); font-size:11px; font-weight:700; }
.cdm-meta { display:flex; justify-content:space-between; align-items:center; margin-top:2px; }
.cdm-pc   { font-family:var(--font-mono); font-size:8px; color:var(--text2); }
.cdm-tca  { font-family:var(--font-mono); font-size:8px; color:var(--text3); }
.cdm-pruned { font-family:var(--font-mono); font-size:7px; color:var(--purple); margin-top:1px; }

/* ── Contact window ── */
.contact-win {
  padding:8px 10px; border:1px solid var(--border); border-radius:3px;
  margin-bottom:5px; background:rgba(0,0,0,0.2);
  transition:all 0.18s cubic-bezier(0.34,1.56,0.64,1);
}
.contact-win:hover {
  border-color:var(--border2); background:rgba(0,200,180,0.04);
  transform:translateY(-2px);
  box-shadow:0 6px 24px rgba(0,0,0,0.4), 0 0 14px rgba(0,200,180,0.1);
}
.contact-win.blackout { border-color:rgba(255,123,0,0.3); }
.contact-win.blackout:hover {
  border-color:rgba(255,123,0,0.55);
  box-shadow:0 6px 24px rgba(0,0,0,0.4), 0 0 14px rgba(255,123,0,0.12);
}
.cw-gs   { font-family:var(--font-mono); font-size:9px; color:var(--cyan); margin-bottom:3px; display:flex; align-items:center; gap:6px; }
.cw-time { font-family:var(--font-mono); font-size:8px; color:var(--text2); }
.cw-dur  { font-family:var(--font-mono); font-size:9px; color:var(--green); }
.cw-el   { font-family:var(--font-mono); font-size:8px; color:var(--text2); }
.cw-blackout-tag { font-family:var(--font-mono); font-size:7px; color:var(--orange); margin-top:3px; padding:2px 5px; border:1px solid rgba(255,123,0,0.3); border-radius:2px; display:inline-block; }

/* ── Uptime bars ── */
.uptime-bar-row { display:flex; align-items:center; gap:6px; margin-bottom:4px; transition:transform 0.15s; }
.uptime-bar-row:hover { transform:translateX(4px); }
.uptime-sat-id { font-family:var(--font-mono); font-size:8px; color:var(--text2); width:72px; flex-shrink:0; overflow:hidden; text-overflow:ellipsis; }
.uptime-bar-bg { flex:1; height:4px; background:rgba(255,255,255,0.06); border-radius:2px; overflow:hidden; }
.uptime-bar-fill { height:100%; border-radius:2px; transition:width 1s, box-shadow 0.2s; }
.uptime-bar-row:hover .uptime-bar-fill { box-shadow:0 0 8px currentColor; filter:brightness(1.15); }
.uptime-pct { font-family:var(--font-mono); font-size:8px; width:36px; text-align:right; flex-shrink:0; }
.uptime-fleet-card { background:rgba(0,200,180,0.06); border:1px solid var(--border2); border-radius:4px; padding:12px; margin-bottom:10px; text-align:center; }
.uptime-fleet-num { font-family:var(--font-display); font-size:24px; font-weight:700; color:var(--cyan2); text-shadow:var(--glow-cyan); }
.uptime-grade { font-family:var(--font-mono); font-size:9px; margin-top:3px; }
.grade-EXCELLENT { color:var(--green);  } .grade-GOOD { color:var(--cyan); }
.grade-ACCEPTABLE { color:var(--yellow); } .grade-POOR { color:var(--red); }

/* ── Telem grid ── */
.telem-grid { display:grid; grid-template-columns:1fr 1fr; gap:5px; margin-top:6px; }
.telem-val  { background:rgba(0,0,0,0.28); border:1px solid var(--border); border-radius:3px; padding:5px 8px; transition:border-color .18s, background .18s; position:relative; overflow:hidden; }
.telem-val::after {
  content:''; position:absolute; top:0; left:-100%; width:100%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(0,210,180,0.07),transparent);
  pointer-events:none;
}
.telem-val:hover { border-color:var(--border2); background:rgba(0,200,180,0.05); }
.telem-val:hover::after { animation:telemscani .5s ease-out forwards; }
.telem-val:hover .telem-num { text-shadow:var(--glow-cyan); }
@keyframes telemscani { from{left:-100%} to{left:100%} }
.telem-label { font-family:var(--font-mono); font-size:7px; color:var(--text2); letter-spacing:0.08em; margin-bottom:2px; text-transform:uppercase; }
.telem-num   { font-family:var(--font-mono); font-size:12px; color:var(--cyan2); }
.telem-num.warn   { color:var(--yellow); } .telem-num.crit { color:var(--red); text-shadow:var(--glow-red); }
.telem-num.green  { color:var(--green);  } .telem-num.purple { color:var(--purple); }

/* ── GS items & cards ── */
.gs-item { padding:6px 10px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(0,200,180,0.04); transition:background 0.1s, padding-left 0.15s; }
.gs-item:hover { background:rgba(0,200,180,0.04); padding-left:14px; }
.gs-item:hover .gs-name { color:var(--cyan); }
.gs-card {
  background:linear-gradient(135deg,var(--bg3),var(--bg2)); border:1px solid var(--border);
  border-radius:3px; padding:12px 14px; margin-bottom:8px;
  position:relative; overflow:hidden;
  transition:border-color .18s, box-shadow .18s, transform .18s cubic-bezier(0.34,1.56,0.64,1); cursor:default;
}
.gs-card:hover {
  border-color:var(--border2);
  box-shadow:0 0 22px rgba(0,200,180,0.14), 0 4px 20px rgba(0,0,0,0.4);
  transform:translateY(-2px);
}
.gs-card.active::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,var(--green),transparent); }

/* ── Rubric boxes ── */
.rubric-box { text-align:center; padding:10px 8px; border-radius:3px; border:1px solid transparent; transition:border-color .2s, box-shadow .2s, transform .18s cubic-bezier(0.34,1.56,0.64,1); cursor:default; }
.rubric-box:hover { transform:translateY(-3px); }
.rubric-excellent { background:rgba(0,255,136,0.05); border-color:rgba(0,255,136,0.12); }
.rubric-excellent:hover { border-color:rgba(0,255,136,0.4); box-shadow:0 4px 20px rgba(0,255,136,0.15); }
.rubric-good  { background:rgba(0,200,180,0.05); border-color:rgba(0,200,180,0.12); }
.rubric-good:hover  { border-color:rgba(0,200,180,0.4); box-shadow:0 4px 20px rgba(0,200,180,0.15); }
.rubric-ok    { background:rgba(255,215,0,0.05);  border-color:rgba(255,215,0,0.12); }
.rubric-ok:hover    { border-color:rgba(255,215,0,0.4);  box-shadow:0 4px 20px rgba(255,215,0,0.12); }
.rubric-poor  { background:rgba(255,34,68,0.05);  border-color:rgba(255,34,68,0.12); }
.rubric-poor:hover  { border-color:rgba(255,34,68,0.4);  box-shadow:0 4px 20px rgba(255,34,68,0.12); }

/* ── Sidebar logo ── */
.tb-logo {
  font-family:var(--font-display); font-size:20px; font-weight:900;
  color:var(--cyan2); letter-spacing:0.15em;
  text-shadow:var(--glow-cyan); margin-bottom:2px;
  animation:logoPulse 4s ease-in-out infinite alternate;
}
@keyframes logoPulse {
  from { text-shadow:0 0 10px rgba(0,232,208,0.4); }
  to   { text-shadow:0 0 22px rgba(0,232,208,0.8), 0 0 40px rgba(0,232,208,0.2); }
}
.tb-sub  { font-family:var(--font-mono); font-size:9px; color:var(--text2); letter-spacing:0.2em; margin-bottom:14px; }
.sidebar-uptime {
  background:rgba(0,200,180,0.06); border:1px solid var(--border2); border-radius:3px;
  padding:10px 12px; margin-bottom:8px;
  transition:border-color .2s, box-shadow .2s;
}
.sidebar-uptime:hover {
  border-color:var(--border3);
  box-shadow:0 0 16px rgba(0,200,180,0.18);
}

/* ── Animations ── */
@keyframes blink   { 0%,100%{opacity:1} 50%{opacity:0.3} }
@keyframes pulse   { 0%,100%{opacity:1} 50%{opacity:0.35} }
.live-dot { animation:liveDotPulse 1.8s ease-in-out infinite; }
@keyframes liveDotPulse {
  0%,100% { opacity:1; text-shadow:0 0 4px rgba(0,255,136,0.6); }
  50%     { opacity:0.3; text-shadow:none; }
}
.crit-blink { animation:blink 0.8s infinite; }

/* ── stMetric dark theme ── */
div[data-testid="metric-container"],
div[data-testid="stMetric"] {
  background:var(--bg3) !important; border:1px solid var(--border) !important;
  border-radius:3px !important; padding:10px 12px !important;
  transition:border-color .2s, box-shadow .2s, transform .18s !important;
}
div[data-testid="metric-container"]:hover,
div[data-testid="stMetric"]:hover {
  border-color:var(--border3) !important;
  box-shadow:0 0 14px rgba(0,210,180,0.1) inset !important;
  transform:translateY(-1px);
}
div[data-testid="stMetricLabel"] > div,
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] label {
  font-family:var(--font-mono) !important; font-size:8px !important;
  letter-spacing:0.12em !important; color:var(--text2) !important; text-transform:uppercase !important;
}
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] > div {
  font-family:var(--font-display) !important; color:var(--cyan2) !important; font-size:20px !important;
}
div[data-testid="stMetricDelta"] svg { display:none !important; }
div[data-testid="stMetricDelta"] > div {
  color:var(--text2) !important; font-family:var(--font-mono) !important; font-size:8px !important;
}

/* ── stTabs dark theme ── */
div[data-testid="stTabs"] button[role="tab"] {
  font-family:var(--font-mono) !important; font-size:9px !important;
  letter-spacing:0.12em !important; color:var(--text2) !important;
  background:transparent !important; border:none !important;
  border-bottom:2px solid transparent !important; border-radius:0 !important;
  padding:6px 14px !important;
  transition:color 0.15s, border-color 0.15s, text-shadow 0.15s !important;
  text-transform:uppercase !important;
}
div[data-testid="stTabs"] button[role="tab"]:hover {
  color:var(--cyan2) !important; border-bottom-color:var(--border2) !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color:var(--cyan2) !important; border-bottom:2px solid var(--cyan2) !important;
  text-shadow:var(--glow-cyan) !important;
}
div[data-testid="stTabs"] > div[role="tabpanel"] {
  border-top:1px solid var(--border) !important; background:transparent !important; padding-top:12px !important;
}
div[data-testid="stTabs"] > div[role="tablist"] {
  background:var(--bg2) !important; border-bottom:1px solid var(--border) !important; gap:0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:var(--bg2); }
::-webkit-scrollbar-thumb { background:rgba(0,210,180,0.25); border-radius:2px; }
::-webkit-scrollbar-thumb:hover { background:rgba(0,210,180,0.5); }

/* ── Hide ALL Streamlit loading/running indicators ── */
[data-testid="stSpinner"],
[data-testid="stSpinner"] > div,
.stSpinner, .stSpinner > div,
div[data-testid="stSpinner"],
/* Running circle top-left */
[data-testid="stStatusWidget"],
.stStatusWidget,
/* Page running indicator (the grey circle) */
[data-testid="stApp"] > div:first-child > div:first-child > div[style*="position: fixed"],
iframe[title="st_connection_status"],
/* Streamlit running animation overlay */
.stApp > div > div[data-testid="stDecoration"],
[class*="StatusWidget"],
[class*="stSpinner"] {
  display:none !important;
  opacity:0 !important;
  pointer-events:none !important;
}
/* Nuclear option: hide any fixed-position circles in top-left */
body > div[style*="position: fixed"][style*="top: 0"],
.main > div > div > div[style*="position: fixed"] {
  display:none !important;
}

/* ── Progress bar — match theme ── */
[data-testid="stProgress"] > div {
  background:rgba(0,210,180,0.08) !important;
  border-radius:2px !important;
}
[data-testid="stProgress"] > div > div {
  background:linear-gradient(90deg,var(--cyan),var(--cyan2)) !important;
  border-radius:2px !important;
  box-shadow:0 0 8px rgba(0,210,180,0.4) !important;
  transition:width 0.4s ease !important;
}

/* ── Altair / Vega chart dark theme ── */
.vega-embed {
  background:transparent !important;
}
.vega-embed summary { display:none !important; }
canvas.marks { background:transparent !important; }

/* ── Global smoothing ── */
*, *::before, *::after { -webkit-font-smoothing:antialiased; }
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

# ── Star background canvas + cursor ring + tooltip (shown after login) ────────
st.markdown("""
<!-- ═══ ANIMATED STAR BACKGROUND (identical to login page) ═══ -->
<canvas id="oi-stars" style="
  position:fixed;inset:0;width:100%;height:100%;
  pointer-events:none;z-index:0;opacity:0.55;
"></canvas>

<!-- ═══ CURSOR RING ═══ -->
<div id="oi-cur"></div>
<div id="oi-tip"></div>

<!-- ═══ AMBIENT PULSE RINGS (corner decoration) ═══ -->
<div style="
  position:fixed;bottom:40px;right:40px;
  width:180px;height:180px;border-radius:50%;
  border:1px solid rgba(0,210,180,0.06);
  animation:ambientSpin 28s linear infinite;
  pointer-events:none;z-index:0;
">
  <div style="
    position:absolute;top:50%;left:50%;
    transform:translate(-50%,-50%);
    width:110px;height:110px;border-radius:50%;
    border:1px solid rgba(0,210,180,0.04);
    animation:ambientSpin 16s linear infinite reverse;
  "></div>
</div>

<style>
/* ── Star canvas z-ordering ── */
.stApp { isolation:isolate; }
.main .block-container { position:relative; z-index:1; }
[data-testid="stSidebar"] { z-index:10 !important; }

/* ── Ambient ring spin ── */
@keyframes ambientSpin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }

/* ════════════════════════════════════════════════════════
   PAGE ENTRY ANIMATION — content fades up on nav change
════════════════════════════════════════════════════════ */
.main .block-container {
  animation:pageEnter 0.35s ease-out both;
}
@keyframes pageEnter {
  from { opacity:0; transform:translateY(8px); }
  to   { opacity:1; transform:translateY(0); }
}

/* ════════════════════════════════════════════════════════
   SIDEBAR SCAN LINE — horizontal sweep every 6s
════════════════════════════════════════════════════════ */
.sidebar-scan {
  position:absolute; left:0; right:0; height:1px; pointer-events:none;
  background:linear-gradient(90deg,
    transparent 0%, rgba(0,210,180,0.5) 40%,
    rgba(0,232,208,0.9) 50%, rgba(0,210,180,0.5) 60%, transparent 100%);
  box-shadow:0 0 6px rgba(0,210,180,0.4);
  animation:sidebarScan 6s linear infinite;
}
@keyframes sidebarScan {
  0%   { top:-2px; opacity:0; }
  3%   { opacity:1; }
  97%  { opacity:1; }
  100% { top:100%; opacity:0; }
}

/* ════════════════════════════════════════════════════════
   TOP AREA GLOW BAR — cyan line under the topmost element
════════════════════════════════════════════════════════ */
.main .block-container::before {
  content:''; display:block; height:1px; width:100%;
  background:linear-gradient(90deg,
    transparent 0%, rgba(0,210,180,0.15) 20%,
    rgba(0,210,180,0.5) 50%, rgba(0,210,180,0.15) 80%, transparent 100%);
  margin-bottom:8px;
  box-shadow:0 0 8px rgba(0,210,180,0.15);
}

/* ── Cursor ring — identical to index.html ── */
#oi-cur {
  position:fixed; pointer-events:none; z-index:99999;
  width:24px; height:24px; border-radius:50%;
  border:1px solid rgba(0,210,180,0.5);
  transform:translate(-50%,-50%);
  transition:width .18s ease,height .18s ease,border-color .18s ease,
             top .18s ease,left .18s ease,opacity .18s ease,
             background .15s ease;
  mix-blend-mode:screen;
}
#oi-cur.h {
  width:38px; height:38px;
  border-color:var(--cyan2);
  box-shadow:0 0 12px rgba(0,210,180,0.35);
  background:rgba(0,210,180,0.04);
}
#oi-cur.c {
  width:16px; height:16px;
  background:rgba(0,210,180,0.15);
  border-color:var(--cyan3);
  box-shadow:0 0 18px rgba(0,210,180,0.5);
}

/* ── Tooltip ── */
#oi-tip {
  position:fixed; pointer-events:none; z-index:99998;
  font-family:var(--font-mono); font-size:9px;
  color:var(--cyan2); background:rgba(3,11,16,0.95);
  border:1px solid var(--border2); border-radius:3px;
  padding:5px 9px; letter-spacing:0.06em; line-height:1.6;
  opacity:0; transition:opacity .15s; max-width:240px;
  white-space:pre-line;
  box-shadow:0 4px 20px rgba(0,0,0,0.5),0 0 10px rgba(0,210,180,0.08);
}

/* ════════════════════════════════════════════════════════
   ENHANCED METRIC CARDS — glow sweep + lift
════════════════════════════════════════════════════════ */
.metric-card {
  transition:
    border-color .22s ease,
    background   .22s ease,
    transform    .18s cubic-bezier(0.34,1.56,0.64,1),
    box-shadow   .22s ease !important;
}
.metric-card:hover {
  border-color:var(--border3) !important;
  background:rgba(0,200,180,0.07) !important;
  box-shadow:
    0 0 20px rgba(0,200,180,0.12) inset,
    0 4px 24px rgba(0,0,0,0.4),
    0 0 0 1px rgba(0,210,180,0.08) !important;
  transform:translateY(-3px) !important;
}

/* ════════════════════════════════════════════════════════
   CDM ITEMS — stronger glow on crit pulse
════════════════════════════════════════════════════════ */
.cdm-item.crit {
  animation:cdmpulse 1.5s infinite;
  box-shadow:0 0 14px rgba(255,34,68,0.08) inset;
}
.cdm-item.crit:hover {
  box-shadow:0 0 20px rgba(255,34,68,0.15) inset !important;
  border-left-color:var(--red2) !important;
}
.cdm-item.warn:hover {
  box-shadow:0 0 14px rgba(255,215,0,0.1) inset !important;
}
.cdm-item.safe:hover {
  box-shadow:0 0 14px rgba(0,255,136,0.08) inset !important;
}

/* ════════════════════════════════════════════════════════
   PANEL HEADERS — animated left-edge sweep on hover
════════════════════════════════════════════════════════ */
.panel-hdr {
  position:relative; overflow:hidden;
}
.panel-hdr::after {
  content:''; position:absolute; left:-100%; top:0; bottom:0; width:100%;
  background:linear-gradient(90deg,transparent,rgba(0,210,180,0.04),transparent);
  transition:none; pointer-events:none;
}
.panel-hdr:hover::after { animation:panelSweep .5s ease-out forwards; }
@keyframes panelSweep { from{left:-100%} to{left:100%} }

/* ════════════════════════════════════════════════════════
   SIDEBAR NAV — left accent bar slides in on hover
════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] .stRadio label {
  position:relative; overflow:hidden;
}
[data-testid="stSidebar"] .stRadio label::before {
  content:''; position:absolute; left:0; top:0; bottom:0; width:2px;
  background:var(--cyan2); transform:scaleY(0); transform-origin:center;
  transition:transform .2s ease; box-shadow:var(--glow-cyan);
}
[data-testid="stSidebar"] .stRadio label:hover::before { transform:scaleY(1); }

/* ════════════════════════════════════════════════════════
   CONTACT WINDOWS — lift + coloured glow on hover
════════════════════════════════════════════════════════ */
.contact-win:hover {
  transform:translateY(-2px) !important;
  box-shadow:0 6px 24px rgba(0,0,0,0.4), 0 0 14px rgba(0,200,180,0.1) !important;
}
.contact-win.blackout:hover {
  box-shadow:0 6px 24px rgba(0,0,0,0.4), 0 0 14px rgba(255,123,0,0.12) !important;
}

/* ════════════════════════════════════════════════════════
   GS CARDS — glow intensity upgrade
════════════════════════════════════════════════════════ */
.gs-card:hover {
  box-shadow:0 0 22px rgba(0,200,180,0.14), 0 4px 20px rgba(0,0,0,0.4) !important;
  transform:translateY(-2px) !important;
}

/* ════════════════════════════════════════════════════════
   UPTIME BARS — fill glow on row hover
════════════════════════════════════════════════════════ */
.uptime-bar-row:hover { transform:translateX(4px) !important; }
.uptime-bar-row:hover .uptime-bar-fill {
  box-shadow:0 0 8px currentColor !important;
  filter:brightness(1.15);
}

/* ════════════════════════════════════════════════════════
   RUBRIC BOXES — stronger border glow
════════════════════════════════════════════════════════ */
.rubric-excellent:hover { box-shadow:0 4px 20px rgba(0,255,136,0.15) !important; }
.rubric-good:hover      { box-shadow:0 4px 20px rgba(0,200,180,0.15) !important; }
.rubric-ok:hover        { box-shadow:0 4px 20px rgba(255,215,0,0.12) !important; }
.rubric-poor:hover      { box-shadow:0 4px 20px rgba(255,34,68,0.12) !important; }

/* ════════════════════════════════════════════════════════
   STREAMLIT METRICS — glow on hover
════════════════════════════════════════════════════════ */
div[data-testid="metric-container"]:hover {
  border-color:var(--border3) !important;
  box-shadow:0 0 14px rgba(0,210,180,0.1) inset !important;
  transform:translateY(-1px);
}

/* ════════════════════════════════════════════════════════
   DATAFRAME ROWS — stronger row hover
════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] tr:hover td {
  background:rgba(0,200,180,0.06) !important;
  color:var(--cyan2) !important;
}

/* ════════════════════════════════════════════════════════
   TELEM VALUES — scan sweep on hover
════════════════════════════════════════════════════════ */
.telem-val { position:relative; overflow:hidden; }
.telem-val::after {
  content:''; position:absolute; top:0; left:-100%; width:100%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(0,210,180,0.07),transparent);
  pointer-events:none;
}
.telem-val:hover::after { animation:telemscani .5s ease-out forwards; }
.telem-val:hover .telem-num { text-shadow:var(--glow-cyan); }
@keyframes telemscani { from{left:-100%} to{left:100%} }

/* ════════════════════════════════════════════════════════
   BUTTON UPGRADE — ripple + stronger glow
════════════════════════════════════════════════════════ */
.stButton > button:hover {
  box-shadow:0 0 24px rgba(0,200,180,0.28),
             0 4px 16px rgba(0,0,0,0.4) !important;
}
.stButton > button:active {
  box-shadow:0 0 12px rgba(0,200,180,0.4) !important;
}

/* ════════════════════════════════════════════════════════
   SIDEBAR LOGO GLOW PULSE
════════════════════════════════════════════════════════ */
.tb-logo {
  animation:logoPulse 4s ease-in-out infinite alternate;
}
@keyframes logoPulse {
  from { text-shadow:0 0 10px rgba(0,232,208,0.4); }
  to   { text-shadow:0 0 22px rgba(0,232,208,0.8), 0 0 40px rgba(0,232,208,0.2); }
}

/* ════════════════════════════════════════════════════════
   BADGES — hover glow by colour
════════════════════════════════════════════════════════ */
.badge-cyan:hover   { box-shadow:0 0 10px rgba(0,232,208,0.4); }
.badge-green:hover  { box-shadow:0 0 10px rgba(0,255,136,0.35); }
.badge-red:hover    { box-shadow:0 0 10px rgba(255,34,68,0.4); }
.badge-yellow:hover { box-shadow:0 0 10px rgba(255,215,0,0.3); }
.badge-purple:hover { box-shadow:0 0 10px rgba(170,68,255,0.35); }
.badge-blue:hover   { box-shadow:0 0 10px rgba(0,180,255,0.3); }

/* ════════════════════════════════════════════════════════
   LIVE DOT — pulsing dot upgrade
════════════════════════════════════════════════════════ */
.live-dot {
  animation:liveDotPulse 1.8s ease-in-out infinite !important;
}
@keyframes liveDotPulse {
  0%,100% { opacity:1; text-shadow:0 0 4px rgba(0,255,136,0.6); }
  50%     { opacity:0.3; text-shadow:none; }
}

/* ════════════════════════════════════════════════════════
   SIDEBAR UPTIME CARD — active glow
════════════════════════════════════════════════════════ */
.sidebar-uptime:hover {
  box-shadow:0 0 16px rgba(0,200,180,0.18) !important;
  border-color:var(--border3) !important;
}

/* ════════════════════════════════════════════════════════
   SCROLLBAR STYLING — matches index.html dark aesthetic
════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:var(--bg2); }
::-webkit-scrollbar-thumb {
  background:rgba(0,210,180,0.25); border-radius:2px;
}
::-webkit-scrollbar-thumb:hover { background:rgba(0,210,180,0.5); }
</style>

<script>
(function(){
  // ═══════════════════════════════════════════════════════════════
  //  ANIMATED STAR BACKGROUND — same engine as login page
  // ═══════════════════════════════════════════════════════════════
  function initStars(){
    var sc = document.getElementById('oi-stars');
    if(!sc) return;
    var sctx = sc.getContext('2d');
    function resize(){ sc.width=window.innerWidth; sc.height=window.innerHeight; }
    resize();
    window.addEventListener('resize', resize);

    var stars = [];
    for(var i=0;i<220;i++){
      stars.push({
        x: Math.random()*sc.width,
        y: Math.random()*sc.height,
        r: Math.random()*1.3+0.15,
        speed: Math.random()*0.08+0.015,
        twinkle: Math.random()*Math.PI*2,
        twinkleSpeed: Math.random()*0.018+0.008
      });
    }

    // shooting star state
    var shoot = null, shootTimer = 3000;

    function spawnShoot(){
      shoot = {
        x: Math.random()*sc.width*0.7,
        y: Math.random()*sc.height*0.35,
        len: 70+Math.random()*70,
        speed: 4+Math.random()*5,
        angle: Math.PI/6,
        life: 1.0
      };
    }

    var _vigCache = null, _vigW = 0, _vigH = 0;

    function frame(){
      if(sc.width!==window.innerWidth||sc.height!==window.innerHeight){
        sc.width=window.innerWidth; sc.height=window.innerHeight;
        _vigCache=null;
      }
      sctx.clearRect(0,0,sc.width,sc.height);

      // vignette — cached
      if(!_vigCache || _vigW!==sc.width || _vigH!==sc.height){
        _vigW=sc.width; _vigH=sc.height;
        _vigCache = sctx.createRadialGradient(
          sc.width/2,sc.height/2,0,
          sc.width/2,sc.height/2,sc.width*0.65
        );
        _vigCache.addColorStop(0,'rgba(0,15,30,0.25)');
        _vigCache.addColorStop(1,'rgba(0,0,0,0.55)');
      }
      sctx.fillStyle=_vigCache; sctx.fillRect(0,0,sc.width,sc.height);

      // stars
      for(var i=0;i<stars.length;i++){
        var s=stars[i];
        s.y -= s.speed;
        s.twinkle += s.twinkleSpeed;
        if(s.y < -2){ s.y=sc.height+2; s.x=Math.random()*sc.width; }
        var alpha = 0.25 + 0.45*Math.abs(Math.sin(s.twinkle));
        sctx.fillStyle='rgba(180,235,255,'+alpha.toFixed(2)+')';
        sctx.beginPath();
        sctx.arc(s.x,s.y,s.r,0,Math.PI*2);
        sctx.fill();
      }

      // shooting star
      shootTimer -= 16;
      if(shootTimer<=0){ spawnShoot(); shootTimer=5000+Math.random()*6000; }
      if(shoot){
        var dx=Math.cos(shoot.angle)*shoot.speed;
        var dy=Math.sin(shoot.angle)*shoot.speed;
        var tx=Math.cos(shoot.angle)*shoot.len;
        var ty=Math.sin(shoot.angle)*shoot.len;
        shoot.x+=dx; shoot.y+=dy; shoot.life-=0.018;
        if(shoot.life>0 && shoot.x<sc.width && shoot.y<sc.height){
          var g=sctx.createLinearGradient(shoot.x,shoot.y,shoot.x-tx,shoot.y-ty);
          g.addColorStop(0,'rgba(0,235,215,'+shoot.life.toFixed(2)+')');
          g.addColorStop(1,'transparent');
          sctx.strokeStyle=g; sctx.lineWidth=1.5;
          sctx.beginPath(); sctx.moveTo(shoot.x,shoot.y);
          sctx.lineTo(shoot.x-tx,shoot.y-ty); sctx.stroke();
        } else { shoot=null; }
      }

      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  // ═══════════════════════════════════════════════════════════════
  //  CURSOR RING + TOOLTIP
  // ═══════════════════════════════════════════════════════════════
  function initCursor(){
    var ring=document.getElementById('oi-cur');
    var tip=document.getElementById('oi-tip');
    if(ring){
      document.addEventListener('mousemove',function(e){
        ring.style.left=e.clientX+'px';
        ring.style.top=e.clientY+'px';
      });
      document.addEventListener('mousedown',function(){ ring.classList.add('c'); });
      document.addEventListener('mouseup',function(){   ring.classList.remove('c'); });
      document.addEventListener('mouseover',function(e){
        if(e.target.closest(
          'button,a,.metric-card,.cdm-item,.contact-win,.gs-card,.gs-item,' +
          '.sat-row,.rubric-box,.panel-badge,.uptime-bar-row,[data-tip]'
        )) ring.classList.add('h');
        else ring.classList.remove('h');
      });
    }
    if(tip){
      document.addEventListener('mousemove',function(e){
        tip.style.left=(e.clientX+18)+'px';
        tip.style.top=(e.clientY-10)+'px';
      });
      document.addEventListener('mouseover',function(e){
        var el=e.target.closest('[data-tip]');
        if(el){ tip.innerHTML=el.getAttribute('data-tip').replace(/&#10;/g,'<br>'); tip.style.opacity='1'; }
      });
      document.addEventListener('mouseout',function(e){
        if(e.target.closest('[data-tip]')) tip.style.opacity='0';
      });
    }
  }

  // Wait for DOM then init both
  if(document.readyState==='loading'){
    document.addEventListener('DOMContentLoaded',function(){ initStars(); initCursor(); hideSpinners(); });
  } else {
    initStars(); initCursor(); hideSpinners();
  }

  // ═══════════════════════════════════════════════════════════════
  //  HIDE STREAMLIT RUNNING CIRCLE — MutationObserver approach
  // ═══════════════════════════════════════════════════════════════
  function hideSpinners(){
    var sel = [
      '[data-testid="stSpinner"]',
      '[data-testid="stStatusWidget"]',
      '.stSpinner',
      '[class*="StatusWidget"]',
      '[class*="stSpinner"]',
    ].join(',');
    function nuke(){
      document.querySelectorAll(sel).forEach(function(el){
        el.style.display='none'; el.style.opacity='0';
      });
    }
    nuke();
    var mo = new MutationObserver(nuke);
    mo.observe(document.body, {childList:true, subtree:true});
  }
})();
</script>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  API HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=2)
def get_status():
    try: return requests.get(f"{API}/status", timeout=10).json()
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

@st.cache_data(ttl=5)
def get_events(limit=200):
    try: return requests.get(f"{API}/events", timeout=3).json()
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

def api_online() -> bool:
    """Reuses the cached get_status() result — no extra HTTP round-trip."""
    return bool(get_status())

# ── Component helpers ──────────────────────────────────────────────────────────
def ph(title, badge="", badge_cls="badge-cyan"):
    """panel_header — exact index.html .panel-hdr with corner bracket"""
    b = f'<span class="panel-badge {badge_cls}">{badge}</span>' if badge else ""
    return (
        f'<div class="panel-hdr">'
        f'<div class="panel-title">{title}</div>'
        f'<div style="display:flex;align-items:center;gap:6px">{b}'
        f'<span style="font-family:var(--font-mono);font-size:8px;'
        f'color:var(--text3);letter-spacing:0.04em">[ ]</span>'
        f'</div></div>'
    )

def mc(label, val, color="", delta="", tip=""):
    """metric_card with scan-sweep glow, corner accent, and optional tooltip"""
    tip_attr = f' data-tip="{tip}"' if tip else ""
    dh = f'<div class="metric-delta">{delta}</div>' if delta else ""
    glow_color = {
        "red":"rgba(255,34,68,0.3)", "green":"rgba(0,255,136,0.25)",
        "cyan":"rgba(0,210,180,0.3)", "yellow":"rgba(255,215,0,0.25)",
        "purple":"rgba(170,68,255,0.25)", "blue":"rgba(0,180,255,0.25)",
        "orange":"rgba(255,123,0,0.25)"
    }.get(color, "rgba(0,210,180,0.15)")
    accent = {
        "red":"var(--red)", "green":"var(--green)", "cyan":"var(--cyan2)",
        "yellow":"var(--yellow)", "purple":"var(--purple)",
        "blue":"var(--blue2)", "orange":"var(--orange)"
    }.get(color, "var(--cyan)")
    return (
        f'<div class="metric-card" {tip_attr} '
        f'style="--mc-accent:{accent};--mc-glow:{glow_color}">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:1px;'
        f'background:linear-gradient(90deg,transparent,{accent},transparent);opacity:0.5"></div>'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {color}">{val}</div>'
        f'{dh}</div>'
    )

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

def dark_table(df, max_rows=50):
    """Render a pandas DataFrame as a fully dark-themed HTML table matching index.html."""
    df2 = df.head(max_rows).reset_index(drop=True)
    cols = list(df2.columns)
    hdr = "".join(
        f'<th style="font-family:var(--font-mono);font-size:8px;letter-spacing:0.12em;'
        f'color:var(--cyan);background:rgba(0,200,180,0.06);padding:6px 10px;'
        f'border-bottom:1px solid var(--border2);text-transform:uppercase;white-space:nowrap">{c}</th>'
        for c in cols)
    rows_html = ""
    for _, row in df2.iterrows():
        cells = ""
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                txt = f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
            else:
                txt = str(v)
            # colour-code known status/risk values
            clr = "var(--text)"
            if txt in ("NOMINAL","✓","LOW","SAFE"): clr = "var(--green)"
            elif txt in ("MANEUVERING","OUT_OF_SLOT","WARN","WARNING","YELLOW"): clr = "var(--yellow)"
            elif txt in ("EOL","CRITICAL","RED","✗","HIGH"): clr = "var(--red)"
            elif txt.startswith("SAT-") or txt.startswith("A-"): clr = "var(--cyan2)"
            elif txt.startswith("DEB-") or txt.startswith("GS-"): clr = "var(--text2)"
            cells += (
                f'<td style="font-family:var(--font-mono);font-size:9px;color:{clr};'
                f'padding:5px 10px;border-bottom:1px solid var(--border)">{txt}</td>'
            )
        rows_html += (
            f'<tr onmouseover="this.style.background=\'rgba(0,200,180,0.04)\'"'
            f' onmouseout="this.style.background=\'\'"> {cells}</tr>'
        )
    return (
        f'<div style="overflow-x:auto;border:1px solid var(--border);'
        f'border-radius:3px;margin-top:6px">'
        f'<table style="width:100%;border-collapse:collapse;background:var(--bg3)">'
        f'<thead><tr>{hdr}</tr></thead><tbody>{rows_html}</tbody></table></div>'
    )

def altair_chart(df, x_field, y_field, color_field=None, color_scale=None, height=300, tooltips=None):
    """
    Safe Altair bar chart helper.
    Accepts either a shorthand string or a pre-built alt.X / alt.Y object for
    x_field / y_field — does NOT double-wrap existing encoding objects.
    configure_axis() is intentionally omitted: when callers pass alt.Y objects
    with their own axis= config, a top-level configure_axis() silently overrides
    those per-field settings and causes a SchemaValidationError in Altair v5.
    Grid styling is handled via the shared axis= specs on each encoding instead.
    """
    # y encoding — pass through if already an alt.Y, otherwise build one
    if isinstance(y_field, alt.Y):
        y_enc = y_field
    else:
        y_enc = alt.Y(y_field, axis=alt.Axis(
            labelColor="rgba(180,230,220,0.5)",
            labelFont="Share Tech Mono",
            labelFontSize=9,
            gridColor="rgba(0,210,180,0.07)",
            domainColor="rgba(0,210,180,0.15)",
        ))

    # x encoding — pass through if already an alt.X, otherwise build one
    if isinstance(x_field, alt.X):
        x_enc = x_field
    else:
        x_enc = alt.X(x_field, axis=alt.Axis(
            labelColor="rgba(180,230,220,0.5)",
            labelFont="Share Tech Mono",
            labelFontSize=9,
            gridColor="rgba(0,210,180,0.07)",
            domainColor="rgba(0,210,180,0.15)",
        ))

    enc = dict(x=x_enc, y=y_enc)
    if color_field and color_scale:
        enc["color"] = alt.Color(color_field, scale=color_scale, legend=None)
    if tooltips:
        enc["tooltip"] = tooltips

    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=2, cornerRadiusBottomRight=2)
        .encode(**enc)
        .properties(height=height, background="transparent")
        .configure_view(strokeOpacity=0)
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

# ── Topbar (shown after login) ────────────────────────────────────────────────
_tb_status = get_status() if True else {}
_tb_sim_t  = int(_tb_status.get("sim_time", 0))
_tb_nom    = _tb_status.get("satellites_nominal", 0)
_tb_cdm    = _tb_status.get("active_conjunctions", 0)
_tb_mans   = _tb_status.get("maneuvers_executed", 0)
_tb_sp     = _tb_status.get("spatial_index", "KDTree").upper()
_tb_online = bool(_tb_status)
_tb_up     = get_fleet_uptime() if _tb_online else {}
_tb_fp     = _tb_up.get("fleet_uptime_pct", 0)
_tb_grade  = _tb_up.get("grade", "—")
_tb_gcol   = {"EXCELLENT":"green","GOOD":"cyan","ACCEPTABLE":"yellow","POOR":"red"}.get(_tb_grade,"")
_tb_met    = get_metrics() if _tb_online else {}
_tb_ms     = _tb_met.get("step_ms_avg", 0)

st.markdown(f'''
<div class="oi-topbar">
  <div class="oi-logo">ORBITAL <span>INSIGHT</span></div>
  <div class="oi-tb-divider"></div>
  <div class="oi-tb-item">
    <span>SIM TIME</span>
    <span class="val cyan">T+{_tb_sim_t//3600:04d}H {(_tb_sim_t%3600)//60:02d}M</span>
  </div>
  <div class="oi-tb-divider"></div>
  <div class="oi-tb-item">
    <span>FLEET</span>
    <span class="val">{_tb_nom}/55</span>
  </div>
  <div class="oi-tb-item">
    <span>CDM ACTIVE</span>
    <span class="val {'red' if _tb_cdm else 'green'}">{_tb_cdm}</span>
  </div>
  <div class="oi-tb-item">
    <span>UPTIME</span>
    <span class="val {_tb_gcol}">{_tb_fp:.1f}%</span>
  </div>
  <div class="oi-tb-item">
    <span>MANEUVERS</span>
    <span class="val cyan">{_tb_mans}</span>
  </div>
  <div class="oi-tb-item">
    <span>STEP AVG</span>
    <span class="val">{_tb_ms:.0f}ms</span>
  </div>
  <div class="oi-tb-divider"></div>
  <div class="oi-tb-item">
    <span class="val" style="font-size:9px;color:rgba(170,68,255,0.9)">{_tb_sp}</span>
  </div>
  <div class="oi-tb-spacer"></div>
  <div class="oi-tb-item" style="flex-direction:row;align-items:center">
    <span class="oi-live-dot"></span>
    <span class="val {'green' if _tb_online else 'red'}" style="font-size:9px">
      {'LIVE' if _tb_online else 'OFFLINE'}
    </span>
  </div>
</div>
''', unsafe_allow_html=True)


with st.sidebar:
    status      = get_status()   # single cached call — api_online() reuses this
    online      = bool(status)
    uptime_q    = get_fleet_uptime() if online else {}
    metrics     = get_metrics()     if online else {}

    st.markdown(f'''
    <div class="tb-logo">🛰 ORBITAL<span style="color:var(--muted);font-weight:400"> INSIGHT</span></div>
    <div class="tb-sub">NSH 2026 · ACM v7.4 · TEAM BROCODE</div>
    <div class="sidebar-scan"></div>
    ''', unsafe_allow_html=True)

    if online:
        spatial = status.get("spatial_index", "?")
        sim_t_s = int(status.get("sim_time", 0))
        st.markdown(
            f'{badge("⬤ LIVE","green")} {badge(spatial.upper(),"purple")} {badge(f"T+{sim_t_s//3600:03d}H","cyan")}',
            unsafe_allow_html=True)
    else:
        # Check if backend is starting up (not yet ready) vs truly offline
        try:
            ready_r = requests.get(f"{API}/ready", timeout=10).json()
            if ready_r.get("stage") == "warming_up":
                st.markdown(badge("⬤ STARTING","orange"), unsafe_allow_html=True)
                st.info("⏳ Backend is warming up (training anomaly detector + pre-warming contact schedules). This takes ~15s on first start. The page will refresh automatically.")
                import time as _time; _time.sleep(2); st.rerun()
            else:
                st.markdown(badge("⬤ OFFLINE","red"), unsafe_allow_html=True)
                st.warning("Backend not reachable — is the Docker container running?")
        except Exception:
            st.markdown(badge("⬤ OFFLINE","red"), unsafe_allow_html=True)
            st.warning("Backend not reachable — is the Docker container running?")

    st.markdown(ph("NAVIGATION"), unsafe_allow_html=True)
    page = st.radio("Navigation", [
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
            st.session_state["sim_step_target"] = r.get("sim_time", 0)
            st.session_state["sim_step_start"]  = r.get("sim_time", 0) - (step_hrs * 3600)
            st.session_state["sim_step_hrs"]    = step_hrs
            st.rerun()
        except Exception as e:
            st.error(f"Step failed: {e}")

    # Show live progress if a step is in progress
    if "sim_step_target" in st.session_state and online:
        try:
            target_t      = st.session_state["sim_step_target"]
            start_t       = st.session_state["sim_step_start"]
            step_s        = st.session_state.get("sim_step_hrs", 1) * 3600
            _prog_status  = get_status()   # reuse cached — no extra HTTP call
            cur_t         = _prog_status.get("sim_time", start_t)
            mans          = _prog_status.get("maneuvers_executed", 0)
            pct           = min(1.0, (cur_t - start_t) / max(step_s, 1))

            if cur_t >= target_t - 30:
                st.success(f"✓ {st.session_state.get('sim_step_hrs',1):.2f}h simulated")
                for k in ["sim_step_target","sim_step_start","sim_step_hrs"]:
                    st.session_state.pop(k, None)
                st.cache_data.clear()
            else:
                st.progress(pct)
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
        <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px;align-items:center">
          <span class="panel-badge badge-green" style="font-size:8px;padding:2px 7px;border-radius:2px">⬤ {nom_q} NOM</span>
          {'<span class="panel-badge badge-purple" style="font-size:8px;padding:2px 7px;border-radius:2px">⬤ ' + str(eol_q) + ' EOL</span>' if eol_q else ''}
          {'<span class="panel-badge badge-red" style="font-size:8px;padding:2px 7px;border-radius:2px">⬤ ' + str(cdm_q) + ' CDM</span>' if cdm_q else '<span class="panel-badge badge-green" style="font-size:8px;padding:2px 7px;border-radius:2px">⬤ 0 CDM</span>'}
          {'<span class="panel-badge badge-cyan" style="font-size:8px;padding:2px 7px;border-radius:2px">⏱ ' + f"{step_ms:.0f}" + 'ms</span>' if step_ms else ''}
        </div>''', unsafe_allow_html=True)

    auto_refresh = st.checkbox("Auto-refresh (3s)", value=False)
    if auto_refresh:
        import streamlit.components.v1 as _sc
        _sc.html(
            "<script>setTimeout(()=>window.parent.postMessage({type:'streamlit:rerun'},'*'),3000)</script>",
            height=0,
        )
        st.cache_data.clear()

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
            INIT_FUEL = 50.0  # matches backend STD_FUEL_MASS
            rows_chart = []
            for s in satellites:
                fuel_kg  = s.get("fuel_mass_kg") or s.get("fuel_kg") or 0
                fuel_pct_raw = s.get("fuel_pct")
                # Compute fuel_pct from fuel_mass_kg if fuel_pct missing/wrong
                if fuel_pct_raw is None or fuel_pct_raw > 100:
                    fuel_pct_val = min(100.0, max(0.0, fuel_kg / INIT_FUEL * 100))
                else:
                    fuel_pct_val = float(fuel_pct_raw)
                dv_ms = round((s.get("total_dv_used_kms", 0) or 0) * 1000, 2)
                rows_chart.append({
                    "Satellite":  s["id"].replace("SAT-Alpha-","A-"),
                    "Fuel %":     round(fuel_pct_val, 1),
                    "Fuel kg":    round(fuel_kg, 2),
                    "ΔV m/s":    dv_ms,
                    "Status":     s.get("status", "NOMINAL"),
                    "Slot Δ km":  round(s.get("slot_distance_km", 0), 2),
                    "Avoided":    s.get("collisions_avoided", 0),
                    "Pc Pruned":  s.get("pc_prune_count", 0),
                })

            df = pd.DataFrame(rows_chart).sort_values("Fuel %")

            # Tab view: Fuel vs ΔV
            _ct1, _ct2 = st.tabs(["⛽ FUEL %", "🚀 ΔV CONSUMED"])

            with _ct1:
                fuel_chart = altair_chart(
                    df,
                    x_field=alt.X("Fuel %:Q",
                        scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(labelColor="rgba(180,230,220,0.5)",
                                      labelFont="Share Tech Mono", labelFontSize=9,
                                      gridColor="rgba(0,210,180,0.07)",
                                      domainColor="rgba(0,210,180,0.15)")),
                    y_field=alt.Y("Satellite:N", sort=alt.SortField("Fuel %", order="ascending"),
                        axis=alt.Axis(labelColor="rgba(180,230,220,0.5)",
                                      labelFont="Share Tech Mono", labelFontSize=9,
                                      gridColor="rgba(0,210,180,0.07)",
                                      domainColor="rgba(0,210,180,0.15)")),
                    color_field="Fuel %:Q",
                    color_scale=alt.Scale(
                        domain=[0, 10, 25, 50, 75, 100],
                        range=["#ff2244","#ff7b00","#ffd700","#00c8b4","#00e8d0","#00ff88"]
                    ),
                    height=max(280, len(df) * 6),
                    tooltips=["Satellite","Fuel %","Fuel kg","Status","Slot Δ km","Avoided","Pc Pruned"]
                )
                st.altair_chart(fuel_chart, use_container_width=True)

            with _ct2:
                if df["ΔV m/s"].sum() > 0:
                    dv_chart = altair_chart(
                        df.sort_values("ΔV m/s", ascending=False),
                        x_field=alt.X("ΔV m/s:Q",
                            axis=alt.Axis(labelColor="rgba(180,230,220,0.5)",
                                          labelFont="Share Tech Mono", labelFontSize=9,
                                          gridColor="rgba(0,210,180,0.07)",
                                          domainColor="rgba(0,210,180,0.15)")),
                        y_field=alt.Y("Satellite:N", sort=alt.SortField("ΔV m/s", order="descending"),
                            axis=alt.Axis(labelColor="rgba(180,230,220,0.5)",
                                          labelFont="Share Tech Mono", labelFontSize=9,
                                          gridColor="rgba(0,210,180,0.07)",
                                          domainColor="rgba(0,210,180,0.15)")),
                        color_field="ΔV m/s:Q",
                        color_scale=alt.Scale(
                            domain=[0, 5, 15, 30, 50],
                            range=["#00e8d0","#00b4ff","#ffd700","#ff7b00","#ff2244"]
                        ),
                        height=max(280, len(df) * 6),
                        tooltips=["Satellite","ΔV m/s","Fuel %","Status","Avoided"]
                    )
                    st.altair_chart(dv_chart, use_container_width=True)
                else:
                    st.markdown(f'<div style="padding:20px 0;text-align:center">'
                                f'{badge("NO ΔV CONSUMED YET — RUN SIM STEP","cyan")}</div>',
                                unsafe_allow_html=True)

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
        st.markdown(dark_table(pd.DataFrame(rows)), unsafe_allow_html=True)

    st.markdown(ph("PER-SATELLITE DETAIL"), unsafe_allow_html=True)
    sat_ids   = [s["id"] for s in satellites[:20]]
    sat_short = [s.replace("SAT-Alpha-","A-") for s in sat_ids]
    sel       = st.selectbox("Select Satellite", sat_short)
    sel_id    = sat_ids[sat_short.index(sel)] if sel in sat_short else None

    if sel_id:
        cs      = get_contact_schedule(sel_id)
        windows = cs.get("windows", [])
        # Also check contact_windows from /api/satellites for the full list
        sat_detail = next((s for s in satellites if s["id"] == sel_id), {})
        all_windows = sat_detail.get("contact_windows") or windows
        if all_windows:
            st.markdown(ph(f"CONTACT WINDOWS — {len(all_windows)} SCHEDULED"), unsafe_allow_html=True)
            win_rows = [{
                "GS":           w.get("gs_id","?"),
                "Start (UTC)":  w.get("start_iso","?")[:19],
                "End (UTC)":    w.get("end_iso","?")[:19],
                "Duration s":   round(w.get("duration_s",0),1),
                "Peak El °":    w.get("peak_el_deg") or w.get("peak_elevation_deg","?"),
                "Pre-Blackout": "⚠ YES" if w.get("is_last_before_blackout") else "",
            } for w in all_windows]
            st.markdown(dark_table(pd.DataFrame(win_rows)), unsafe_allow_html=True)
        windows = all_windows  # use full list for card display below
        if windows:
            cols = st.columns(min(len(windows), 5))
            for i, w in enumerate(windows[:5]):
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
        ).properties(height=160,background="transparent").configure_view(strokeOpacity=0)
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
        st.markdown(dark_table(df), unsafe_allow_html=True)
    else:
        st.info("No maneuver history yet — simulation is initialising.")

    # ── ML Decision Log ────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(ph("ML DECISION LOG", "AUTONOMOUS EVENTS", "badge-purple"), unsafe_allow_html=True)
    events = get_events(200)
    ml_event_types = {
        "kalman_escalated_burn_timing":    ("🎯", "cyan",   "Kalman escalated burn — confident converging pair"),
        "sk_skipped_would_trigger_eol":    ("⚠",  "orange", "SK skipped — would trigger EOL"),
        "recovery_burn2_skipped_low_fuel": ("⛽", "red",    "Recovery burn 2 skipped — insufficient forecast fuel"),
        "blind_preupload_scheduled":       ("📡", "blue",   "Blind pre-upload — burn scheduled before blackout"),
        "collision_detected_no_evasion":   ("💥", "red",    "Collision detected — Pc below prune threshold"),
        "actual_collision":                ("🔴", "red",    "Actual hard-body collision detected"),
        "hohmann_recovery_planned":        ("🔄", "cyan",   "Hohmann recovery planned"),
        "eol_graveyard_planned":           ("⚰", "purple", "EOL graveyard transfer planned"),
        "ml_early_eol_warning":            ("⚠",  "orange", "ML early EOL warning — graveyard preemptive"),
    }
    ml_events = [e for e in events if e.get("type") in ml_event_types]
    if ml_events:
        rows = []
        for e in ml_events[-50:][::-1]:
            icon, color, label = ml_event_types[e["type"]]
            rows.append({
                "Time":      e.get("timestamp","?")[:16],
                "Event":     f"{icon} {e['type']}",
                "Satellite": (e.get("satellite","") or "").replace("SAT-Alpha-","A-"),
                "Detail":    str({k:v for k,v in e.items()
                                  if k not in ("type","time","timestamp","satellite")})[:120],
            })
        st.markdown(dark_table(pd.DataFrame(rows)), unsafe_allow_html=True)
    else:
        st.info("No ML decision events yet — events accumulate as the sim runs.")


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

    tab1, tab2, tab3, tab4,tab5 = st.tabs([
        "🎰 ΔV Bandit", "🔍 Anomaly Detector", "⛽ Fuel Forecast", "📉 Risk Trends","🎯 Risk Predictor"
    ])

    with tab1:
        b = get_ml_bandit()
        if b.get("_error"):
            st.warning(f"ML-1 endpoint error: {b['_error']}")
        st.markdown(ph("THOMPSON SAMPLING BANDIT — CONTEXTUAL ΔV OPTIMISER"), unsafe_allow_html=True)
        st.markdown(f'''<div style="background:linear-gradient(135deg,var(--bg3),var(--bg2));
          border:1px solid rgba(170,68,255,0.2);border-radius:4px;padding:12px 14px;margin:10px 0">
          <div class="metric-label">HOW IT WORKS</div>
          <div style="font-size:9px;color:var(--text2);line-height:1.75">
            6 arms: 0.004–0.015 km/s · <b>Thompson Sampling</b> (Beta posteriors) replaces UCB1<br>
            Contextual gates: urgent TCA &lt;30 min → restricts to larger arms only<br>
            Converges 2–3× faster than UCB1 · saves 20–40% fuel vs fixed 0.010 km/s<br>
            Sampler: <b>{b.get("sampler","thompson_sampling")}</b> · Updates: <b>{b.get("total_updates",0)}</b>
          </div></div>''', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Updates", b.get("total_updates", 0))
        with c2: st.metric("Best ΔV Arm",   f"{b.get('best_dv_kms','—')} km/s")
        with c3: st.metric("Sampler",        b.get("sampler","thompson_sampling").replace("_"," ").title())
        arms = b.get("arms", [])
        if arms:
            best_dv = b.get("best_dv_kms", 0)
            df = pd.DataFrame(arms)
            df["arm_type"] = df["dv_kms"].apply(lambda v: "best" if v == best_dv else "other")
            base = alt.Chart(df).encode(
                x=alt.X("dv_kms:O", axis=alt.Axis(labelColor="#a8c8e0", title="ΔV (km/s)")),
            )
            bar = base.mark_bar(cornerRadiusTopRight=3).encode(
                y=alt.Y("posterior_mean:Q", axis=alt.Axis(labelColor="#a8c8e0", title="Posterior Mean α/(α+β)")),
                color=alt.Color("arm_type:N",
                    scale=alt.Scale(domain=["best","other"], range=["#aa44ff","#00c8b4"]),
                    legend=None),
                tooltip=["dv_kms","posterior_mean","mean_reward","visits","alpha","beta"]
            )
            rule = base.mark_rule(color="#ffd700", strokeDash=[4,2], strokeWidth=1).encode(
                y=alt.Y("mean_reward:Q")
            )
            chart = (bar + rule).properties(
                height=200, background="#060f1c",
                title=alt.TitleParams("Posterior Mean (bars) + Mean Reward (gold line)", color="#a8c8e0")
            ).configure_view(strokeOpacity=0)
            st.altair_chart(chart, use_container_width=True)
            show_cols = [c for c in ["dv_kms","posterior_mean","mean_reward","visits","alpha","beta"] if c in df.columns]
            # Custom dark HTML table — st.dataframe uses an iframe that ignores our CSS
            _df_show = df[show_cols].reset_index(drop=True)
            _hdr = "".join(f'<th style="font-family:var(--font-mono);font-size:8px;letter-spacing:0.12em;color:var(--cyan);background:rgba(0,200,180,0.06);padding:6px 10px;border-bottom:1px solid var(--border2);text-transform:uppercase;white-space:nowrap">{c}</th>' for c in show_cols)
            _rows = ""
            for _, row in _df_show.iterrows():
                _cells = "".join(
                    f'<td style="font-family:var(--font-mono);font-size:9px;color:var(--text);padding:5px 10px;border-bottom:1px solid var(--border)">'
                    f'{f"{row[c]:.4f}" if isinstance(row[c], float) else row[c]}</td>'
                    for c in show_cols)
                _rows += f'<tr style="transition:background 0.1s" onmouseover="this.style.background=\'rgba(0,200,180,0.04)\'" onmouseout="this.style.background=\'\'"> {_cells}</tr>'
            st.markdown(f'<div style="overflow-x:auto;border:1px solid var(--border);border-radius:3px;margin-top:8px"><table style="width:100%;border-collapse:collapse;background:var(--bg3)"><thead><tr>{_hdr}</tr></thead><tbody>{_rows}</tbody></table></div>', unsafe_allow_html=True)
        else:
            st.info("Bandit data accumulates after first evasion burn (~60s)")

    with tab2:
        an = get_ml_anomalies()
        if an.get("_error"):
            st.warning(f"ML-2 endpoint error: {an['_error']}")
        st.markdown(ph("ISOLATION FOREST v2 — 12-D ONLINE ANOMALY SCORER"), unsafe_allow_html=True)
        st.markdown(f'''<div style="background:linear-gradient(135deg,var(--bg3),var(--bg2));
          border:1px solid rgba(170,68,255,0.2);border-radius:4px;padding:12px 14px;margin:10px 0">
          <div class="metric-label">HOW IT WORKS</div>
          <div style="font-size:9px;color:var(--text2);line-height:1.75">
            <b>{an.get("n_trees","?")} trees</b> · {an.get("subsample","?")} samples ·
            <b>{an.get("n_features","12")}-D features</b> · max depth {an.get("max_depth","10")} ·
            retrains every {int((an.get("retrain_interval_s") or 3600)//60)} min
            (train #{an.get("train_count",0)})<br>
            Score blend: 0.7 × forest + 0.3 × heuristic (v_residual)<br>
            &gt;0.45 → 1.5× · &gt;0.60 → 3× · &gt;0.75 → 5× · &gt;0.88 → <b>8× Pc boost</b> → earlier evasion<br>
            New debris scored <b>immediately on ingest</b> (no retrain wait)
          </div></div>''', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Status",        "TRAINED ✓" if an.get("trained") else "INIT…")
        with c2: st.metric("Debris Scored", an.get("debris_scored",0))
        with c3: st.metric("Features",      an.get("n_features", 12))
        with c4: st.metric("Retrains",      an.get("train_count", 0))
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
        st.markdown(ph("QUADRATIC RLS + EMA — FUEL DEPLETION FORECAST"), unsafe_allow_html=True)
        st.markdown(f'''<div style="background:linear-gradient(135deg,var(--bg3),var(--bg2));
          border:1px solid rgba(170,68,255,0.2);border-radius:4px;padding:12px 14px;margin:10px 0">
          <div class="metric-label">HOW IT WORKS</div>
          <div style="font-size:9px;color:var(--text2);line-height:1.75">
            <b>fuel(t) = w₀ + w₁·t + w₂·t²</b> (quadratic RLS, was linear) ·
            λ={ff.get("lambda_forgetting",0.97)} · EOL threshold: {ff.get("eol_threshold_kg","?")} kg<br>
            EMA burn-rate α={ff.get("ema_alpha",0.3)} · burst threshold: {ff.get("burst_rate_threshold_kgs","5.0")} g/s<br>
            EOL = min(quadratic solve, EMA estimate) → catches post-evasion fuel spikes fast<br>
            Skips SK when EOL &lt;3600s · pre-emptive graveyard at &lt;2h warning
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
                "Now kg":    s.get("fuel_now_kg",0),
                "+1h kg":    s.get("fuel_1h_kg",0),
                "+6h kg":    s.get("fuel_6h_kg",0),
                "+24h kg":   s.get("fuel_24h_kg",0),
                "Rate g/s":  round((s.get("burn_rate_ema_kgs",0) or 0) * 1000, 3),
                "EOL ⚠":     "⚠" if s.get("eol_warning") else "",
                "Status":    s.get("status","?"),
            } for s in sats_ff])
            fuel_max = max((s.get("fuel_now_kg", 0) for s in sats_ff), default=50)
            fuel_max = max(fuel_max, 1)
            st.dataframe(df, use_container_width=True, hide_index=True,
                         column_config={
                             "Now kg": st.column_config.ProgressColumn(
                                 "Now kg", min_value=0, max_value=fuel_max, format="%.1f kg"),
                             "Rate g/s": st.column_config.NumberColumn(
                                 "Burn Rate g/s", format="%.3f",
                                 help="EMA burn rate — spikes indicate post-evasion cluster"),
                         })
        else:
            st.info("Forecast populates after first burn events")

    with tab4:
        rt = get_ml_risk_trends()
        if rt.get("_error"):
            st.warning(f"ML-4 endpoint error: {rt['_error']}")
        st.markdown(ph("KALMAN RISK TRACKER — CONJUNCTION STATE ESTIMATOR"), unsafe_allow_html=True)
        st.markdown(f'''<div style="background:linear-gradient(135deg,var(--bg3),var(--bg2));
          border:1px solid rgba(170,68,255,0.2);border-radius:4px;padding:12px 14px;margin:10px 0">
          <div class="metric-label">HOW IT WORKS</div>
          <div style="font-size:9px;color:var(--text2);line-height:1.75">
            <b>Kalman filter</b> per (sat, debris) pair — state = [miss_km, rate_kms]<br>
            Adaptive measurement noise R scales with miss distance<br>
            Skip gate: trend &gt;{rt.get("skip_threshold_kms",0.04)} km/s <b>AND</b> P[1,1] &lt;0.01
            → 24h scan skipped (~30% CPU saving)<br>
            Unseen pairs get −∞ priority → always assessed first · evicts safest pairs under memory pressure
          </div></div>''', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Tracked Pairs",  int(rt.get("tracked_pairs") or 0))
        with c2: st.metric("Skip Threshold", f"{rt.get('skip_threshold_kms', 0.04):.2f} km/s")
        with c3: st.metric("Max Pairs",      rt.get("max_pairs", 6000))
        pairs = rt.get("converging_pairs", [])
        if pairs:
            df = pd.DataFrame(pairs)
            has_unc = "rate_uncertainty" in df.columns
            tooltip_fields = ["satellite_id", "debris_id", "trend_kms",
                               "smoothed_miss_km"] + (["rate_uncertainty"] if has_unc else [])
            bar = alt.Chart(df).mark_bar(cornerRadiusTopRight=2).encode(
                x=alt.X("trend_kms:Q",
                        axis=alt.Axis(labelColor="#a8c8e0",
                                      title="Kalman rate km/s (negative = converging)")),
                y=alt.Y("satellite_id:N", sort="x",
                        axis=alt.Axis(labelColor="#a8c8e0", labelFontSize=8)),
                color=alt.condition(
                    alt.datum.trend_kms < 0,
                    alt.value("#ff2244"), alt.value("#00ff88")),
                opacity=alt.Opacity(
                    "rate_uncertainty:Q",
                    scale=alt.Scale(domain=[0, 0.1], range=[1.0, 0.3]),
                    legend=None
                ) if has_unc else alt.value(1.0),
                tooltip=tooltip_fields,
            ).properties(height=280, background="#060f1c",
                         title=alt.TitleParams(
                             "Converging pairs — opacity = Kalman rate uncertainty",
                             color="#a8c8e0"))
            st.altair_chart(bar, use_container_width=True)
            show_cols = [c for c in
                         ["satellite_id","debris_id","trend_kms","smoothed_miss_km","rate_uncertainty"]
                         if c in df.columns]
            st.markdown(dark_table(df[show_cols]), unsafe_allow_html=True)
        else:
            st.info("Risk trends accumulate after first conjunction assessment (~60s)")
    with tab5:
        st.markdown(ph("XGBOOST COLLISION RISK PREDICTOR + XAI", "v4.0 · 21 FEATURES", "badge-cyan"), unsafe_allow_html=True)
        st.markdown('''<div style="background:linear-gradient(135deg,var(--bg3),var(--bg2));
          border:1px solid rgba(0,200,180,0.2);border-radius:4px;padding:12px 14px;margin:10px 0">
          <div class="metric-label">HOW IT WORKS — v4.0 PHYSICS-INFORMED ML</div>
          <div style="font-size:9px;color:var(--text2);line-height:1.75">
            <b>21 features</b> including RTN velocity components, Chan Pc log-prior, orbital resonance ratio<br>
            DART booster + Focal Loss + isotonic calibration + conformal prediction intervals<br>
            SHAP explainability: shows <i>why</i> the model made each decision — Glass Box, not Black Box
          </div></div>''', unsafe_allow_html=True)

        # ── Input sliders ────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            miss_distance     = st.slider("Miss Distance (m)",       10,   1000,  100)
            relative_velocity = st.slider("Relative Velocity (m/s)", 100, 15000, 7500)
            altitude_km       = st.slider("Altitude (km)",           300,   800,  550)
        with c2:
            inclination_diff  = st.slider("Inclination Diff (°)",      0,   180,   23)
            time_to_tca       = st.slider("Time to TCA (s)",           0, 14400, 1800)
            eccentricity      = st.slider("Debris Eccentricity",     0.0,  0.05, 0.02)
        with c3:
            combined_radius   = st.slider("Combined Radius (m)",     0.5,  20.0,  3.0)
            dist_rate         = st.slider("Dist Rate (km/s)",        -10.0, 10.0, 1.0)
            atm_density       = st.slider("Atm Density Multiplier",  0.5,   6.0,  1.0)

        if st.button("🔍 PREDICT RISK", use_container_width=True):
            try:
                resp = requests.post(f"{API}/ml/predict_risk", json={
                    "miss_distance_m":               miss_distance,
                    "relative_velocity_ms":          relative_velocity,
                    "altitude_km":                   altitude_km,
                    "inclination_diff_deg":          inclination_diff,
                    "time_to_closest_s":             time_to_tca,
                    "debris_eccentricity":           eccentricity,
                    "combined_radius_m":             combined_radius,
                    "dist_rate_kms":                 dist_rate,
                    "atmospheric_density_multiplier": atm_density,
                }, timeout=5)
                result = resp.json()

                if "error" in result:
                    st.warning(f"⚠ {result['error']}")
                else:
                    prob     = result.get("collision_probability", 0)
                    chan_pc  = result.get("chan_pc", 0)
                    model_id = result.get("model", "—")
                    unc      = result.get("uncertainty", {})

                    # ── Risk result card ──────────────────────────────────────
                    risk_color = "#ff2244" if result["risk_level"] == "HIGH" else "#00ff88"
                    risk_icon  = "🔴" if result["risk_level"] == "HIGH" else "🟢"
                    st.markdown(f'''<div style="background:rgba(0,0,0,0.3);border:1px solid {risk_color};
                      border-radius:4px;padding:14px 18px;margin:10px 0;text-align:center">
                      <div style="font-family:var(--font-display);font-size:22px;
                           font-weight:700;color:{risk_color};text-shadow:0 0 14px {risk_color}">
                        {risk_icon} {result["risk_level"]} RISK
                      </div>
                      <div style="font-family:var(--font-mono);font-size:11px;color:var(--text2);margin-top:6px">
                        ML Probability: <b style="color:{risk_color}">{prob*100:.2f}%</b>
                        &nbsp;·&nbsp; Chan Pc: <b>{chan_pc:.2e}</b>
                        &nbsp;·&nbsp; Model: <b>{model_id}</b>
                      </div>
                    </div>''', unsafe_allow_html=True)

                    # ── Conformal uncertainty interval ────────────────────────
                    if unc and unc.get("lower") is not None:
                        lo = unc["lower"]; hi = unc["upper"]
                        alert = unc.get("high_alert", False)
                        unc_color = "#ff7b00" if alert else "#00c8b4"
                        st.markdown(f'''<div style="font-family:var(--font-mono);font-size:9px;
                          color:{unc_color};border:1px solid {unc_color}33;
                          border-radius:3px;padding:6px 10px;margin:4px 0">
                          {"⚠ HIGH UNCERTAINTY — Chan fallback active" if alert else "✓ CONFORMAL INTERVAL"}
                          &nbsp; [{lo*100:.1f}% – {hi*100:.1f}%] &nbsp;·&nbsp;
                          {unc.get("coverage", 0.9)*100:.0f}% coverage &nbsp;·&nbsp;
                          calibration n={unc.get("calibration_n", 0)}
                        </div>''', unsafe_allow_html=True)

                    # ── Probability gauge bar ─────────────────────────────────
                    bar_w = int(prob * 100)
                    bar_col = "#ff2244" if prob > 0.7 else "#ffd700" if prob > 0.3 else "#00ff88"
                    st.markdown(f'''<div style="margin:8px 0">
                      <div style="font-family:var(--font-mono);font-size:8px;
                           color:var(--text2);letter-spacing:0.1em;margin-bottom:4px">
                        RISK PROBABILITY GAUGE
                      </div>
                      <div style="background:rgba(255,255,255,0.05);border-radius:2px;height:8px">
                        <div style="width:{bar_w}%;height:100%;background:{bar_col};
                             border-radius:2px;box-shadow:0 0 8px {bar_col};
                             transition:width 0.8s ease"></div>
                      </div>
                    </div>''', unsafe_allow_html=True)

                    # ── SHAP Feature Importance (static images from train_model.py) ──
                    st.markdown(ph("XAI — SHAP EXPLAINABILITY", "GLASS BOX", "badge-purple"), unsafe_allow_html=True)
                    _shap_tabs = st.tabs(["📊 Feature Importance", "🐝 Beeswarm", "📈 PR Curve"])

                    with _shap_tabs[0]:
                        import os as _os2
                        if _os2.path.exists("importance.png"):
                            st.image("importance.png", use_container_width=True,
                                     caption="Mean |SHAP| — higher = more influence on collision risk predictions")
                        else:
                            st.info("Run train_model.py with shap installed to generate importance.png")
                            st.markdown('''<div style="font-family:var(--font-mono);font-size:9px;
                              color:var(--text2);line-height:1.8">
                              <b>What SHAP shows:</b><br>
                              Each bar = how much that feature changes the model output on average.<br>
                              <b>miss_distance_m</b> and <b>vel_t_ms</b> (transverse velocity)
                              are typically the dominant drivers — confirming the physics intuition
                              that fast along-track approaches with small separation are most dangerous.
                            </div>''', unsafe_allow_html=True)

                    with _shap_tabs[1]:
                        if _os2.path.exists("shap_beeswarm.png"):
                            st.image("shap_beeswarm.png", use_container_width=True,
                                     caption="SHAP Beeswarm — each dot is one sample. "
                                             "Red = high feature value, Blue = low. "
                                             "Right of centre = pushes toward HIGH RISK.")
                        else:
                            st.info("Run train_model.py with shap installed to generate shap_beeswarm.png")
                            st.markdown('''<div style="font-family:var(--font-mono);font-size:9px;
                              color:var(--text2);line-height:1.8">
                              <b>How to read the beeswarm:</b><br>
                              • Each dot = one conjunction event from the test set<br>
                              • Horizontal position = SHAP value (impact on prediction)<br>
                              • Colour = feature value (red=high, blue=low)<br>
                              • A red dot far right for <b>miss_distance_m</b> = small miss distance
                                strongly pushed the model toward HIGH RISK
                            </div>''', unsafe_allow_html=True)

                    with _shap_tabs[2]:
                        if _os2.path.exists("model_report.png"):
                            st.image("model_report.png", use_container_width=True,
                                     caption="Left: XGBoost feature importance (gain). "
                                             "Right: Precision-Recall curve with optimal threshold marked.")
                        else:
                            st.info("Run train_model.py to generate model_report.png")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # ── Model metadata card ───────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        import os as _os3, json as _json3
        if _os3.path.exists("model_meta.json"):
            try:
                with open("model_meta.json") as _mf:
                    meta = _json3.load(_mf)
                c1m, c2m, c3m, c4m = st.columns(4)
                with c1m: st.metric("ROC-AUC",   f"{meta.get('test_roc_auc',0):.4f}")
                with c2m: st.metric("Avg-Prec",  f"{meta.get('test_avg_precision',0):.4f}")
                with c3m: st.metric("Recall",    f"{meta.get('test_recall_default',0):.4f}")
                with c4m: st.metric("Features",  meta.get("n_features", "—"))
                st.markdown(f'''<div style="font-family:var(--font-mono);font-size:8px;
                  color:var(--text3);margin-top:4px">
                  {meta.get("model_type","—")} ·
                  booster={meta.get("booster","—")} ·
                  focal_loss={meta.get("focal_loss","—")} ·
                  cv={meta.get("cv_strategy","—")} ·
                  hp_search={meta.get("hyperparameter_search","—")}
                </div>''', unsafe_allow_html=True)
            except Exception:
                pass
        else:
            st.info("model_meta.json not found — run train_model.py to see metrics here")






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