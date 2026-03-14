"""
Orbital Insight — Streamlit Analytics Dashboard v7.0
Connects to the FastAPI backend (port 8000).

New in v7.0:
  • Fleet Uptime panel  (/api/fleet/uptime)
  • Contact Schedule    (/api/satellite/{id}/contact_schedule)
  • Pc Prune monitor    (pc_prune_count per satellite)
  • Graveyard transfer  (burn type 'graveyard' in history)
  • Spatial index mode  (/api/status → spatial_index field)
  • Fleet contact summary (/api/fleet/contact_summary)
"""

import streamlit as st
import requests
import pandas as pd
import altair as alt
import time
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Orbital Insight v7.0 — Analytics",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os as _os

API = _os.environ.get("BACKEND_URL", "http://localhost:8000/api")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #010609;
    color: #a8c8e0;
}
.stApp { background: #010609; }
.main .block-container { padding-top: 1rem; }

.orbital-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 22px; font-weight: 900;
    color: #00d2ff;
    text-shadow: 0 0 25px rgba(0,210,255,0.5);
    letter-spacing: 0.12em; margin-bottom: 2px;
}
.orbital-sub { color: rgba(168,200,224,0.45); font-size: 10px; letter-spacing: 0.18em; margin-bottom: 16px; }

.metric-card {
    background: linear-gradient(135deg, #060f1c, #081220);
    border: 1px solid rgba(0,210,255,0.12);
    border-radius: 6px; padding: 14px 16px;
    transition: all 0.2s; position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(0,210,255,0.5), transparent);
}
.metric-card:hover { border-color: rgba(0,210,255,0.35); transform: translateY(-1px); }
.metric-label { font-size: 8px; letter-spacing: 0.2em; color: rgba(0,210,255,0.55); text-transform: uppercase; margin-bottom: 5px; }
.metric-value { font-family: 'Orbitron', sans-serif; font-size: 20px; font-weight: 700; color: #e5f3ff; }
.metric-value.red { color: #ff2d55; text-shadow: 0 0 10px rgba(255,45,85,0.45); }
.metric-value.green { color: #00ffaa; text-shadow: 0 0 10px rgba(0,255,170,0.35); }
.metric-value.yellow { color: #ffd60a; }
.metric-value.purple { color: #bf5af2; }
.metric-value.blue { color: #00d2ff; }
.metric-delta { font-size: 9px; margin-top: 3px; color: rgba(168,200,224,0.45); }

.section-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 10px; font-weight: 700;
    color: rgba(0,210,255,0.65); letter-spacing: 0.22em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(0,210,255,0.08);
    padding-bottom: 7px; margin: 18px 0 10px;
}

.badge { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 8px; letter-spacing: 0.1em; font-weight: 700; }
.badge-green  { background: rgba(0,255,170,0.12); color: #00ffaa; border: 1px solid rgba(0,255,170,0.28); }
.badge-red    { background: rgba(255,45,85,0.12);  color: #ff2d55; border: 1px solid rgba(255,45,85,0.28); }
.badge-yellow { background: rgba(255,214,10,0.12); color: #ffd60a; border: 1px solid rgba(255,214,10,0.28); }
.badge-purple { background: rgba(191,90,242,0.12); color: #bf5af2; border: 1px solid rgba(191,90,242,0.28); }
.badge-blue   { background: rgba(0,210,255,0.12);  color: #00d2ff; border: 1px solid rgba(0,210,255,0.28); }

.stButton > button {
    background: rgba(0,210,255,0.07) !important;
    border: 1px solid rgba(0,210,255,0.28) !important;
    color: #00d2ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important; letter-spacing: 0.1em !important;
    border-radius: 4px !important; transition: all 0.12s !important;
}
.stButton > button:hover {
    background: rgba(0,210,255,0.15) !important;
    border-color: rgba(0,210,255,0.55) !important;
}
div[data-testid="stMetric"] {
    background: #060f1c; border-radius: 5px; padding: 8px;
    border: 1px solid rgba(0,210,255,0.08);
}
.stDataFrame { background: #060f1c; border-radius: 5px; }
.stSelectbox div[data-baseweb="select"] { background: #060f1c; border: 1px solid rgba(0,210,255,0.2); }
.contact-win {
    background: #060f1c; border: 1px solid rgba(0,210,255,0.12);
    border-radius: 5px; padding: 10px 12px; margin-bottom: 6px;
}
.contact-win.last-before-blackout { border-color: rgba(255,123,0,0.4); }
.uptime-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── API helpers ───────────────────────────────────────────────────────────────
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

@st.cache_data(ttl=2)
def get_events():
    try: return requests.get(f"{API}/events", timeout=3).json()
    except: return []

@st.cache_data(ttl=5)
def get_cdm_registry(limit=50):
    try: return requests.get(f"{API}/cdm/registry?limit={limit}", timeout=3).json()
    except: return []

@st.cache_data(ttl=3)
def get_maneuver_history(limit=200):
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

def api_online():
    try: requests.get(f"{API}/status", timeout=2); return True
    except: return False

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="orbital-header">🛰 ORBITAL<br>INSIGHT</div>', unsafe_allow_html=True)
    st.markdown('<div class="orbital-sub">NSH 2026 · ACM v7.0</div>', unsafe_allow_html=True)

    online = api_online()
    status = get_status() if online else {}

    if online:
        spatial = status.get("spatial_index", "?")
        st.markdown(f'<span class="badge badge-green">● API ONLINE</span>&nbsp;&nbsp;<span class="badge badge-purple">{spatial.upper()}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-red">● API OFFLINE</span>', unsafe_allow_html=True)
        st.warning("Backend not reachable at localhost:8000")

    st.markdown("---")
    st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "📊 Dashboard",
        "🛰 Fleet Status",
        "📡 Contact Schedule",
        "📈 Uptime Monitor",
        "⚠ CDM Registry",
        "🔥 Maneuver History",
        "🌐 Ground Stations",
        "🗺 Live Visualizer",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-title">Simulation Control</div>', unsafe_allow_html=True)
    step_hrs = st.slider("Step (hours)", 0.25, 24.0, 1.0, 0.25)
    if st.button("▶ ADVANCE SIM", use_container_width=True):
        try:
            resp = requests.post(f"{API}/simulate/step",
                                 json={"step_seconds": step_hrs * 3600}, timeout=30)
            r = resp.json()
            st.success(f"✓ {r.get('status','DONE')} · {r.get('maneuvers_executed',0)} burns")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Step failed: {e}")

    if online:
        st.markdown("---")
        st.markdown('<div class="section-title">Fleet Quick Stats</div>', unsafe_allow_html=True)
        uptime_data = get_fleet_uptime()
        if uptime_data:
            fleet_pct = uptime_data.get("fleet_uptime_pct", 0)
            grade = uptime_data.get("grade", "—")
            grade_col = {"EXCELLENT":"#00ffaa","GOOD":"#00d2ff","ACCEPTABLE":"#ffd60a","POOR":"#ff2d55"}.get(grade,"#a8c8e0")
            st.markdown(f"""<div class="metric-card" style="margin-bottom:8px">
              <div class="metric-label">FLEET UPTIME</div>
              <div class="metric-value" style="color:{grade_col};font-size:18px">{fleet_pct:.1f}%</div>
              <div class="metric-delta">{grade}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (3s)", value=False)
    if auto_refresh:
        time.sleep(3)
        st.cache_data.clear()
        st.rerun()

# ─── Pages ─────────────────────────────────────────────────────────────────────

if "Dashboard" in page:
    status = get_status()
    satellites = get_satellites()
    conjunctions = get_conjunctions()
    uptime_data = get_fleet_uptime()
    heatmap = get_heatmap()

    st.markdown('<div class="orbital-header">MISSION COMMAND</div>', unsafe_allow_html=True)
    sim_t = int(status.get("sim_time", 0))
    st.markdown(
        f'<div class="orbital-sub">SIM T+{sim_t//3600:04d}H {(sim_t%3600)//60:02d}M'
        f' &nbsp;|&nbsp; {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC'
        f' &nbsp;|&nbsp; INDEX: {status.get("spatial_index","—").upper()}</div>',
        unsafe_allow_html=True
    )

    # ── Top metrics row ──────────────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    fleet_pct = uptime_data.get("fleet_uptime_pct", None)
    grade = uptime_data.get("grade", "—")
    grade_col = {"EXCELLENT":"green","GOOD":"blue","ACCEPTABLE":"yellow","POOR":"red"}.get(grade,"")
    total_prune = sum(s.get("pc_prune_count",0) for s in satellites)
    avg_fuel = (sum(s.get("fuel_pct",0) for s in satellites)/len(satellites) if satellites else 0)
    eol_count = status.get("satellites_eol", 0)

    for col, label, val, color, delta in [
        (c1, "NOMINAL SATS", status.get("satellites_nominal","—"), "green", "In slot"),
        (c2, "CDM ACTIVE", status.get("active_conjunctions","—"), "red" if (status.get("active_conjunctions",0) or 0)>0 else "green", "Critical"),
        (c3, "FLEET UPTIME", f"{fleet_pct:.1f}%" if fleet_pct is not None else "—", grade_col, grade),
        (c4, "Pc PRUNED", total_prune, "purple", "Burns saved"),
        (c5, "FUEL AVG", f"{avg_fuel:.0f}%", "yellow" if avg_fuel<50 else "", "Fleet mean"),
        (c6, "MANEUVERS", status.get("maneuvers_executed","—"), "", "Executed"),
        (c7, "EOL SATS", eol_count, "red" if eol_count>0 else "green", "Graveyard"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value {color}">{val}</div>
              <div class="metric-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([2.2, 1])

    with col_left:
        st.markdown('<div class="section-title">Fleet Fuel & ΔV Distribution</div>', unsafe_allow_html=True)
        if satellites:
            df = pd.DataFrame([{
                "Satellite": s["id"].replace("SAT-Alpha-","A-"),
                "Fuel %": round(s.get("fuel_pct", 0), 1),
                "Status": s.get("status", "?"),
                "Slot Dist km": round(s.get("slot_distance_km", 0), 2),
                "ΔV m/s": round((s.get("total_dv_used_kms", 0) or 0) * 1000, 1),
                "Avoided": s.get("collisions_avoided", 0),
                "Pc Pruned": s.get("pc_prune_count", 0),
            } for s in satellites]).sort_values("Fuel %")

            color_scale = alt.Scale(
                domain=[0, 15, 35, 65, 100],
                range=["#ff2d55", "#ff9500", "#ffd60a", "#00d2ff", "#00ffaa"]
            )
            chart = alt.Chart(df).mark_bar(cornerRadiusTopRight=2, cornerRadiusBottomRight=2).encode(
                x=alt.X("Fuel %:Q", scale=alt.Scale(domain=[0,100]),
                        axis=alt.Axis(labelColor="#a8c8e0", gridColor="rgba(0,210,255,0.08)")),
                y=alt.Y("Satellite:N", sort="-x",
                        axis=alt.Axis(labelColor="#a8c8e0", labelFontSize=8)),
                color=alt.Color("Fuel %:Q", scale=color_scale, legend=None),
                tooltip=["Satellite","Fuel %","Status","ΔV m/s","Avoided","Pc Pruned"]
            ).properties(height=300, background="#060f1c").configure_axis(
                gridColor="rgba(0,210,255,0.06)", domainColor="rgba(0,210,255,0.18)"
            ).configure_view(stroke="rgba(0,210,255,0.12)")
            st.altair_chart(chart, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">Active Conjunctions</div>', unsafe_allow_html=True)
        if conjunctions:
            for c in conjunctions[:7]:
                miss = c.get("miss_distance", 0)
                risk = c.get("risk_level", "GREEN")
                badge = "badge-red" if risk=="RED" else "badge-yellow" if risk=="YELLOW" else "badge-green"
                pc = c.get("probability", 0)
                pruned = "🟣 Pc PRUNED" if c.get("pc_pruned") else ""
                st.markdown(f"""<div style="background:#060f1c;border:1px solid rgba(0,210,255,0.1);
                    border-radius:5px;padding:8px 10px;margin-bottom:5px;">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:9px;color:#e5f3ff;font-weight:700">{c.get('satellite_id','?').replace('SAT-Alpha-','A-')}</span>
                        <span class="badge {badge}">{risk}</span>
                    </div>
                    <div style="font-size:8px;color:rgba(168,200,224,0.6);margin-top:3px">
                        vs {c.get('debris_id','?')} | {miss*1000:.0f}m | Pc:{pc:.1e}
                    </div>
                    <div style="font-size:8px;color:rgba(168,200,224,0.4)">{c.get('tca_iso','?')[:16]} {pruned}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:rgba(0,255,170,0.6);font-size:10px">✓ No active conjunctions</p>', unsafe_allow_html=True)

    # ── Uptime mini chart ────────────────────────────────────────────────────
    if uptime_data and uptime_data.get("per_satellite"):
        st.markdown('<div class="section-title">Per-Satellite Uptime</div>', unsafe_allow_html=True)
        df_up = pd.DataFrame([{
            "Satellite": s["id"].replace("SAT-Alpha-","A-"),
            "Uptime %": s["uptime_pct"],
            "Status": s["status"],
        } for s in uptime_data["per_satellite"]]).sort_values("Uptime %")

        up_chart = alt.Chart(df_up).mark_bar(cornerRadiusTopRight=2, cornerRadiusBottomRight=2).encode(
            x=alt.X("Uptime %:Q", scale=alt.Scale(domain=[0,100]),
                    axis=alt.Axis(labelColor="#a8c8e0")),
            y=alt.Y("Satellite:N", sort="-x", axis=alt.Axis(labelColor="#a8c8e0", labelFontSize=8)),
            color=alt.condition(
                alt.datum["Uptime %"] >= 99, alt.value("#00ffaa"),
                alt.condition(alt.datum["Uptime %"] >= 95, alt.value("#00d2ff"),
                alt.condition(alt.datum["Uptime %"] >= 90, alt.value("#ffd60a"), alt.value("#ff2d55")))
            ),
            tooltip=["Satellite","Uptime %","Status"]
        ).properties(height=280, background="#060f1c").configure_axis(
            gridColor="rgba(0,210,255,0.06)"
        ).configure_view(stroke="rgba(0,210,255,0.12)")
        st.altair_chart(up_chart, use_container_width=True)


elif "Fleet Status" in page:
    satellites = get_satellites()
    heatmap = get_heatmap()
    conj_ids = set(c["satellite_id"] for c in get_conjunctions())

    st.markdown('<div class="orbital-header">FLEET STATUS</div>', unsafe_allow_html=True)

    if satellites:
        rows = []
        for s in satellites:
            fp = s.get("fuel_pct", 0)
            status_badge = {"NOMINAL":"🟢","MANEUVERING":"🟠","OUT_OF_SLOT":"🟡","EOL":"🔴"}.get(s["status"],"⚪")
            rows.append({
                "": status_badge, "ID": s["id"].replace("SAT-Alpha-","A-"),
                "Status": s["status"],
                "Fuel %": round(fp, 1),
                "Alt km": s.get("altitude_km","?"),
                "Slot Δ km": s.get("slot_distance_km","?"),
                "ΔV m/s": round((s.get("total_dv_used_kms",0) or 0)*1000, 1),
                "Avoided": s.get("collisions_avoided",0),
                "Pc Pruned": s.get("pc_prune_count",0),
                "In Slot": "✓" if s.get("in_slot") else "✗",
                "CDM": "⚠" if s["id"] in conj_ids else "",
                "Cooldown s": s.get("cooldown_remaining_s","?"),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "Fuel %": st.column_config.ProgressColumn("Fuel %", min_value=0, max_value=100, format="%.1f%%"),
                         "Slot Δ km": st.column_config.NumberColumn(format="%.2f km"),
                         "Pc Pruned": st.column_config.NumberColumn("🟣 Pc Pruned"),
                     })


elif "Contact Schedule" in page:
    satellites = get_satellites()
    st.markdown('<div class="orbital-header">CONTACT SCHEDULE</div>', unsafe_allow_html=True)
    st.markdown('<div class="orbital-sub">PREDICTIVE GROUND STATION CONTACT WINDOWS — 4-HOUR HORIZON</div>',
                unsafe_allow_html=True)

    contact_summary = get_fleet_contact_summary()
    summary_sats = contact_summary.get("satellites", [])

    # Fleet contact overview table
    if summary_sats:
        st.markdown('<div class="section-title">Fleet Contact Overview</div>', unsafe_allow_html=True)
        rows = []
        for s in summary_sats:
            nw = s.get("next_window") or {}
            in_contact = s.get("in_contact_now", False)
            rows.append({
                "Satellite": s["id"].replace("SAT-Alpha-","A-"),
                "In Contact": "📡 YES" if in_contact else "○ No",
                "Current GS": s.get("current_gs") or "—",
                "Elevation °": s.get("current_elevation_deg") or "—",
                "Next GS": nw.get("gs_id","—"),
                "Window Start": (nw.get("start_iso","")[:16] if nw.get("start_iso") else "—"),
                "Duration s": nw.get("duration_s","—"),
                "Peak El °": nw.get("peak_el_deg","—"),
                "Blackout?": "⚠ YES" if nw.get("is_last_before_blackout") else "",
                "Pc Pruned": s.get("pc_prune_count",0),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Per-satellite detail
    st.markdown('<div class="section-title">Per-Satellite Contact Detail</div>', unsafe_allow_html=True)
    sat_ids = [s["id"] for s in satellites[:20]]
    sat_short = [s.replace("SAT-Alpha-","A-") for s in sat_ids]
    sel_short = st.selectbox("Select Satellite", sat_short)
    sel_id = sat_ids[sat_short.index(sel_short)] if sel_short in sat_short else None

    if sel_id:
        cs = get_contact_schedule(sel_id)
        windows = cs.get("windows", [])
        if windows:
            cols = st.columns(min(len(windows), 3))
            for i, w in enumerate(windows[:3]):
                with cols[i]:
                    is_blackout = w.get("is_last_before_blackout", False)
                    border_col = "rgba(255,123,0,0.4)" if is_blackout else "rgba(0,210,255,0.15)"
                    st.markdown(f"""<div style="background:#060f1c;border:1px solid {border_col};
                        border-radius:6px;padding:14px;margin-bottom:8px">
                        <div style="font-size:9px;color:rgba(0,210,255,0.6);letter-spacing:0.12em;margin-bottom:6px">
                            WINDOW {i+1}
                        </div>
                        <div style="font-family:'Orbitron',sans-serif;font-size:14px;
                            color:#00d2ff;margin-bottom:8px">{w.get('gs_id','?')}</div>
                        <div style="font-size:9px;color:rgba(168,200,224,0.7);margin-bottom:3px">
                            START: {w.get('start_iso','?')[:19]} UTC
                        </div>
                        <div style="font-size:9px;color:rgba(168,200,224,0.7);margin-bottom:6px">
                            END:   {w.get('end_iso','?')[:19]} UTC
                        </div>
                        <div style="display:flex;justify-content:space-between;align-items:center">
                            <span style="font-size:11px;color:#00ffaa;font-weight:700">
                                ⏱ {w.get('duration_s',0):.0f}s
                            </span>
                            <span style="font-size:10px;color:rgba(168,200,224,0.6)">
                                ⬆ {w.get('peak_elevation_deg',0):.1f}°
                            </span>
                        </div>
                        {'<div style="margin-top:8px;padding:3px 8px;background:rgba(255,123,0,0.12);border:1px solid rgba(255,123,0,0.35);border-radius:3px;font-size:8px;color:#ff7b00">⚠ LAST BEFORE BLACKOUT</div>' if is_blackout else ''}
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("No contact windows found for this satellite")


elif "Uptime Monitor" in page:
    uptime_data = get_fleet_uptime()
    st.markdown('<div class="orbital-header">FLEET UPTIME MONITOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="orbital-sub">STATION-KEEPING BOX COMPLIANCE · 10 km RADIUS</div>',
                unsafe_allow_html=True)

    if uptime_data:
        fleet_pct = uptime_data.get("fleet_uptime_pct", 0)
        grade = uptime_data.get("grade", "—")
        sim_elapsed = uptime_data.get("sim_time_elapsed_s", 0)
        active_sats = uptime_data.get("active_satellites", 0)
        grade_col = {"EXCELLENT":"green","GOOD":"blue","ACCEPTABLE":"yellow","POOR":"red"}.get(grade,"")

        c1,c2,c3,c4 = st.columns(4)
        for col, label, val, color, delta in [
            (c1, "FLEET UPTIME", f"{fleet_pct:.2f}%", grade_col, grade),
            (c2, "ACTIVE SATS", active_sats, "", "In constellation"),
            (c3, "SIM ELAPSED", f"{sim_elapsed/3600:.1f}h", "", "Simulation time"),
            (c4, "SCORE GRADE", grade, grade_col, "NSH 2026 rubric"),
        ]:
            with col:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value {color}">{val}</div>
                  <div class="metric-delta">{delta}</div>
                </div>""", unsafe_allow_html=True)

        # NSH scoring rubric callout
        st.markdown("---")
        st.markdown("""<div style="background:#060f1c;border:1px solid rgba(0,210,255,0.12);
            border-radius:6px;padding:14px;margin-bottom:16px">
            <div style="font-family:'Orbitron',sans-serif;font-size:9px;
                color:rgba(0,210,255,0.6);letter-spacing:0.2em;margin-bottom:8px">
                NSH 2026 SCORING RUBRIC
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">
                <div style="text-align:center;padding:8px;background:rgba(0,255,170,0.06);border-radius:4px">
                    <div style="font-size:12px;color:#00ffaa;font-weight:700">≥ 99%</div>
                    <div style="font-size:8px;color:rgba(168,200,224,0.5);margin-top:2px">EXCELLENT · 15 pts</div>
                </div>
                <div style="text-align:center;padding:8px;background:rgba(0,210,255,0.06);border-radius:4px">
                    <div style="font-size:12px;color:#00d2ff;font-weight:700">≥ 95%</div>
                    <div style="font-size:8px;color:rgba(168,200,224,0.5);margin-top:2px">GOOD · ~12 pts</div>
                </div>
                <div style="text-align:center;padding:8px;background:rgba(255,214,10,0.06);border-radius:4px">
                    <div style="font-size:12px;color:#ffd60a;font-weight:700">≥ 90%</div>
                    <div style="font-size:8px;color:rgba(168,200,224,0.5);margin-top:2px">ACCEPTABLE · ~9 pts</div>
                </div>
                <div style="text-align:center;padding:8px;background:rgba(255,45,85,0.06);border-radius:4px">
                    <div style="font-size:12px;color:#ff2d55;font-weight:700">< 90%</div>
                    <div style="font-size:8px;color:rgba(168,200,224,0.5);margin-top:2px">POOR</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Per-satellite uptime bars
        st.markdown('<div class="section-title">Per-Satellite Uptime</div>', unsafe_allow_html=True)
        per_sat = uptime_data.get("per_satellite", [])
        if per_sat:
            df_up = pd.DataFrame([{
                "Satellite": s["id"].replace("SAT-Alpha-","A-"),
                "Uptime %": s["uptime_pct"],
                "Samples In": s["samples_in_slot"],
                "Samples Total": s["samples_total"],
                "Status": s["status"],
            } for s in per_sat])
            st.dataframe(df_up, use_container_width=True, hide_index=True,
                         column_config={
                             "Uptime %": st.column_config.ProgressColumn(
                                 "Uptime %", min_value=0, max_value=100, format="%.2f%%"),
                         })
    else:
        st.info("Uptime data not yet available — wait for simulation to run.")


elif "CDM Registry" in page:
    cdms = get_cdm_registry(100)
    st.markdown('<div class="orbital-header">CDM REGISTRY</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="orbital-sub">{len(cdms)} CONJUNCTION DATA MESSAGES · CHAN Pc ALGORITHM · BISECTION TCA</div>',
                unsafe_allow_html=True)

    # Summary
    c1,c2,c3,c4 = st.columns(4)
    red_count = sum(1 for c in cdms if c.get("risk_level")=="RED")
    pruned = sum(1 for c in cdms if c.get("pc_pruned"))
    evading = sum(1 for c in cdms if c.get("evasion_planned"))
    for col, label, val, color in [
        (c1,"TOTAL CDMs",len(cdms),""),
        (c2,"RED RISK",red_count,"red"),
        (c3,"EVASION PLANNED",evading,"green"),
        (c4,"Pc PRUNED",pruned,"purple"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value {color}">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if cdms:
        rows = []
        for c in cdms:
            risk = c.get("risk_level","GREEN")
            rows.append({
                "CDM ID": c.get("cdm_id","?")[-18:],
                "Satellite": c.get("satellite_id","?").replace("SAT-Alpha-","A-"),
                "Debris": c.get("debris_id","?"),
                "Miss m": round(c.get("miss_distance_m",0), 1),
                "Pc": f"{c.get('probability_of_collision',0):.2e}",
                "Risk": risk,
                "Rel Vel km/s": round(c.get("relative_velocity_kms",0), 3),
                "TCA": c.get("tca_iso","?")[:16],
                "TCA s": round(c.get("time_to_tca_s",0)),
                "Evasion": "✓" if c.get("evasion_planned") else "—",
                "Pruned": "🟣" if c.get("pc_pruned") else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     column_config={
                         "Miss m": st.column_config.NumberColumn("Miss (m)", format="%.1f m"),
                         "TCA s": st.column_config.NumberColumn("TCA (s)", format="%d s"),
                     })


elif "Maneuver History" in page:
    history = get_maneuver_history(300)
    st.markdown('<div class="orbital-header">MANEUVER HISTORY</div>', unsafe_allow_html=True)

    if history:
        total_dv = sum((h.get("dv_mag_kms",0) or 0)*1000 for h in history)
        pre_upload = sum(1 for h in history if h.get("pre_upload"))
        graveyard = sum(1 for h in history if h.get("burn_type")=="graveyard")
        stationkeep = sum(1 for h in history if h.get("burn_type")=="stationkeep")
        evasion = sum(1 for h in history if h.get("burn_type")=="evasion")

        c1,c2,c3,c4,c5 = st.columns(5)
        for col, label, val, color in [
            (c1,"TOTAL BURNS",len(history),""),
            (c2,"TOTAL ΔV m/s",f"{total_dv:.1f}","yellow"),
            (c3,"EVASION BURNS",evasion,"red"),
            (c4,"PRE-UPLOAD",pre_upload,"blue"),
            (c5,"GRAVEYARD",graveyard,"purple"),
        ]:
            with col:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value {color}">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Burn type chart
        burn_counts = {}
        for h in history:
            bt = h.get("burn_type","?")
            burn_counts[bt] = burn_counts.get(bt, 0) + 1
        df_bt = pd.DataFrame(list(burn_counts.items()), columns=["Type","Count"])
        color_map = {
            "evasion":"#ff2d55","recovery":"#00d2ff","stationkeep":"#0090ff",
            "graveyard":"#bf5af2","commanded":"#ffd60a",
        }
        bt_chart = alt.Chart(df_bt).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
            y=alt.Y("Type:N", axis=alt.Axis(labelColor="#a8c8e0")),
            x=alt.X("Count:Q", axis=alt.Axis(labelColor="#a8c8e0")),
            color=alt.Color("Type:N",
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                legend=alt.Legend(labelColor="#a8c8e0", titleColor="#a8c8e0")),
            tooltip=["Type","Count"]
        ).properties(height=160, background="#060f1c").configure_axis(
            gridColor="rgba(0,210,255,0.06)"
        ).configure_view(stroke="rgba(0,210,255,0.12)")
        st.altair_chart(bt_chart, use_container_width=True)

        df = pd.DataFrame([{
            "Time": h.get("executed_iso","?")[:16],
            "Satellite": h.get("satellite_id","?").replace("SAT-Alpha-","A-"),
            "Type": h.get("burn_type","?"),
            "ΔV m/s": round((h.get("dv_mag_kms",0) or 0)*1000, 2),
            "Fuel kg": round(h.get("fuel_consumed_kg",0), 3),
            "Remaining kg": round(h.get("fuel_remaining_kg",0), 2),
            "Pre-Upload": "📡" if h.get("pre_upload") else "",
            "GS Window": h.get("contact_window","—") or "—",
        } for h in history])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No maneuver history yet — simulation is initializing.")


elif "Ground Stations" in page:
    gs_list = get_ground_stations()
    st.markdown('<div class="orbital-header">GROUND STATION NETWORK</div>', unsafe_allow_html=True)

    if gs_list:
        # Summary row
        total_vis = sum(g.get("visible_count",0) for g in gs_list)
        active_gs = sum(1 for g in gs_list if g.get("visible_count",0)>0)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">ACTIVE STATIONS</div>
              <div class="metric-value green">{active_gs}/{len(gs_list)}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">TOTAL VISIBILITY</div>
              <div class="metric-value">{total_vis} satellite passes</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        cols = st.columns(3)
        for i, gs in enumerate(gs_list):
            with cols[i % 3]:
                vis = gs.get("visible_count", 0)
                color = "green" if vis > 10 else "yellow" if vis > 3 else ""
                visible_list = ", ".join(s.replace("SAT-Alpha-","A-") for s in (gs.get("visible_satellites") or [])[:5])
                st.markdown(f"""<div class="metric-card" style="margin-bottom:10px">
                    <div class="metric-label">{gs['id']}</div>
                    <div style="font-size:12px;font-weight:700;color:#e5f3ff;margin:4px 0">
                        {gs.get('name','?').replace('_',' ')}
                    </div>
                    <div style="font-size:8px;color:rgba(168,200,224,0.55)">
                        Lat: {gs['lat']}° · Lon: {gs['lon']}°<br>
                        Min El: {gs.get('min_el',5)}° · Alt: {gs.get('elev_m',0)}m
                    </div>
                    <div style="margin-top:8px;display:flex;justify-content:space-between;align-items:center">
                        <span class="metric-value {color}" style="font-size:16px">{vis}</span>
                        <span style="font-size:8px;color:rgba(168,200,224,0.45)">visible sats</span>
                    </div>
                    {f'<div style="font-size:7px;color:rgba(0,255,170,0.5);margin-top:4px">{visible_list}</div>' if visible_list else ''}
                </div>""", unsafe_allow_html=True)


elif "Live Visualizer" in page:
    st.markdown('<div class="orbital-header">LIVE VISUALIZER</div>', unsafe_allow_html=True)
    st.markdown('<div class="orbital-sub">HTML FRONTEND — FULL ORBITAL INSIGHT DASHBOARD</div>',
                unsafe_allow_html=True)

    frontend_url = "http://localhost:80"
    st.markdown(f"""
    <div style="border:1px solid rgba(0,210,255,0.2);border-radius:6px;overflow:hidden;
        background:#010609;box-shadow:0 0 40px rgba(0,0,0,0.6)">
      <iframe src="{frontend_url}" width="100%" height="760" frameborder="0"
        style="display:block;border:none;border-radius:6px"></iframe>
    </div>
    <p style="font-size:8px;color:rgba(168,200,224,0.3);margin-top:6px;text-align:center">
      ↗ <a href="{frontend_url}" target="_blank" style="color:rgba(0,210,255,0.5)">Open in new tab</a>
      for full-screen experience &nbsp;|&nbsp;
      <a href="http://localhost:8000/docs" target="_blank" style="color:rgba(0,210,255,0.35)">API Docs</a>
    </p>
    """, unsafe_allow_html=True)
