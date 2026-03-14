"""
Orbital Insight — Streamlit Analytics Dashboard v7.2
NSH 2026 · Team BroCODE
Login overlay + full hover/tooltip system matching index.html aesthetic.
"""

import streamlit as st
import requests
import pandas as pd
import altair as alt
import time
import os as _os
from datetime import datetime

st.set_page_config(
    page_title="Orbital Insight v7.0 — Analytics",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API = _os.environ.get("BACKEND_URL", "http://localhost:8000/api")

# ═══════════════════════════════════════════════════════════════════
#  LOGIN OVERLAY — mirrors index.html card, gated by sessionStorage
# ═══════════════════════════════════════════════════════════════════
LOGIN_CSS = """
<style>
#oi-login{position:fixed;inset:0;z-index:9999999;background:#000a0f;
  display:flex;align-items:center;justify-content:center;
  transition:opacity 0.8s ease,transform 0.8s ease;overflow:hidden;}
#oi-login.fade-out{opacity:0;transform:scale(1.04);pointer-events:none;}
#oi-lcanvas{position:absolute;inset:0;pointer-events:none;}
.oi-ring{position:absolute;border-radius:50%;border:1px solid;
  animation:oi-spin linear infinite;pointer-events:none;}
.oi-ring-1{width:440px;height:440px;border-color:rgba(0,210,180,0.10);
  top:50%;left:50%;margin:-220px 0 0 -220px;animation-duration:22s;}
.oi-ring-2{width:320px;height:320px;border-color:rgba(0,210,180,0.07);
  top:50%;left:50%;margin:-160px 0 0 -160px;animation-duration:14s;animation-direction:reverse;}
.oi-ring-3{width:200px;height:200px;border-color:rgba(0,200,180,0.05);
  top:50%;left:50%;margin:-100px 0 0 -100px;animation-duration:8s;}
.oi-ring::after{content:\'\';position:absolute;top:-3px;left:50%;width:5px;height:5px;
  margin-left:-2px;border-radius:50%;background:#00e8d0;box-shadow:0 0 12px rgba(0,232,208,0.6);}
@keyframes oi-spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.oi-scan{position:absolute;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(0,210,180,0.7),transparent);
  animation:oi-scandown 4s linear infinite;pointer-events:none;}
@keyframes oi-scandown{0%{top:-2px;opacity:0}5%{opacity:1}95%{opacity:1}100%{top:100%;opacity:0}}
#oi-splash{position:absolute;z-index:4;top:50%;left:50%;transform:translate(-50%,-50%);
  text-align:center;pointer-events:none;--shift:-155px;
  animation:oi-splashMove 3s cubic-bezier(0.4,0,0.2,1) forwards;}
#oi-splash-text{font-family:\'Orbitron\',sans-serif;font-size:26px;font-weight:900;
  color:#00e8d0;letter-spacing:0.16em;white-space:nowrap;
  animation:oi-splashGlow 1.2s ease-in-out infinite alternate;}
#oi-splash-sub{font-family:\'Share Tech Mono\',monospace;font-size:9px;
  color:rgba(180,230,220,0.4);letter-spacing:0.22em;margin-top:7px;}
@keyframes oi-splashMove{
  0%{opacity:0;transform:translate(-50%,-50%) translateY(12px)}
  14%{opacity:1;transform:translate(-50%,-50%) translateY(0)}
  44%{opacity:1;transform:translate(-50%,-50%) translateY(0)}
  72%{opacity:1;transform:translate(-50%,-50%) translateY(var(--shift))}
  84%{opacity:0;transform:translate(-50%,-50%) translateY(var(--shift))}
  100%{opacity:0;transform:translate(-50%,-50%) translateY(var(--shift))}}
@keyframes oi-splashGlow{
  from{text-shadow:0 0 16px rgba(0,232,208,0.55),0 0 36px rgba(0,232,208,0.18)}
  to{text-shadow:0 0 28px rgba(0,232,208,1.0),0 0 70px rgba(0,232,208,0.38)}}
.oi-card{position:relative;z-index:2;width:400px;padding:44px 40px 40px;
  background:rgba(3,12,18,0.97);border:1px solid rgba(0,210,180,0.2);
  border-radius:4px;box-shadow:0 0 60px rgba(0,0,0,0.8);overflow:hidden;
  animation:oi-cardGrow 0.75s cubic-bezier(0.16,1,0.3,1) both;animation-delay:1.85s;}
@keyframes oi-cardGrow{
  from{clip-path:inset(0 0 100% 0 round 4px);opacity:0.8}
  to{clip-path:inset(0 0 0% 0 round 4px);opacity:1}}
.oi-card::before{content:\'\';position:absolute;top:-1px;left:20%;right:20%;height:1px;
  background:linear-gradient(90deg,transparent,#00e8d0,transparent);box-shadow:0 0 12px #00e8d0;}
.oi-logo{font-family:\'Orbitron\',sans-serif;font-size:20px;font-weight:900;
  color:#00e8d0;letter-spacing:0.14em;text-align:center;
  text-shadow:0 0 12px rgba(0,232,208,0.4);margin-bottom:4px;
  animation:oi-rev 0.35s ease both;animation-delay:2.45s;}
.oi-sub{font-family:\'Share Tech Mono\',monospace;font-size:9px;
  color:rgba(180,230,220,0.4);letter-spacing:0.2em;text-align:center;margin-bottom:28px;
  animation:oi-rev 0.35s ease both;animation-delay:2.5s;}
@keyframes oi-rev{from{opacity:0}to{opacity:1}}
.oi-body{animation:oi-bodyIn 0.45s ease both;animation-delay:2.55s;}
@keyframes oi-bodyIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.oi-flabel{font-family:\'Share Tech Mono\',monospace;font-size:8px;letter-spacing:0.16em;
  color:rgba(180,230,220,0.4);margin-bottom:6px;display:block;transition:color 0.2s;}
.oi-fwrap{margin-bottom:16px;position:relative;}
.oi-fwrap:focus-within .oi-flabel{color:#00e8d0;}
.oi-input{width:100%;background:rgba(0,20,28,0.8);border:1px solid rgba(0,210,180,0.18);
  border-radius:3px;padding:10px 14px;font-family:\'Share Tech Mono\',monospace;font-size:11px;
  color:rgba(220,255,248,0.88);outline:none;
  transition:border-color 0.2s,box-shadow 0.2s;letter-spacing:0.06em;box-sizing:border-box;}
.oi-input:focus{border-color:#00e8d0;
  box-shadow:0 0 0 2px rgba(0,210,180,0.12),0 0 12px rgba(0,210,180,0.08) inset;}
.oi-sweep{position:absolute;bottom:0;left:0;right:0;height:2px;overflow:hidden;border-radius:0 0 3px 3px;}
.oi-sweep::after{content:\'\';position:absolute;top:0;left:-100%;width:100%;height:100%;
  background:linear-gradient(90deg,transparent,#00e8d0,transparent);}
.oi-fwrap:focus-within .oi-sweep::after{animation:oi-sweep 0.4s ease-out forwards;}
@keyframes oi-sweep{from{left:-100%}to{left:100%}}
.oi-err{font-family:\'Share Tech Mono\',monospace;font-size:9px;color:#ff2244;
  text-align:center;margin-bottom:12px;min-height:14px;letter-spacing:0.08em;
  animation:oi-shake 0.35s ease-out;}
@keyframes oi-shake{0%,100%{transform:translateX(0)}20%{transform:translateX(-6px)}40%{transform:translateX(6px)}60%{transform:translateX(-4px)}80%{transform:translateX(4px)}}
.oi-btn{width:100%;padding:13px;margin-top:4px;
  font-family:\'Orbitron\',sans-serif;font-size:11px;font-weight:700;
  letter-spacing:0.18em;color:#000a0f;
  background:linear-gradient(135deg,#00c8b4,#00e8d0);
  border:none;border-radius:3px;cursor:pointer;
  transition:transform 0.15s,box-shadow 0.15s;box-shadow:0 0 20px rgba(0,210,180,0.3);}
.oi-btn:hover{transform:translateY(-1px);box-shadow:0 4px 24px rgba(0,210,180,0.5);}
.oi-btn:active{transform:translateY(1px);}
.oi-granted{position:absolute;inset:0;z-index:10;
  display:none;align-items:center;justify-content:center;
  background:rgba(0,255,136,0.04);border-radius:4px;}
.oi-granted.active{display:flex;}
.oi-granted-text{font-family:\'Orbitron\',sans-serif;font-size:14px;font-weight:900;
  color:#00ff88;letter-spacing:0.25em;text-shadow:0 0 12px rgba(0,255,136,0.5);
  animation:oi-grantPulse 0.5s ease-out;}
@keyframes oi-grantPulse{from{opacity:0;transform:scale(0.8)}to{opacity:1;transform:scale(1)}}
.oi-dots{display:none;text-align:center;margin:10px 0;}
.oi-dots.active{display:block;}
.oi-dots span{display:inline-block;width:5px;height:5px;border-radius:50%;
  background:#00e8d0;margin:0 3px;animation:oi-dot 1.2s ease-in-out infinite;}
.oi-dots span:nth-child(2){animation-delay:0.2s}
.oi-dots span:nth-child(3){animation-delay:0.4s}
@keyframes oi-dot{0%,80%,100%{transform:scale(0.6);opacity:0.3}40%{transform:scale(1);opacity:1}}
.oi-corner{position:absolute;width:14px;height:14px;pointer-events:none;}
.oi-corner::before,.oi-corner::after{content:\'\';position:absolute;background:#00e8d0;}
.oi-corner::before{width:100%;height:1px;}.oi-corner::after{width:1px;height:100%;}
.oi-corner.tl{top:10px;left:10px;}.oi-corner.tl::before,.oi-corner.tl::after{top:0;left:0;}
.oi-corner.tr{top:10px;right:10px;}.oi-corner.tr::before,.oi-corner.tr::after{top:0;right:0;}
.oi-corner.bl{bottom:10px;left:10px;}.oi-corner.bl::before,.oi-corner.bl::after{bottom:0;left:0;}
.oi-corner.br{bottom:10px;right:10px;}.oi-corner.br::before,.oi-corner.br::after{bottom:0;right:0;}
.oi-hint{font-family:\'Share Tech Mono\',monospace;font-size:8px;
  color:rgba(180,230,220,0.3);text-align:center;margin-top:18px;letter-spacing:0.1em;}
.oi-hint span{color:#00c8b4;cursor:pointer;transition:color 0.15s;}
.oi-hint span:hover{color:#00e8d0;}
/* Cursor ring */
#oi-cur{position:fixed;top:0;left:0;width:22px;height:22px;margin:-11px 0 0 -11px;
  border-radius:50%;border:1px solid rgba(0,200,180,0.4);pointer-events:none;
  z-index:9999998;transition:width 0.15s,height 0.15s,margin 0.15s,border-color 0.15s,opacity 0.2s;opacity:0;}
body:hover #oi-cur{opacity:1;}
#oi-cur.h{width:34px;height:34px;margin:-17px 0 0 -17px;border-color:#00e8d0;box-shadow:0 0 8px rgba(0,200,180,0.3);}
#oi-cur.c{width:14px;height:14px;margin:-7px 0 0 -7px;opacity:0.5;}
/* Tooltip */
#oi-tip{position:fixed;z-index:9999999;pointer-events:none;opacity:0;
  transition:opacity 0.12s;background:rgba(3,12,18,0.97);
  border:1px solid rgba(0,210,180,0.25);border-radius:3px;padding:9px 13px;
  font-family:\'Share Tech Mono\',monospace;font-size:9.5px;
  color:rgba(220,255,248,0.88);max-width:230px;line-height:1.8;
  box-shadow:0 4px 24px rgba(0,0,0,0.7);}
#oi-tip::before{content:\'\';position:absolute;top:0;left:15%;right:15%;height:1px;
  background:linear-gradient(90deg,transparent,#00e8d0,transparent);}
/* Sat row hover */
.sat-row{background:var(--bg3);border:1px solid var(--border);border-radius:3px;
  padding:8px 10px;margin-bottom:3px;cursor:pointer;
  transition:border-color 0.15s,background 0.15s,transform 0.1s;position:relative;}
.sat-row:hover{background:rgba(0,200,180,0.06);border-color:var(--border2);transform:translateX(2px);}
.sat-row.sa{border-color:rgba(255,34,68,0.3);}
.sat-row.sa:hover{border-color:var(--red);box-shadow:0 0 10px rgba(255,34,68,0.12);}
.fuel-bar-wrap{height:3px;background:rgba(255,255,255,0.07);border-radius:2px;overflow:hidden;margin-top:5px;}
.fuel-bar-fill{height:100%;border-radius:2px;}
</style>
"""

LOGIN_HTML = """
<div id="oi-login">
  <canvas id="oi-lcanvas"></canvas>
  <div class="oi-ring oi-ring-1"></div>
  <div class="oi-ring oi-ring-2"></div>
  <div class="oi-ring oi-ring-3"></div>
  <div class="oi-scan"></div>
  <div id="oi-splash">
    <div id="oi-splash-text">ORBITAL INSIGHT</div>
    <div id="oi-splash-sub">NSH 2026 · ACM v7.0 · ANALYTICS CHANNEL</div>
  </div>
  <div class="oi-card">
    <div class="oi-corner tl"></div><div class="oi-corner tr"></div>
    <div class="oi-corner bl"></div><div class="oi-corner br"></div>
    <div class="oi-granted" id="oi-granted"><div class="oi-granted-text">ACCESS GRANTED</div></div>
    <div class="oi-logo">ORBITAL INSIGHT</div>
    <div class="oi-sub">NSH 2026 · ACM ANALYTICS · v7.0</div>
    <div class="oi-body">
      <div class="oi-fwrap">
        <label class="oi-flabel">OPERATOR ID</label>
        <input class="oi-input" id="oi-user" type="text" placeholder="Enter operator ID" autocomplete="off" spellcheck="false"/>
        <div class="oi-sweep"></div>
      </div>
      <div class="oi-fwrap">
        <label class="oi-flabel">ACCESS CODE</label>
        <input class="oi-input" id="oi-pass" type="password" placeholder="Enter access code" autocomplete="off"/>
        <div class="oi-sweep"></div>
      </div>
      <div class="oi-err" id="oi-err"></div>
      <div class="oi-dots" id="oi-dots"><span></span><span></span><span></span></div>
      <button class="oi-btn" id="oi-btn" onclick="oiLogin(event)">AUTHENTICATE</button>
      <div class="oi-hint">Demo: <span onclick="oiFill()">admin / orbital2026</span></div>
    </div>
  </div>
</div>
<div id="oi-cur"></div>
<div id="oi-tip"></div>
<script>
(function(){
  if(sessionStorage.getItem('oi_auth')==='1'){
    var el=document.getElementById('oi-login');
    if(el)el.style.display='none';
    boot(); return;
  }
  var lc=document.getElementById('oi-lcanvas');
  if(lc){
    var W=lc.width=window.innerWidth,H=lc.height=window.innerHeight;
    var cx2=lc.getContext('2d');
    var stars=Array.from({length:180},function(){return{x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.2,o:Math.random()*0.5+0.1};});
    var t2=0;
    (function frame(){
      cx2.clearRect(0,0,W,H); t2+=0.008;
      stars.forEach(function(s){cx2.fillStyle='rgba(255,255,255,'+(s.o*(0.7+0.3*Math.sin(t2+s.x)))+')';cx2.beginPath();cx2.arc(s.x,s.y,s.r,0,Math.PI*2);cx2.fill();});
      requestAnimationFrame(frame);
    })();
  }
  requestAnimationFrame(function(){
    var sp=document.getElementById('oi-splash'),card=document.querySelector('.oi-card');
    if(sp&&card){var cr=card.getBoundingClientRect();sp.style.setProperty('--shift',(cr.top+54-window.innerHeight/2)+'px');}
  });
  setTimeout(function(){var s=document.getElementById('oi-splash');if(s)s.remove();},3100);
  var CREDS=[{u:'admin',p:'orbital2026'},{u:'nsh2026',p:'acm'},{u:'brocode',p:'orbital'}];
  window.oiFill=function(){document.getElementById('oi-user').value='admin';document.getElementById('oi-pass').value='orbital2026';document.getElementById('oi-err').textContent='';};
  window.oiLogin=function(e){
    var btn=document.getElementById('oi-btn');
    var user=document.getElementById('oi-user').value.trim();
    var pass=document.getElementById('oi-pass').value.trim();
    var err=document.getElementById('oi-err'),dots=document.getElementById('oi-dots');
    err.textContent='';
    if(!user||!pass){err.textContent='CREDENTIALS REQUIRED';return;}
    btn.disabled=true;btn.textContent='AUTHENTICATING...';dots.classList.add('active');
    setTimeout(function(){
      dots.classList.remove('active');
      var ok=CREDS.some(function(c){return c.u===user&&c.p===pass;});
      if(ok){
        document.getElementById('oi-granted').classList.add('active');
        var card=document.querySelector('.oi-card');
        card.style.borderColor='rgba(0,255,136,0.5)';
        setTimeout(function(){
          var sc=document.getElementById('oi-login');
          sc.classList.add('fade-out');
          setTimeout(function(){sc.style.display='none';sessionStorage.setItem('oi_auth','1');boot();},850);
        },900);
      } else {
        btn.disabled=false;btn.textContent='AUTHENTICATE';
        err.textContent=''; void err.offsetWidth; err.textContent='INVALID CREDENTIALS — ACCESS DENIED';
      }
    },1400);
  };
  document.getElementById('oi-pass').addEventListener('keydown',function(e){if(e.key==='Enter')document.getElementById('oi-btn').click();});
  document.getElementById('oi-user').addEventListener('keydown',function(e){if(e.key==='Enter')document.getElementById('oi-pass').focus();});
  boot();
})();
function boot(){
  var ring=document.getElementById('oi-cur');
  var tip=document.getElementById('oi-tip');
  if(ring){
    document.addEventListener('mousemove',function(e){ring.style.left=e.clientX+'px';ring.style.top=e.clientY+'px';});
    document.addEventListener('mousedown',function(){ring.classList.add('c');});
    document.addEventListener('mouseup',function(){ring.classList.remove('c');});
    document.addEventListener('mouseover',function(e){
      if(e.target.closest('button,a,.metric-card,.conj-card,.contact-win,.gs-card,.sat-row,.rubric-box,.badge,[data-tip]'))ring.classList.add('h');
      else ring.classList.remove('h');
    });
  }
  if(tip){
    document.addEventListener('mousemove',function(e){tip.style.left=(e.clientX+16)+'px';tip.style.top=(e.clientY-8)+'px';});
    document.addEventListener('mouseover',function(e){
      var el=e.target.closest('[data-tip]');
      if(el){tip.innerHTML=el.getAttribute('data-tip').replace(/&#10;/g,'<br>');tip.style.opacity='1';}
    });
    document.addEventListener('mouseout',function(e){if(e.target.closest('[data-tip]'))tip.style.opacity='0';});
  }
}
</script>
"""
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');
:root {
  --bg:#010508;--bg2:#030b10;--bg3:#061420;--panel:#05111a;
  --cyan:#00c8b4;--cyan2:#00e8d0;--cyan3:#00fff0;
  --blue:#0090ff;--yellow:#ffd700;--orange:#ff7b00;
  --red:#ff2244;--green:#00ff88;--purple:#aa44ff;
  --text:rgba(220,255,248,0.88);--text2:rgba(180,230,220,0.50);
  --border:rgba(0,210,180,0.10);--border2:rgba(0,210,180,0.22);--border3:rgba(0,210,180,0.40);
  --glow-cyan:0 0 12px rgba(0,210,180,0.5);--glow-red:0 0 12px rgba(255,34,68,0.5);
  --glow-green:0 0 12px rgba(0,255,136,0.4);
}
html,body,[class*="css"],.stApp{font-family:'Share Tech Mono',monospace!important;background:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg)!important;}
.main .block-container{padding-top:0.75rem!important;max-width:100%!important;}
.stApp::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.04) 2px,rgba(0,0,0,0.04) 4px);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#020c14 0%,#010508 100%)!important;border-right:1px solid var(--border2)!important;}
[data-testid="stSidebar"]::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan2),transparent);}
[data-testid="stSidebar"] .stRadio>div{gap:2px!important;}
[data-testid="stSidebar"] .stRadio label{background:transparent!important;border:1px solid transparent!important;border-radius:4px!important;padding:7px 12px!important;margin:1px 0!important;cursor:pointer!important;transition:all 0.15s!important;color:var(--text2)!important;font-size:11px!important;letter-spacing:0.08em!important;display:block!important;width:100%!important;}
[data-testid="stSidebar"] .stRadio label:hover{background:rgba(0,200,180,0.08)!important;border-color:var(--border2)!important;color:var(--cyan2)!important;padding-left:16px!important;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]>label>div:first-child{display:none!important;}
.stSlider [data-baseweb="slider"] div[role="slider"]{background:var(--cyan2)!important;box-shadow:0 0 8px var(--cyan2)!important;}
.stButton>button{background:rgba(0,200,180,0.06)!important;border:1px solid var(--border2)!important;color:var(--cyan2)!important;font-family:'Share Tech Mono',monospace!important;font-size:10px!important;letter-spacing:0.12em!important;border-radius:3px!important;transition:all 0.15s!important;text-transform:uppercase!important;}
.stButton>button:hover{background:rgba(0,200,180,0.14)!important;border-color:var(--border3)!important;box-shadow:0 0 16px rgba(0,200,180,0.2)!important;transform:translateY(-1px)!important;}
.stButton>button:active{transform:translateY(0px)!important;box-shadow:0 0 8px rgba(0,200,180,0.3)!important;}
.stCheckbox label{color:var(--text2)!important;font-size:10px!important;letter-spacing:0.1em!important;transition:color 0.15s!important;}
.stCheckbox label:hover{color:var(--cyan2)!important;}
.stSelectbox div[data-baseweb="select"]>div{background:var(--bg3)!important;border-color:var(--border2)!important;color:var(--text)!important;font-family:'Share Tech Mono',monospace!important;font-size:10px!important;transition:border-color 0.15s,box-shadow 0.15s!important;}
.stSelectbox div[data-baseweb="select"]:focus-within>div{border-color:var(--cyan2)!important;box-shadow:0 0 10px rgba(0,200,180,0.2)!important;}
[data-baseweb="popover"]{background:var(--bg3)!important;border:1px solid var(--border2)!important;}
[data-baseweb="menu"] li{color:var(--text2)!important;font-size:10px!important;transition:all 0.1s!important;}
[data-baseweb="menu"] li:hover{background:rgba(0,200,180,0.1)!important;color:var(--cyan2)!important;}
.stDataFrame,[data-testid="stDataFrame"]{background:var(--bg3)!important;border:1px solid var(--border)!important;border-radius:4px!important;}
[data-testid="stDataFrame"] th{background:rgba(0,200,180,0.08)!important;color:var(--cyan)!important;font-family:'Share Tech Mono',monospace!important;font-size:9px!important;letter-spacing:0.12em!important;text-transform:uppercase!important;border-bottom:1px solid var(--border2)!important;}
[data-testid="stDataFrame"] td{color:var(--text)!important;font-family:'Share Tech Mono',monospace!important;font-size:10px!important;border-bottom:1px solid var(--border)!important;}
[data-testid="stDataFrame"] tr:hover td{background:rgba(0,200,180,0.05)!important;}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:12px 0!important;}
div[data-testid="stMetric"]{background:var(--bg3)!important;border:1px solid var(--border)!important;border-radius:4px!important;padding:10px 12px!important;transition:border-color 0.2s,box-shadow 0.2s!important;}
div[data-testid="stMetric"]:hover{border-color:var(--border2)!important;box-shadow:0 0 12px rgba(0,200,180,0.1)!important;}
.orbital-header{font-family:'Orbitron',sans-serif;font-size:20px;font-weight:900;color:var(--cyan2);text-shadow:0 0 20px rgba(0,232,208,0.4);letter-spacing:0.14em;margin-bottom:2px;border-left:3px solid var(--cyan2);padding-left:10px;}
.orbital-sub{color:var(--text2);font-size:9px;letter-spacing:0.2em;margin-bottom:14px;padding-left:13px;}
.section-title{font-family:'Orbitron',sans-serif;font-size:9px;font-weight:700;color:rgba(0,200,180,0.6);letter-spacing:0.25em;text-transform:uppercase;border-bottom:1px solid var(--border);padding-bottom:6px;margin:16px 0 10px;position:relative;}
.section-title::after{content:'';position:absolute;bottom:-1px;left:0;width:40px;height:1px;background:var(--cyan2);}
.metric-card{background:linear-gradient(135deg,var(--bg3),var(--bg2));border:1px solid var(--border);border-radius:4px;padding:12px 14px;position:relative;overflow:hidden;transition:border-color 0.2s,box-shadow 0.2s,transform 0.2s;cursor:default;}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:0.5;}
.metric-card:hover{border-color:var(--border2);box-shadow:0 0 18px rgba(0,200,180,0.12),inset 0 0 12px rgba(0,200,180,0.03);transform:translateY(-1px);}
.metric-label{font-size:7px;letter-spacing:0.22em;color:rgba(0,200,180,0.5);text-transform:uppercase;margin-bottom:6px;}
.metric-value{font-family:'Orbitron',sans-serif;font-size:18px;font-weight:700;color:var(--text);line-height:1;}
.metric-value.red{color:var(--red);text-shadow:0 0 8px rgba(255,34,68,0.4);}
.metric-value.green{color:var(--green);text-shadow:0 0 8px rgba(0,255,136,0.35);}
.metric-value.yellow{color:var(--yellow);text-shadow:0 0 8px rgba(255,215,0,0.3);}
.metric-value.purple{color:var(--purple);text-shadow:0 0 8px rgba(170,68,255,0.35);}
.metric-value.blue{color:var(--blue);text-shadow:0 0 8px rgba(0,144,255,0.35);}
.metric-value.cyan{color:var(--cyan2);text-shadow:0 0 8px rgba(0,232,208,0.35);}
.metric-value.orange{color:var(--orange);text-shadow:0 0 8px rgba(255,123,0,0.35);}
.metric-delta{font-size:8px;margin-top:4px;color:var(--text2);letter-spacing:0.08em;}
.badge{display:inline-block;padding:2px 8px;border-radius:2px;font-size:8px;letter-spacing:0.1em;font-weight:700;text-transform:uppercase;transition:box-shadow 0.2s;}
.badge:hover{box-shadow:0 0 8px currentColor;}
.badge-green{background:rgba(0,255,136,0.1);color:var(--green);border:1px solid rgba(0,255,136,0.3);}
.badge-red{background:rgba(255,34,68,0.1);color:var(--red);border:1px solid rgba(255,34,68,0.3);}
.badge-yellow{background:rgba(255,215,0,0.1);color:var(--yellow);border:1px solid rgba(255,215,0,0.3);}
.badge-purple{background:rgba(170,68,255,0.1);color:var(--purple);border:1px solid rgba(170,68,255,0.3);}
.badge-blue{background:rgba(0,144,255,0.1);color:var(--blue);border:1px solid rgba(0,144,255,0.3);}
.badge-cyan{background:rgba(0,200,180,0.1);color:var(--cyan2);border:1px solid rgba(0,200,180,0.3);}
.badge-orange{background:rgba(255,123,0,0.1);color:var(--orange);border:1px solid rgba(255,123,0,0.3);}
.conj-card{background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:9px 11px;margin-bottom:5px;transition:border-color 0.15s,box-shadow 0.15s,transform 0.15s;cursor:default;position:relative;overflow:hidden;}
.conj-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--border2);}
.conj-card:hover{border-color:var(--border2);box-shadow:0 0 12px rgba(0,200,180,0.08);transform:translateX(2px);}
.conj-card.risk-red{border-color:rgba(255,34,68,0.25);}
.conj-card.risk-red::before{background:var(--red);box-shadow:var(--glow-red);}
.conj-card.risk-yellow{border-color:rgba(255,215,0,0.2);}
.conj-card.risk-yellow::before{background:var(--yellow);}
.conj-card.risk-green::before{background:var(--green);}
.conj-card:hover.risk-red{box-shadow:0 0 14px rgba(255,34,68,0.15);}
.contact-win{background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:12px 14px;margin-bottom:6px;transition:border-color 0.2s,box-shadow 0.2s,transform 0.15s;cursor:default;}
.contact-win:hover{border-color:var(--border2);box-shadow:0 0 14px rgba(0,200,180,0.1);transform:translateY(-1px);}
.contact-win.last-before-blackout{border-color:rgba(255,123,0,0.3);}
.contact-win.last-before-blackout:hover{border-color:rgba(255,123,0,0.55);box-shadow:0 0 14px rgba(255,123,0,0.12);}
.gs-card{background:linear-gradient(135deg,var(--bg3),var(--bg2));border:1px solid var(--border);border-radius:4px;padding:12px 14px;margin-bottom:10px;position:relative;overflow:hidden;transition:border-color 0.2s,box-shadow 0.2s,transform 0.15s;cursor:default;}
.gs-card:hover{border-color:var(--border2);box-shadow:0 0 16px rgba(0,200,180,0.1);transform:translateY(-1px);}
.gs-card.active::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--green),transparent);}
.rubric-box{text-align:center;padding:10px 8px;border-radius:3px;border:1px solid transparent;transition:border-color 0.2s,box-shadow 0.2s,transform 0.15s;cursor:default;}
.rubric-box:hover{transform:translateY(-2px);}
.rubric-excellent{background:rgba(0,255,136,0.06);border-color:rgba(0,255,136,0.15);}
.rubric-excellent:hover{border-color:rgba(0,255,136,0.4);box-shadow:0 4px 16px rgba(0,255,136,0.1);}
.rubric-good{background:rgba(0,200,180,0.06);border-color:rgba(0,200,180,0.15);}
.rubric-good:hover{border-color:rgba(0,200,180,0.4);box-shadow:0 4px 16px rgba(0,200,180,0.1);}
.rubric-ok{background:rgba(255,215,0,0.06);border-color:rgba(255,215,0,0.15);}
.rubric-ok:hover{border-color:rgba(255,215,0,0.4);box-shadow:0 4px 16px rgba(255,215,0,0.1);}
.rubric-poor{background:rgba(255,34,68,0.06);border-color:rgba(255,34,68,0.15);}
.rubric-poor:hover{border-color:rgba(255,34,68,0.4);box-shadow:0 4px 16px rgba(255,34,68,0.1);}
.sidebar-logo{font-family:'Orbitron',sans-serif;font-size:18px;font-weight:900;color:var(--cyan2);text-shadow:0 0 20px rgba(0,232,208,0.4);letter-spacing:0.15em;padding:8px 4px 2px;}
.sidebar-sub{font-size:8px;color:var(--text2);letter-spacing:0.2em;margin-bottom:12px;padding:0 4px;}
.sidebar-uptime{background:linear-gradient(135deg,var(--bg3),var(--bg2));border:1px solid var(--border);border-radius:4px;padding:10px 12px;margin-bottom:8px;transition:border-color 0.2s;}
.sidebar-uptime:hover{border-color:var(--border2);}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.4;}}
.live-dot{animation:pulse 1.8s ease-in-out infinite;}
</style>
""", unsafe_allow_html=True)

# ─── API helpers ─────────────────────────────────────────────────────────────

# ── Login + cursor overlay ────────────────────────────────────────────────────
st.markdown(LOGIN_CSS + LOGIN_HTML, unsafe_allow_html=True)

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

@st.cache_data(ttl=3)
def get_metrics():
    try: return requests.get(f"{API}/metrics", timeout=3).json()
    except: return {}

def api_online():
    try: requests.get(f"{API}/status", timeout=2); return True
    except: return False

def mk_badge(text, cls="cyan"):
    return f'<span class="badge badge-{cls}">{text}</span>'

def risk_badge(risk):
    cls = {"RED": "red", "YELLOW": "yellow", "GREEN": "green"}.get(risk, "cyan")
    return mk_badge(risk, cls)

def metric_card(label, value, color="", delta=""):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    return f'''<div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {color}">{value}</div>
      {delta_html}
    </div>'''

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    online = api_online()
    status = get_status() if online else {}
    uptime_quick = get_fleet_uptime() if online else {}

    st.markdown('''<div class="sidebar-logo">🛰 ORBITAL<br>INSIGHT</div>
    <div class="sidebar-sub">NSH 2026 · ACM v7.0 · TEAM BROCODE</div>''', unsafe_allow_html=True)

    if online:
        spatial = status.get("spatial_index", "?")
        sim_t_s = int(status.get("sim_time", 0))
        st.markdown(
            f'{mk_badge("⬤ LIVE", "green")} {mk_badge(spatial.upper(), "purple")} {mk_badge(f"T+{sim_t_s//3600:03d}H", "cyan")}',
            unsafe_allow_html=True)
    else:
        st.markdown(mk_badge("⬤ OFFLINE", "red"), unsafe_allow_html=True)
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


    if online and uptime_quick:
        st.markdown("---")
        st.markdown('<div class="section-title">Fleet Health</div>', unsafe_allow_html=True)
        fp_q    = uptime_quick.get("fleet_uptime_pct", 0)
        grade_q = uptime_quick.get("grade", "—")
        gcol_q  = {"EXCELLENT":"green","GOOD":"cyan","ACCEPTABLE":"yellow","POOR":"red"}.get(grade_q, "")
        nom_q   = status.get("satellites_nominal", 0)
        eol_q   = status.get("satellites_eol", 0)
        cdm_q   = status.get("active_conjunctions", 0)
        st.markdown(f'''<div class="sidebar-uptime">
          <div class="metric-label">FLEET UPTIME</div>
          <div class="metric-value {gcol_q}" style="font-size:22px">{fp_q:.1f}%</div>
          <div class="metric-delta">{grade_q}</div>
        </div>
        <div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px">
          {mk_badge(f"{nom_q} NOM","green")}
          {mk_badge(f"{eol_q} EOL","purple") if eol_q else ""}
          {mk_badge(f"{cdm_q} CDM","red") if cdm_q else mk_badge("0 CDM","green")}
        </div>''', unsafe_allow_html=True)

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
                _risk_cls = "badge-red" if risk=="RED" else "badge-yellow" if risk=="YELLOW" else "badge-green"
                pc = c.get("probability", 0)
                pruned_badge = f'&nbsp;{mk_badge("Pc PRUNED", "purple")}' if c.get("pc_pruned") else ""
                st.markdown(f'''<div class="conj-card risk-{risk.lower()}">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-size:10px;color:var(--text);font-weight:700">
                      {c.get("satellite_id","?").replace("SAT-Alpha-","A-")}</span>
                    {risk_badge(risk)}{pruned_badge}
                  </div>
                  <div style="font-size:8px;color:var(--text2);margin-top:4px">
                    ↔ {c.get("debris_id","?")} &nbsp;|&nbsp; {miss*1000:.0f} m &nbsp;|&nbsp; Pc {pc:.1e}
                  </div>
                  <div style="font-size:7px;color:rgba(180,230,220,0.3);margin-top:2px">
                    TCA {c.get("tca_iso","?")[:16]}</div>
                </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="padding:12px 0">{mk_badge("✓ NO ACTIVE CONJUNCTIONS", "green")}</div>', unsafe_allow_html=True)

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
                    cls = "contact-win last-before-blackout" if is_blackout else "contact-win"
                    warn = '''<div style="margin-top:8px;padding:3px 8px;background:rgba(255,123,0,0.1);
                        border:1px solid rgba(255,123,0,0.3);border-radius:3px;font-size:8px;
                        color:var(--orange);letter-spacing:0.1em">⚠ LAST BEFORE BLACKOUT</div>''' if is_blackout else ""
                    st.markdown(f'''<div class="{cls}">
                      <div style="font-size:8px;color:rgba(0,200,180,0.5);letter-spacing:0.15em;margin-bottom:6px">WINDOW {i+1}</div>
                      <div style="font-family:'Orbitron',sans-serif;font-size:15px;color:var(--cyan2);
                        margin-bottom:8px;text-shadow:0 0 10px rgba(0,200,180,0.3)">{w.get("gs_id","?")}</div>
                      <div style="font-size:9px;color:var(--text2);margin-bottom:2px">START &nbsp;{w.get("start_iso","?")[:19]} UTC</div>
                      <div style="font-size:9px;color:var(--text2);margin-bottom:8px">END &nbsp;&nbsp;{w.get("end_iso","?")[:19]} UTC</div>
                      <div style="display:flex;justify-content:space-between">
                        <span style="color:var(--green);font-size:12px;font-weight:700">⏱ {w.get("duration_s",0):.0f}s</span>
                        <span style="color:var(--text2);font-size:10px">⬆ {w.get("peak_elevation_deg",0):.1f}°</span>
                      </div>{warn}
                    </div>''', unsafe_allow_html=True)
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

        st.markdown("---")
        st.markdown('<div class="section-title">NSH 2026 Scoring Rubric</div>', unsafe_allow_html=True)
        rc1, rc2, rc3, rc4 = st.columns(4)
        for col, pct, lbl, pts, cls, color in [
            (rc1,"≥ 99%","EXCELLENT","15 pts","excellent","#00ff88"),
            (rc2,"≥ 95%","GOOD","~12 pts","good","#00c8b4"),
            (rc3,"≥ 90%","ACCEPTABLE","~9 pts","ok","#ffd700"),
            (rc4,"< 90%","POOR","—","poor","#ff2244"),
        ]:
            with col:
                st.markdown(f'''<div class="rubric-box rubric-{cls}">
                  <div style="font-size:16px;color:{color};font-weight:700;font-family:'Orbitron',sans-serif">{pct}</div>
                  <div style="font-size:9px;color:{color};letter-spacing:0.1em;margin-top:3px">{lbl}</div>
                  <div style="font-size:8px;color:var(--text2);margin-top:2px">{pts}</div>
                </div>''', unsafe_allow_html=True)

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
                active_cls = "gs-card active" if vis > 0 else "gs-card"
                st.markdown(f'''<div class="{active_cls}">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <span style="font-size:8px;color:var(--text2);letter-spacing:0.12em">{gs["id"]}</span>
                    {mk_badge("ACTIVE","green") if vis > 0 else mk_badge("IDLE","cyan")}
                  </div>
                  <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:6px;
                    font-family:'Rajdhani',sans-serif;letter-spacing:0.05em">{gs.get("name","?").replace("_"," ")}</div>
                  <div style="font-size:8px;color:var(--text2);line-height:1.6">
                    Lat {gs["lat"]}° · Lon {gs["lon"]}°<br>Min El {gs.get("min_el",5)}° · Alt {gs.get("elev_m",0)} m
                  </div>
                  <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center">
                    <span class="metric-value {color}" style="font-size:20px">{vis}</span>
                    <span style="font-size:8px;color:var(--text2)">visible sats</span>
                  </div>
                  {f'<div style="font-size:7px;color:rgba(0,255,136,0.45);margin-top:4px">{visible_list}</div>' if visible_list else ""}
                </div>''', unsafe_allow_html=True)


elif "Live Visualizer" in page:
    st.markdown('<div class="orbital-header">LIVE VISUALIZER</div>', unsafe_allow_html=True)
    st.markdown('<div class="orbital-sub">HTML FRONTEND — FULL ORBITAL INSIGHT DASHBOARD</div>',
                unsafe_allow_html=True)

    frontend_url = _os.environ.get("FRONTEND_URL", "http://localhost:80")
    st.markdown(f'''
    <div style="border:1px solid rgba(0,200,180,0.2);border-radius:4px;overflow:hidden;
        background:var(--bg);box-shadow:0 0 40px rgba(0,0,0,0.7)">
      <div style="background:rgba(0,200,180,0.06);border-bottom:1px solid rgba(0,200,180,0.15);
        padding:6px 12px;display:flex;align-items:center;gap:8px">
        <span style="font-size:8px;color:var(--cyan2);letter-spacing:0.15em">ORBITAL INSIGHT · HTML DASHBOARD</span>
        <span class="badge badge-green live-dot" style="margin-left:auto">⬤ LIVE</span>
      </div>
      <iframe src="{frontend_url}" width="100%" height="760" frameborder="0" style="display:block;border:none"></iframe>
    </div>
    <p style="font-size:8px;color:rgba(180,230,220,0.3);margin-top:6px;text-align:center">
      <a href="{frontend_url}" target="_blank" style="color:rgba(0,200,180,0.45);text-decoration:none">↗ OPEN FULLSCREEN</a>
      &nbsp;·&nbsp;
      <a href="http://localhost:8000/docs" target="_blank" style="color:rgba(0,200,180,0.3);text-decoration:none">API DOCS</a>
      &nbsp;·&nbsp;
      <a href="http://localhost:8000/api/logs" target="_blank" style="color:rgba(0,200,180,0.3);text-decoration:none">AUDIT LOG</a>
    </p>''', unsafe_allow_html=True)
