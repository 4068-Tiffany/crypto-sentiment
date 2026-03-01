import subprocess, sys

def install_deps():
    for dep in ["flask", "requests", "vaderSentiment"]:
        try: __import__(dep.replace("-","_"))
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable,"-m","pip","install",dep,"-q"])

install_deps()

from flask import Flask, jsonify, render_template_string, redirect
import requests, time, threading
from datetime import datetime
from sentiment import analyze_posts, summarize, _pipeline, USE_BERT

app     = Flask(__name__)
cache   = {"data": None, "last_updated": None, "loading": False}
HEADERS = {"User-Agent": "btc-pulse/2.0"}

SUBREDDITS    = ["Bitcoin","CryptoCurrency","BitcoinMarkets","CryptoMarkets"]
POSTS_PER_SUB = 25
POST_SORT     = "new"
REFRESH_SECS  = 300

def fetch_sub(sub):
    try:
        r = requests.get(f"https://www.reddit.com/r/{sub}/{POST_SORT}.json?limit={POSTS_PER_SUB}", headers=HEADERS, timeout=10)
        return [{"source":f"r/{sub}","title":c["data"].get("title",""),"text":(c["data"].get("title","")+" "+c["data"].get("selftext","")).strip(),"upvotes":c["data"].get("score",0)} for c in r.json()["data"]["children"]]
    except: return []

def get_price():
    try:
        d = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true",timeout=10).json()["bitcoin"]
        return d["usd"], round(d["usd_24h_change"],2)
    except: return None, None

def classify(s):
    return "BULLISH" if s>=0.05 else "BEARISH" if s<=-0.05 else "NEUTRAL"

def run_analysis():
    cache["loading"] = True
    posts = []
    for sub in SUBREDDITS:
        posts.extend(fetch_sub(sub)); time.sleep(0.8)
    results  = analyze_posts(posts)
    summary  = summarize(results)
    price, change = get_price()
    sub_stats = {}
    for sub in SUBREDDITS:
        subs = [r for r in results if r["source"]==f"r/{sub}"]
        if subs:
            sc = {"BULLISH":0,"BEARISH":0,"NEUTRAL":0}
            for r in subs: sc[r["label"]] += 1
            sub_stats[f"r/{sub}"] = {k:round(v/len(subs)*100,1) for k,v in sc.items()}
    cache["data"] = {
        "total":summary["total"],"counts":summary["counts"],"percentages":summary["percentages"],
        "avg_score":summary["avg_score"],"overall":summary["overall"],
        "btc_price":price,"btc_change":change,"sub_stats":sub_stats,
        "method": summary["method"],
        "posts":[{"source":r["source"],"title":r["title"],"label":r["label"],"compound":r["compound"],"upvotes":r["upvotes"]} for r in results],
    }
    cache["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cache["loading"] = False
    print(f"✅ {summary['total']} posts — {summary['overall']} ({summary['method']})")

def bg_loop():
    while True: run_analysis(); time.sleep(REFRESH_SECS)

# ═══════════════════════════════════════════════════════════
#  SHARED CSS + NAV
# ═══════════════════════════════════════════════════════════
SHARED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@200;400;600;800&display=swap');

*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}

:root{
  --bg:       #01050f;
  --bg2:      #030a18;
  --surface:  #061228;
  --surface2: #0a1a35;
  --cyan:     #00f5ff;
  --cyan2:    #00c8d4;
  --magenta:  #ff00ff;
  --mag2:     #cc00cc;
  --green:    #00ff88;
  --green2:   #00cc66;
  --red:      #ff2255;
  --red2:     #cc0033;
  --yellow:   #ffe500;
  --yellow2:  #ccb800;
  --orange:   #ff8c00;
  --text:     #b0d0f0;
  --muted:    #2a4060;
  --border:   rgba(0,245,255,0.08);
  --border2:  rgba(0,245,255,0.18);
}

html{scroll-behavior:smooth}

body{
  background:var(--bg);
  color:var(--text);
  font-family:'Exo 2',sans-serif;
  min-height:100vh;
  overflow-x:hidden;
}

/* GRID BG */
body::before{
  content:'';
  position:fixed;inset:0;z-index:0;
  background-image:
    linear-gradient(rgba(0,245,255,0.025) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,245,255,0.025) 1px,transparent 1px);
  background-size:60px 60px;
  pointer-events:none;
}

/* SCAN LINE */
body::after{
  content:'';
  position:fixed;inset:0;z-index:9998;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.04) 3px,rgba(0,0,0,0.04) 4px);
  pointer-events:none;
}

/* GLOW ORBS */
.orbs{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.orb{position:absolute;border-radius:50%;filter:blur(140px);animation:orb 14s ease-in-out infinite}
.orb1{width:700px;height:700px;background:rgba(0,245,255,0.06);top:-200px;left:-200px;animation-delay:0s}
.orb2{width:600px;height:600px;background:rgba(255,0,255,0.05);bottom:-200px;right:-150px;animation-delay:-7s}
.orb3{width:400px;height:400px;background:rgba(0,255,136,0.04);top:50%;left:50%;transform:translate(-50%,-50%);animation-delay:-3.5s}
@keyframes orb{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(40px,-50px) scale(1.1)}66%{transform:translate(-30px,40px) scale(0.9)}}

/* NAV */
nav{
  position:sticky;top:0;z-index:100;
  height:64px;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 48px;
  background:rgba(1,5,15,0.85);
  backdrop-filter:blur(24px);
  border-bottom:1px solid var(--border2);
  animation:navIn 0.5s ease both;
}
@keyframes navIn{from{transform:translateY(-100%);opacity:0}to{transform:translateY(0);opacity:1}}

.nav-logo{
  font-family:'Rajdhani',sans-serif;
  font-weight:700;font-size:1.4rem;
  letter-spacing:4px;
  color:white;
  display:flex;align-items:center;gap:12px;
  text-decoration:none;
}
.logo-gem{
  width:36px;height:36px;border-radius:8px;
  background:linear-gradient(135deg,var(--cyan),var(--magenta));
  display:flex;align-items:center;justify-content:center;
  font-size:1.1rem;
  box-shadow:0 0 24px rgba(0,245,255,0.5),0 0 60px rgba(255,0,255,0.2);
  animation:gemPulse 2.5s ease-in-out infinite;
}
@keyframes gemPulse{0%,100%{box-shadow:0 0 24px rgba(0,245,255,0.5),0 0 60px rgba(255,0,255,0.2)}50%{box-shadow:0 0 40px rgba(0,245,255,0.8),0 0 100px rgba(255,0,255,0.4)}}

.nav-links{display:flex;align-items:center;gap:4px}
.nav-link{
  font-family:'Share Tech Mono',monospace;
  font-size:0.68rem;letter-spacing:2px;
  color:var(--muted);
  text-decoration:none;
  padding:8px 18px;border-radius:6px;
  transition:all 0.2s;
  border:1px solid transparent;
}
.nav-link:hover{color:var(--cyan);border-color:rgba(0,245,255,0.2);background:rgba(0,245,255,0.05)}
.nav-link.active{color:var(--cyan);border-color:rgba(0,245,255,0.3);background:rgba(0,245,255,0.08);text-shadow:0 0 12px var(--cyan)}

.nav-right{display:flex;align-items:center;gap:12px}
.live-pip{
  display:flex;align-items:center;gap:7px;
  font-family:'Share Tech Mono',monospace;
  font-size:0.6rem;letter-spacing:2px;color:var(--green);
  background:rgba(0,255,136,0.07);
  border:1px solid rgba(0,255,136,0.2);
  padding:5px 12px;border-radius:20px;
}
.pip-dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.2}}

.btn{
  font-family:'Share Tech Mono',monospace;
  font-size:0.6rem;letter-spacing:2px;
  border:1px solid rgba(0,245,255,0.3);
  background:rgba(0,245,255,0.06);
  color:var(--cyan);
  padding:7px 18px;border-radius:6px;
  cursor:pointer;transition:all 0.2s;
}
.btn:hover{background:rgba(0,245,255,0.15);box-shadow:0 0 20px rgba(0,245,255,0.3);transform:translateY(-1px)}
.btn:disabled{opacity:0.3;transform:none;cursor:not-allowed}

/* PAGE WRAPPER */
.page{position:relative;z-index:1;max-width:1480px;margin:0 auto;padding:40px 48px 80px}

/* SECTION TITLE */
.section-title{
  font-family:'Rajdhani',sans-serif;
  font-weight:600;font-size:0.65rem;
  letter-spacing:5px;text-transform:uppercase;
  color:var(--muted);margin-bottom:16px;
  display:flex;align-items:center;gap:12px;
}
.section-title::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border2),transparent)}

/* CARDS */
.card{
  background:var(--surface);
  border:1px solid var(--border2);
  border-radius:18px;
  position:relative;overflow:hidden;
  transition:transform 0.3s,box-shadow 0.3s;
}
.card:hover{transform:translateY(-3px)}
.card::before{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,0.02) 0%,transparent 50%);
  pointer-events:none;
}

/* GLOW VARIANTS */
.glow-cyan{box-shadow:0 0 0 1px rgba(0,245,255,0.2);border-color:rgba(0,245,255,0.25)}
.glow-cyan:hover{box-shadow:0 24px 60px rgba(0,245,255,0.12),0 0 0 1px rgba(0,245,255,0.35)}
.glow-mag{box-shadow:0 0 0 1px rgba(255,0,255,0.2);border-color:rgba(255,0,255,0.25)}
.glow-mag:hover{box-shadow:0 24px 60px rgba(255,0,255,0.12),0 0 0 1px rgba(255,0,255,0.35)}
.glow-green{box-shadow:0 0 0 1px rgba(0,255,136,0.2);border-color:rgba(0,255,136,0.25)}
.glow-green:hover{box-shadow:0 24px 60px rgba(0,255,136,0.12),0 0 0 1px rgba(0,255,136,0.35)}
.glow-red{box-shadow:0 0 0 1px rgba(255,34,85,0.2);border-color:rgba(255,34,85,0.25)}
.glow-red:hover{box-shadow:0 24px 60px rgba(255,34,85,0.12),0 0 0 1px rgba(255,34,85,0.35)}

/* TOP EDGE LINE */
.edge-cyan::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent)}
.edge-mag::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--magenta),transparent)}
.edge-green::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--green),transparent)}
.edge-red::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--red),transparent)}
.edge-yellow::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--yellow),transparent)}

/* MONO LABEL */
.mono{font-family:'Share Tech Mono',monospace}

/* LOADING SCREEN */
.loader{
  position:fixed;inset:0;z-index:9999;
  background:var(--bg);
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:32px;
  transition:opacity 0.6s;
}
.loader.gone{opacity:0;pointer-events:none}
.loader-title{
  font-family:'Rajdhani',monospace;font-weight:700;
  font-size:3rem;letter-spacing:12px;
  background:linear-gradient(90deg,var(--cyan),var(--magenta),var(--cyan));
  background-size:200%;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:gradShift 2s linear infinite;
}
@keyframes gradShift{0%{background-position:0%}100%{background-position:200%}}
.loader-track{width:320px;height:2px;background:var(--muted);border-radius:1px;overflow:hidden}
.loader-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));border-radius:1px;animation:sweep 2s ease-in-out infinite}
@keyframes sweep{0%{width:0%;margin-left:0%}50%{width:70%;margin-left:15%}100%{width:0%;margin-left:100%}}
.loader-sub{font-family:'Share Tech Mono',monospace;font-size:0.65rem;letter-spacing:4px;color:var(--muted);animation:textBlink 2s ease-in-out infinite}
@keyframes textBlink{0%,100%{opacity:0.3}50%{opacity:1}}

@keyframes fadeUp{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}
.anim{animation:fadeUp 0.6s ease both}

/* SCROLLBAR */
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--muted);border-radius:2px}

@media(max-width:768px){
  nav{padding:0 16px;height:56px}
  .nav-links{display:none}
  .page{padding:16px 12px 60px}
  .logo-gem{width:28px;height:28px;font-size:0.9rem}
  .nav-logo{font-size:1rem;letter-spacing:2px}
  .live-pip{display:none}
  .btn{font-size:0.55rem;padding:6px 12px}
}
"""

NAV_TEMPLATE = """
<div class="orbs"><div class="orb orb1"></div><div class="orb orb2"></div><div class="orb orb3"></div></div>
<div class="loader" id="loader">
  <div class="loader-title">BTC PULSE</div>
  <div class="loader-track"><div class="loader-fill"></div></div>
  <div class="loader-sub">SCANNING REDDIT · ANALYZING SENTIMENT</div>
</div>
<nav>
  <a href="/" class="nav-logo"><div class="logo-gem">₿</div>BTC PULSE</a>
  <div class="nav-links">
    <a href="/" class="nav-link {d}">◈ DASHBOARD</a>
    <a href="/posts" class="nav-link {p}">◈ POSTS</a>
    <a href="/sources" class="nav-link {s}">◈ SOURCES</a>
    <a href="/about" class="nav-link {a}">◈ ABOUT</a>
  </div>
  <div class="nav-right">
    <div class="live-pip"><div class="pip-dot"></div>LIVE</div>
    <button class="btn" id="refreshBtn" onclick="manualRefresh()">↻ REFRESH</button>
  </div>
</nav>
"""

# ═══════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Pulse — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>{css}</style>
</head><body>
{nav}
<div class="page">

  <!-- HERO GRID -->
  <div style="display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr;gap:18px;margin-bottom:24px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr))" class="anim">

    <!-- BTC Price -->
    <div class="card glow-cyan edge-cyan" style="padding:28px 32px">
      <div class="section-title" style="margin-bottom:18px">Bitcoin Price</div>
      <div id="priceVal" class="mono" style="font-size:2.8rem;font-weight:700;color:var(--orange);text-shadow:0 0 40px rgba(255,140,0,0.5);letter-spacing:-1px">$—</div>
      <div id="priceChg" class="mono" style="margin-top:10px;font-size:0.9rem;display:inline-flex;align-items:center;gap:6px;padding:5px 14px;border-radius:8px;background:rgba(0,255,136,0.08);color:var(--green);border:1px solid rgba(0,255,136,0.2)">—</div>
      <div class="mono" style="margin-top:14px;font-size:0.6rem;letter-spacing:2px;color:var(--muted)">COINGECKO · LIVE</div>
    </div>

    <!-- Overall -->
    <div class="card glow-mag edge-mag" style="padding:28px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center">
      <div class="section-title" style="justify-content:center;margin-bottom:14px">Overall Sentiment</div>
      <div id="overallRing" style="width:80px;height:80px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2rem;margin-bottom:14px;position:relative">
        <div id="ringGlow" style="position:absolute;inset:0;border-radius:50%;animation:spin 3s linear infinite"></div>
        <span id="overallEmoji" style="position:relative;z-index:1">⚡</span>
      </div>
      <div id="overallWord" class="mono" style="font-size:1.3rem;letter-spacing:4px;font-weight:700">LOADING</div>
    </div>

    <!-- Posts -->
    <div class="card glow-green edge-green" style="padding:28px">
      <div class="section-title" style="margin-bottom:18px">Posts Scanned</div>
      <div id="totalVal" class="mono" style="font-size:2.8rem;font-weight:700;color:var(--cyan);text-shadow:0 0 30px rgba(0,245,255,0.4)">—</div>
      <div class="mono" style="margin-top:6px;font-size:0.6rem;letter-spacing:2px;color:var(--muted)">ACROSS 4 SUBREDDITS</div>
      <div style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <div style="background:rgba(0,255,136,0.07);border:1px solid rgba(0,255,136,0.15);border-radius:8px;padding:10px">
          <div id="bullStat" class="mono" style="font-size:1.2rem;color:var(--green)">—</div>
          <div class="mono" style="font-size:0.55rem;letter-spacing:2px;color:var(--muted);margin-top:2px">BULLISH</div>
        </div>
        <div style="background:rgba(255,34,85,0.07);border:1px solid rgba(255,34,85,0.15);border-radius:8px;padding:10px">
          <div id="bearStat" class="mono" style="font-size:1.2rem;color:var(--red)">—</div>
          <div class="mono" style="font-size:0.55rem;letter-spacing:2px;color:var(--muted);margin-top:2px">BEARISH</div>
        </div>
      </div>
    </div>

    <!-- Avg Score -->
    <div class="card glow-cyan edge-yellow" style="padding:28px">
      <div class="section-title" style="margin-bottom:18px">Avg Score</div>
      <div id="avgVal" class="mono" style="font-size:2.8rem;font-weight:700;color:var(--yellow);text-shadow:0 0 30px rgba(255,229,0,0.4)">—</div>
      <div class="mono" style="margin-top:6px;font-size:0.6rem;letter-spacing:2px;color:var(--muted)">VADER COMPOUND</div>
      <div style="margin-top:20px">
        <div id="tsVal" class="mono" style="font-size:0.65rem;color:var(--muted);letter-spacing:1px">LAST UPDATED: —</div>
        <div class="mono" style="font-size:0.6rem;color:var(--muted);letter-spacing:1px;margin-top:4px">AUTO-REFRESH: 5m</div>
      </div>
    </div>
  </div>

  <!-- GAUGE ROW -->
  <div class="section-title anim" style="animation-delay:0.1s">Sentiment Breakdown</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:18px;margin-bottom:24px" class="anim" style="animation-delay:0.15s">

    <!-- Bull -->
    <div class="card edge-green" style="padding:32px;border-color:rgba(0,255,136,0.2);box-shadow:0 0 0 1px rgba(0,255,136,0.1)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
        <span class="mono" style="font-size:0.6rem;letter-spacing:3px;color:var(--green)">● BULLISH</span>
        <span id="bullCnt" class="mono" style="font-size:0.65rem;padding:3px 10px;border-radius:6px;background:rgba(0,255,136,0.1);color:var(--green);border:1px solid rgba(0,255,136,0.2)">—</span>
      </div>
      <div class="mono" style="font-size:5rem;font-weight:700;line-height:1;color:var(--green);text-shadow:0 0 60px rgba(0,255,136,0.5)" id="bullPct">—%</div>
      <div style="margin-top:20px;height:4px;background:var(--muted);border-radius:2px;overflow:visible;position:relative">
        <div id="bullBar" style="height:100%;width:0%;border-radius:2px;background:linear-gradient(90deg,var(--green2),var(--green));box-shadow:0 0 14px var(--green);transition:width 1.2s cubic-bezier(0.4,0,0.2,1);position:relative">
          <div style="position:absolute;right:-1px;top:-5px;width:14px;height:14px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green),0 0 24px var(--green);border:2px solid var(--bg)"></div>
        </div>
      </div>
      <div style="position:absolute;right:20px;bottom:12px;font-size:4rem;opacity:0.05">🚀</div>
    </div>

    <!-- Bear -->
    <div class="card edge-red" style="padding:32px;border-color:rgba(255,34,85,0.2);box-shadow:0 0 0 1px rgba(255,34,85,0.1)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
        <span class="mono" style="font-size:0.6rem;letter-spacing:3px;color:var(--red)">● BEARISH</span>
        <span id="bearCnt" class="mono" style="font-size:0.65rem;padding:3px 10px;border-radius:6px;background:rgba(255,34,85,0.1);color:var(--red);border:1px solid rgba(255,34,85,0.2)">—</span>
      </div>
      <div class="mono" style="font-size:5rem;font-weight:700;line-height:1;color:var(--red);text-shadow:0 0 60px rgba(255,34,85,0.5)" id="bearPct">—%</div>
      <div style="margin-top:20px;height:4px;background:var(--muted);border-radius:2px;overflow:visible;position:relative">
        <div id="bearBar" style="height:100%;width:0%;border-radius:2px;background:linear-gradient(90deg,var(--red2),var(--red));box-shadow:0 0 14px var(--red);transition:width 1.2s cubic-bezier(0.4,0,0.2,1);position:relative">
          <div style="position:absolute;right:-1px;top:-5px;width:14px;height:14px;border-radius:50%;background:var(--red);box-shadow:0 0 12px var(--red),0 0 24px var(--red);border:2px solid var(--bg)"></div>
        </div>
      </div>
      <div style="position:absolute;right:20px;bottom:12px;font-size:4rem;opacity:0.05">📉</div>
    </div>

    <!-- Neut -->
    <div class="card edge-yellow" style="padding:32px;border-color:rgba(255,229,0,0.2);box-shadow:0 0 0 1px rgba(255,229,0,0.1)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
        <span class="mono" style="font-size:0.6rem;letter-spacing:3px;color:var(--yellow)">● NEUTRAL</span>
        <span id="neutCnt" class="mono" style="font-size:0.65rem;padding:3px 10px;border-radius:6px;background:rgba(255,229,0,0.1);color:var(--yellow);border:1px solid rgba(255,229,0,0.2)">—</span>
      </div>
      <div class="mono" style="font-size:5rem;font-weight:700;line-height:1;color:var(--yellow);text-shadow:0 0 60px rgba(255,229,0,0.5)" id="neutPct">—%</div>
      <div style="margin-top:20px;height:4px;background:var(--muted);border-radius:2px;overflow:visible;position:relative">
        <div id="neutBar" style="height:100%;width:0%;border-radius:2px;background:linear-gradient(90deg,var(--yellow2),var(--yellow));box-shadow:0 0 14px var(--yellow);transition:width 1.2s cubic-bezier(0.4,0,0.2,1);position:relative">
          <div style="position:absolute;right:-1px;top:-5px;width:14px;height:14px;border-radius:50%;background:var(--yellow);box-shadow:0 0 12px var(--yellow),0 0 24px var(--yellow);border:2px solid var(--bg)"></div>
        </div>
      </div>
      <div style="position:absolute;right:20px;bottom:12px;font-size:4rem;opacity:0.05">⚖️</div>
    </div>
  </div>

  <!-- BOTTOM -->
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px" class="anim" style="animation-delay:0.2s">
    <div class="card glow-cyan edge-cyan" style="padding:28px">
      <div class="section-title" style="margin-bottom:20px">Distribution</div>
      <div style="position:relative;height:280px"><canvas id="donut"></canvas></div>
    </div>
    <div class="card glow-mag edge-mag" style="padding:28px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
        <div class="section-title" style="margin-bottom:0">Latest Posts Preview</div>
        <a href="/posts" style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;letter-spacing:2px;color:var(--cyan);text-decoration:none;border:1px solid rgba(0,245,255,0.2);padding:5px 12px;border-radius:6px;transition:all 0.2s" onmouseover="this.style.background='rgba(0,245,255,0.08)'" onmouseout="this.style.background='transparent'">VIEW ALL →</a>
      </div>
      <div id="previewList" style="display:flex;flex-direction:column;gap:8px;max-height:300px;overflow-y:auto;padding-right:4px">
        <div class="mono" style="color:var(--muted);text-align:center;padding:40px;font-size:0.7rem;letter-spacing:2px">LOADING...</div>
      </div>
    </div>
  </div>

</div>
<script>
let chart=null;
function spin(){return}

function initChart(b,bear,n){
  const ctx=document.getElementById('donut').getContext('2d');
  if(chart)chart.destroy();
  chart=new Chart(ctx,{
    type:'doughnut',
    data:{labels:['Bullish','Bearish','Neutral'],datasets:[{data:[b,bear,n],backgroundColor:['rgba(0,255,136,0.75)','rgba(255,34,85,0.75)','rgba(255,229,0,0.75)'],borderColor:['#00ff88','#ff2255','#ffe500'],borderWidth:2,hoverOffset:14}]},
    options:{responsive:true,maintainAspectRatio:false,cutout:'70%',animation:{duration:1200,easing:'easeInOutQuart'},plugins:{legend:{position:'bottom',labels:{color:'#2a4060',font:{family:'Share Tech Mono',size:10},padding:20,usePointStyle:true}},tooltip:{backgroundColor:'rgba(3,10,24,0.95)',borderColor:'rgba(0,245,255,0.2)',borderWidth:1,titleFont:{family:'Share Tech Mono',size:10},bodyFont:{family:'Share Tech Mono',size:10},callbacks:{label:c=>'  '+c.parsed+'%'}}}}
  });
}

function updateUI(d){
  const p=d.percentages,c=d.counts,o=d.overall;
  document.getElementById('bullPct').textContent=p.BULLISH+'%';
  document.getElementById('bearPct').textContent=p.BEARISH+'%';
  document.getElementById('neutPct').textContent=p.NEUTRAL+'%';
  document.getElementById('bullCnt').textContent=c.BULLISH+' posts';
  document.getElementById('bearCnt').textContent=c.BEARISH+' posts';
  document.getElementById('neutCnt').textContent=c.NEUTRAL+' posts';
  setTimeout(()=>{
    document.getElementById('bullBar').style.width=p.BULLISH+'%';
    document.getElementById('bearBar').style.width=p.BEARISH+'%';
    document.getElementById('neutBar').style.width=p.NEUTRAL+'%';
  },200);
  if(d.btc_price){
    document.getElementById('priceVal').textContent='$'+d.btc_price.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
    const chg=d.btc_change,el=document.getElementById('priceChg');
    el.textContent=(chg>=0?'▲ +':'▼ ')+chg+'%';
    el.style.color=chg>=0?'var(--green)':'var(--red)';
    el.style.borderColor=chg>=0?'rgba(0,255,136,0.2)':'rgba(255,34,85,0.2)';
    el.style.background=chg>=0?'rgba(0,255,136,0.08)':'rgba(255,34,85,0.08)';
  }
  const emojis={BULLISH:'🚀',BEARISH:'💀',NEUTRAL:'⚖️'};
  const colors={BULLISH:'var(--green)',BEARISH:'var(--red)',NEUTRAL:'var(--yellow)'};
  const glows={BULLISH:'rgba(0,255,136,0.4)',BEARISH:'rgba(255,34,85,0.4)',NEUTRAL:'rgba(255,229,0,0.4)'};
  document.getElementById('overallEmoji').textContent=emojis[o];
  document.getElementById('overallWord').textContent=o;
  document.getElementById('overallWord').style.color=colors[o];
  document.getElementById('overallWord').style.textShadow='0 0 20px '+glows[o];
  document.getElementById('overallRing').style.background='radial-gradient(circle,'+glows[o]+' 0%,transparent 70%)';
  document.getElementById('totalVal').textContent=d.total;
  document.getElementById('avgVal').textContent=d.avg_score;
  document.getElementById('bullStat').textContent=c.BULLISH;
  document.getElementById('bearStat').textContent=c.BEARISH;
  document.getElementById('tsVal').textContent='LAST UPDATED: '+d.last_updated;
  initChart(p.BULLISH,p.BEARISH,p.NEUTRAL);
  const list=document.getElementById('previewList');
  list.innerHTML='';
  d.posts.slice(0,8).forEach((p,i)=>{
    const sc=p.compound>0?'var(--green)':p.compound<0?'var(--red)':'var(--yellow)';
    const tc={BULLISH:'var(--green)',BEARISH:'var(--red)',NEUTRAL:'var(--yellow)'};
    const bg={BULLISH:'rgba(0,255,136,0.07)',BEARISH:'rgba(255,34,85,0.07)',NEUTRAL:'rgba(255,229,0,0.07)'};
    const bc={BULLISH:'rgba(0,255,136,0.15)',BEARISH:'rgba(255,34,85,0.15)',NEUTRAL:'rgba(255,229,0,0.15)'};
    const el=document.createElement('div');
    el.style.cssText='display:flex;align-items:flex-start;gap:10px;background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:10px 12px;animation:fadeUp 0.4s ease '+i*0.04+'s both;transition:all 0.2s;cursor:default';
    el.onmouseover=()=>{el.style.borderColor='rgba(0,245,255,0.2)';el.style.transform='translateX(4px)'};
    el.onmouseout=()=>{el.style.borderColor='var(--border)';el.style.transform='translateX(0)'};
    el.innerHTML='<span style="font-family:Share Tech Mono,monospace;font-size:0.55rem;padding:2px 8px;border-radius:4px;background:'+bg[p.label]+';color:'+tc[p.label]+';border:1px solid '+bc[p.label]+';white-space:nowrap;margin-top:2px;letter-spacing:1px">'+p.label+'</span><div style="flex:1;min-width:0"><div style="font-size:0.8rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--text)">'+p.title+'</div><div style="display:flex;gap:8px;margin-top:4px"><span style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:var(--muted)">'+p.source+'</span><span style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:'+sc+'">'+(p.compound>0?'+':'')+p.compound.toFixed(3)+'</span></div></div>';
    list.appendChild(el);
  });
}

async function fetchData(){const r=await fetch('/api/sentiment');return r.json()}
async function load(showLoader=false){
  if(showLoader)document.getElementById('loader').classList.remove('gone');
  document.getElementById('refreshBtn').disabled=true;
  try{
    let d=await fetchData();
    while(d.loading){await new Promise(r=>setTimeout(r,2500));d=await fetchData()}
    if(d.data)updateUI({...d.data,last_updated:d.last_updated});
  }catch(e){console.error(e)}
  document.getElementById('loader').classList.add('gone');
  document.getElementById('refreshBtn').disabled=false;
}
async function manualRefresh(){
  document.getElementById('refreshBtn').disabled=true;
  await fetch('/api/refresh',{method:'POST'});
  await load(true);
}
setInterval(()=>load(),300000);
load(true);
</script>
</body></html>"""

# ═══════════════════════════════════════════════════════════
#  PAGE 2 — ALL POSTS
# ═══════════════════════════════════════════════════════════
POSTS_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Pulse — Posts</title>
<style>{css}</style>
</head><body>
{nav}
<div class="page">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px" class="anim">
    <div>
      <div class="mono" style="font-size:0.6rem;letter-spacing:4px;color:var(--muted);margin-bottom:8px">◈ ALL POSTS</div>
      <h1 style="font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:white;letter-spacing:2px">Reddit Feed <span style="color:var(--cyan)">Analysis</span></h1>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
      <div style="display:flex;gap:8px" id="filters">
        <button onclick="filter('ALL')"   class="fbtn active" data-f="ALL">ALL</button>
        <button onclick="filter('BULLISH')" class="fbtn bull-btn" data-f="BULLISH">🟢 BULLISH</button>
        <button onclick="filter('BEARISH')" class="fbtn bear-btn" data-f="BEARISH">🔴 BEARISH</button>
        <button onclick="filter('NEUTRAL')" class="fbtn neut-btn" data-f="NEUTRAL">⚪ NEUTRAL</button>
      </div>
    </div>
  </div>

  <div id="postGrid" style="display:flex;flex-direction:column;gap:10px"></div>
  <div id="noData" class="mono" style="display:none;text-align:center;padding:80px;color:var(--muted);font-size:0.7rem;letter-spacing:3px">LOADING POSTS...</div>
</div>

<style>
.fbtn{font-family:'Share Tech Mono',monospace;font-size:0.6rem;letter-spacing:2px;padding:7px 16px;border-radius:6px;border:1px solid var(--border2);background:transparent;color:var(--muted);cursor:pointer;transition:all 0.2s}
.fbtn.active,.fbtn:hover{color:var(--cyan);border-color:rgba(0,245,255,0.3);background:rgba(0,245,255,0.06)}
.bull-btn.active{color:var(--green);border-color:rgba(0,255,136,0.3);background:rgba(0,255,136,0.06)}
.bear-btn.active{color:var(--red);border-color:rgba(255,34,85,0.3);background:rgba(255,34,85,0.06)}
.neut-btn.active{color:var(--yellow);border-color:rgba(255,229,0,0.3);background:rgba(255,229,0,0.06)}
.post-card{
  background:var(--surface);border:1px solid var(--border2);border-radius:14px;
  padding:18px 22px;display:flex;align-items:flex-start;gap:16px;
  transition:all 0.25s;animation:fadeUp 0.4s ease both;
}
.post-card:hover{transform:translateX(6px);border-color:rgba(0,245,255,0.25);background:var(--surface2)}
</style>

<script>
let allPosts=[],currentFilter='ALL';
const tc={BULLISH:'var(--green)',BEARISH:'var(--red)',NEUTRAL:'var(--yellow)'};
const bg={BULLISH:'rgba(0,255,136,0.07)',BEARISH:'rgba(255,34,85,0.07)',NEUTRAL:'rgba(255,229,0,0.07)'};
const bc={BULLISH:'rgba(0,255,136,0.15)',BEARISH:'rgba(255,34,85,0.15)',NEUTRAL:'rgba(255,229,0,0.15)'};

function filter(f){
  currentFilter=f;
  document.querySelectorAll('.fbtn').forEach(b=>{
    b.classList.remove('active');
    if(b.dataset.f===f)b.classList.add('active');
  });
  renderPosts();
}

function renderPosts(){
  const grid=document.getElementById('postGrid');
  const posts=currentFilter==='ALL'?allPosts:allPosts.filter(p=>p.label===currentFilter);
  grid.innerHTML='';
  posts.forEach((p,i)=>{
    const sc=p.compound>0?'var(--green)':p.compound<0?'var(--red)':'var(--yellow)';
    const sv=(p.compound>0?'+':'')+p.compound.toFixed(3);
    const div=document.createElement('div');
    div.className='post-card';
    div.style.animationDelay=(i*0.02)+'s';
    div.innerHTML=`
      <div style="display:flex;flex-direction:column;align-items:center;gap:6px;flex-shrink:0;width:80px">
        <span style="font-family:Share Tech Mono,monospace;font-size:0.55rem;padding:3px 8px;border-radius:5px;background:${bg[p.label]};color:${tc[p.label]};border:1px solid ${bc[p.label]};letter-spacing:1px;text-align:center">${p.label}</span>
        <span style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:${sc}">${sv}</span>
        <span style="font-family:Share Tech Mono,monospace;font-size:0.55rem;color:var(--muted)">↑ ${p.upvotes}</span>
      </div>
      <div style="flex:1;min-width:0;border-left:1px solid var(--border);padding-left:16px">
        <div style="font-size:0.92rem;color:var(--text);line-height:1.5;margin-bottom:8px">${p.title}</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:var(--muted);letter-spacing:1px">${p.source}</div>
      </div>`;
    grid.appendChild(div);
  });
  if(posts.length===0){grid.innerHTML='<div class="mono" style="text-align:center;padding:60px;color:var(--muted);font-size:0.7rem;letter-spacing:3px">NO POSTS FOUND</div>'}
}

async function load(){
  document.getElementById('loader').classList.remove('gone');
  try{
    let d=await (await fetch('/api/sentiment')).json();
    while(d.loading){await new Promise(r=>setTimeout(r,2500));d=await (await fetch('/api/sentiment')).json()}
    if(d.data){allPosts=d.data.posts;renderPosts()}
  }catch(e){console.error(e)}
  document.getElementById('loader').classList.add('gone');
}
async function manualRefresh(){
  document.getElementById('refreshBtn').disabled=true;
  await fetch('/api/refresh',{method:'POST'});
  await load();
  document.getElementById('refreshBtn').disabled=false;
}
load();
</script>
</body></html>"""

# ═══════════════════════════════════════════════════════════
#  PAGE 3 — SOURCES
# ═══════════════════════════════════════════════════════════
SOURCES_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Pulse — Sources</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>{css}</style>
</head><body>
{nav}
<div class="page">
  <div class="anim" style="margin-bottom:32px">
    <div class="mono" style="font-size:0.6rem;letter-spacing:4px;color:var(--muted);margin-bottom:8px">◈ SOURCE BREAKDOWN</div>
    <h1 style="font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:white;letter-spacing:2px">Subreddit <span style="color:var(--magenta)">Intelligence</span></h1>
  </div>

  <div id="subGrid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin-bottom:28px" class="anim"></div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px" class="anim" style="animation-delay:0.15s">
    <div class="card glow-cyan edge-cyan" style="padding:28px">
      <div class="section-title" style="margin-bottom:20px">Sentiment by Source</div>
      <div style="position:relative;height:300px"><canvas id="barChart"></canvas></div>
    </div>
    <div class="card glow-mag edge-mag" style="padding:28px">
      <div class="section-title" style="margin-bottom:20px">Data Sources</div>
      <div style="display:flex;flex-direction:column;gap:14px">
        {src_cards}
      </div>
    </div>
  </div>
</div>

<script>
let barChart=null;
async function load(){
  document.getElementById('loader').classList.remove('gone');
  try{
    let d=await(await fetch('/api/sentiment')).json();
    while(d.loading){await new Promise(r=>setTimeout(r,2500));d=await(await fetch('/api/sentiment')).json()}
    if(d.data)renderSources(d.data);
  }catch(e){console.error(e)}
  document.getElementById('loader').classList.add('gone');
}
function renderSources(data){
  const grid=document.getElementById('subGrid');
  const stats=data.sub_stats||{};
  const colors=['var(--cyan)','var(--magenta)','var(--green)','var(--yellow)'];
  const glowC=['rgba(0,245,255,0.3)','rgba(255,0,255,0.3)','rgba(0,255,136,0.3)','rgba(255,229,0,0.3)'];
  const borderC=['rgba(0,245,255,0.2)','rgba(255,0,255,0.2)','rgba(0,255,136,0.2)','rgba(255,229,0,0.2)'];
  Object.entries(stats).forEach(([sub,s],i)=>{
    const top=s.BULLISH>s.BEARISH?(s.BULLISH>s.NEUTRAL?'BULLISH':'NEUTRAL'):(s.BEARISH>s.NEUTRAL?'BEARISH':'NEUTRAL');
    const topColor={BULLISH:'var(--green)',BEARISH:'var(--red)',NEUTRAL:'var(--yellow)'}[top];
    const card=document.createElement('div');
    card.className='card';card.style.cssText='padding:28px;animation:fadeUp 0.5s ease '+i*0.08+'s both;border-color:'+borderC[i]+';box-shadow:0 0 0 1px '+borderC[i];
    card.innerHTML=`
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px">
        <div>
          <div class="mono" style="font-size:0.6rem;letter-spacing:3px;color:${colors[i]};margin-bottom:6px">SUBREDDIT</div>
          <div style="font-family:Rajdhani,sans-serif;font-size:1.4rem;font-weight:700;color:white">${sub}</div>
        </div>
        <div class="mono" style="font-size:0.65rem;padding:4px 12px;border-radius:6px;background:${topColor === 'var(--green)' ? 'rgba(0,255,136,0.1)' : topColor === 'var(--red)' ? 'rgba(255,34,85,0.1)' : 'rgba(255,229,0,0.1)'};color:${topColor};border:1px solid ${topColor === 'var(--green)' ? 'rgba(0,255,136,0.25)' : topColor === 'var(--red)' ? 'rgba(255,34,85,0.25)' : 'rgba(255,229,0,0.25)'}">${top}</div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">
        ${[['BULLISH','var(--green)','rgba(0,255,136','bull'],['BEARISH','var(--red)','rgba(255,34,85','bear'],['NEUTRAL','var(--yellow)','rgba(255,229,0','neut']].map(([lbl,clr,rgb,cls])=>`
        <div style="background:${rgb},0.07);border:1px solid ${rgb},0.15);border-radius:8px;padding:12px;text-align:center">
          <div class="mono" style="font-size:1.4rem;font-weight:700;color:${clr};text-shadow:0 0 20px ${clr}">${s[lbl]}%</div>
          <div class="mono" style="font-size:0.5rem;letter-spacing:2px;color:var(--muted);margin-top:4px">${lbl}</div>
        </div>`).join('')}
      </div>
      <div style="margin-top:16px;height:3px;background:var(--muted);border-radius:2px;overflow:hidden">
        <div style="display:flex;height:100%">
          <div style="width:${s.BULLISH}%;background:var(--green);transition:width 1s ease"></div>
          <div style="width:${s.BEARISH}%;background:var(--red);transition:width 1s ease"></div>
          <div style="width:${s.NEUTRAL}%;background:var(--yellow);transition:width 1s ease"></div>
        </div>
      </div>`;
    grid.appendChild(card);
  });
  // Bar chart
  const labels=Object.keys(stats);
  const ctx=document.getElementById('barChart').getContext('2d');
  if(barChart)barChart.destroy();
  barChart=new Chart(ctx,{
    type:'bar',
    data:{labels,datasets:[
      {label:'Bullish',data:labels.map(l=>stats[l].BULLISH),backgroundColor:'rgba(0,255,136,0.7)',borderColor:'#00ff88',borderWidth:1,borderRadius:4},
      {label:'Bearish',data:labels.map(l=>stats[l].BEARISH),backgroundColor:'rgba(255,34,85,0.7)',borderColor:'#ff2255',borderWidth:1,borderRadius:4},
      {label:'Neutral',data:labels.map(l=>stats[l].NEUTRAL),backgroundColor:'rgba(255,229,0,0.7)',borderColor:'#ffe500',borderWidth:1,borderRadius:4},
    ]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:1000},scales:{x:{ticks:{color:'#2a4060',font:{family:'Share Tech Mono',size:10}},grid:{color:'rgba(0,245,255,0.04)'}},y:{ticks:{color:'#2a4060',font:{family:'Share Tech Mono',size:10},callback:v=>v+'%'},grid:{color:'rgba(0,245,255,0.04)'},max:100}},plugins:{legend:{labels:{color:'#2a4060',font:{family:'Share Tech Mono',size:10},usePointStyle:true}},tooltip:{backgroundColor:'rgba(3,10,24,0.95)',borderColor:'rgba(0,245,255,0.2)',borderWidth:1,titleFont:{family:'Share Tech Mono',size:10},bodyFont:{family:'Share Tech Mono',size:10},callbacks:{label:c=>c.dataset.label+': '+c.parsed.y+'%'}}}}
  });
}
async function manualRefresh(){
  document.getElementById('refreshBtn').disabled=true;
  await fetch('/api/refresh',{method:'POST'});
  await load();
  document.getElementById('refreshBtn').disabled=false;
}
load();
</script>
</body></html>"""

# ═══════════════════════════════════════════════════════════
#  PAGE 4 — ABOUT
# ═══════════════════════════════════════════════════════════
ABOUT_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Pulse — About</title>
<style>{css}</style>
</head><body>
{nav}
<div class="page" style="max-width:900px">
  <div class="anim" style="margin-bottom:48px">
    <div class="mono" style="font-size:0.6rem;letter-spacing:4px;color:var(--muted);margin-bottom:8px">◈ ABOUT</div>
    <h1 style="font-family:'Rajdhani',sans-serif;font-size:2.5rem;font-weight:700;color:white;letter-spacing:2px;line-height:1.2">How <span style="color:var(--cyan)">BTC Pulse</span><br>Works</h1>
  </div>

  <div style="display:flex;flex-direction:column;gap:16px">
    {cards}
  </div>

  <div class="card glow-cyan edge-cyan anim" style="padding:32px;margin-top:24px;animation-delay:0.5s">
    <div class="mono" style="font-size:0.6rem;letter-spacing:3px;color:var(--muted);margin-bottom:16px">◈ TECH STACK</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px">
      {tech}
    </div>
  </div>
</div>
<script>
document.getElementById('loader').classList.add('gone');
async function manualRefresh(){
  document.getElementById('refreshBtn').disabled=true;
  await fetch('/api/refresh',{method:'POST'});
  document.getElementById('refreshBtn').disabled=false;
}
</script>
</body></html>"""

def make_about_card(i, icon, title, body, color, delay):
    colors = {'cyan': ('rgba(0,245,255,0.2)', 'rgba(0,245,255,0.07)', 'var(--cyan)'),
              'mag':  ('rgba(255,0,255,0.2)',  'rgba(255,0,255,0.07)',  'var(--magenta)'),
              'green':('rgba(0,255,136,0.2)',  'rgba(0,255,136,0.07)', 'var(--green)'),
              'yellow':('rgba(255,229,0,0.2)', 'rgba(255,229,0,0.07)','var(--yellow)')}
    bc,bg,tc = colors[color]
    return f"""<div class="card anim" style="padding:28px;display:flex;gap:20px;align-items:flex-start;border-color:{bc};box-shadow:0 0 0 1px {bc};animation-delay:{delay}s">
      <div style="width:48px;height:48px;border-radius:12px;background:{bg};border:1px solid {bc};display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0">{icon}</div>
      <div>
        <div style="font-family:Rajdhani,sans-serif;font-size:1.1rem;font-weight:700;color:{tc};letter-spacing:2px;margin-bottom:8px">{title}</div>
        <div style="font-size:0.88rem;color:var(--text);line-height:1.7;opacity:0.8">{body}</div>
      </div>
    </div>"""

def make_tech_chip(name, color):
    colors = {'cyan': ('rgba(0,245,255,0.1)', 'rgba(0,245,255,0.25)', 'var(--cyan)'),
              'mag':  ('rgba(255,0,255,0.1)',  'rgba(255,0,255,0.25)',  'var(--magenta)'),
              'green':('rgba(0,255,136,0.1)',  'rgba(0,255,136,0.25)', 'var(--green)'),
              'yellow':('rgba(255,229,0,0.1)', 'rgba(255,229,0,0.25)','var(--yellow)'),
              'orange':('rgba(255,140,0,0.1)', 'rgba(255,140,0,0.25)','var(--orange)')}
    bg, bc, tc = colors[color]
    return f'<div style="background:{bg};border:1px solid {bc};border-radius:8px;padding:12px;text-align:center"><div class="mono" style="font-size:0.7rem;color:{tc};letter-spacing:1px">{name}</div></div>'

ABOUT_CARDS = (
    make_about_card(1,'📡','Data Collection','We scrape the latest posts from 4 Bitcoin subreddits — r/Bitcoin, r/CryptoCurrency, r/BitcoinMarkets, and r/CryptoMarkets — using Reddit\'s public JSON API. No API key required. We collect 25 posts per subreddit, for 100 total.','cyan',0.05) +
    make_about_card(2,'🧠','Sentiment Analysis','Each post is analyzed using VADER (Valence Aware Dictionary and sEntiment Reasoner), an NLP model built specifically for social media. It returns a compound score from -1 (most negative) to +1 (most positive).','mag',0.1) +
    make_about_card(3,'🎯','Crypto Keyword Boosting','We enhance VADER\'s score with a custom crypto keyword layer. Words like "moon", "hodl", and "ath" boost the score bullish, while "crash", "rug", and "liquidation" push it bearish. This makes the model crypto-aware.','green',0.15) +
    make_about_card(4,'📊','Classification','Posts scoring ≥ 0.05 are BULLISH, ≤ -0.05 are BEARISH, and everything in between is NEUTRAL. The overall market sentiment is determined by the average compound score across all 100 posts.','yellow',0.2)
)
TECH_CHIPS = (make_tech_chip('Python','cyan') + make_tech_chip('Flask','mag') +
              make_tech_chip('VADER NLP','green') + make_tech_chip('Chart.js','yellow') +
              make_tech_chip('Reddit JSON','orange') + make_tech_chip('CoinGecko','cyan') +
              make_tech_chip('Requests','mag') + make_tech_chip('Threading','green'))

SRC_CARDS = """
<div style="display:flex;align-items:center;gap:14px;padding:14px;background:var(--bg2);border:1px solid rgba(0,245,255,0.1);border-radius:10px">
  <div style="width:40px;height:40px;border-radius:8px;background:rgba(255,69,0,0.15);border:1px solid rgba(255,69,0,0.3);display:flex;align-items:center;justify-content:center;font-size:1.2rem">🔴</div>
  <div><div class="mono" style="font-size:0.75rem;color:var(--text)">Reddit Public API</div><div class="mono" style="font-size:0.6rem;color:var(--muted);margin-top:3px">reddit.com/r/{sub}/new.json</div></div>
</div>
<div style="display:flex;align-items:center;gap:14px;padding:14px;background:var(--bg2);border:1px solid rgba(0,245,255,0.1);border-radius:10px">
  <div style="width:40px;height:40px;border-radius:8px;background:rgba(0,245,255,0.1);border:1px solid rgba(0,245,255,0.2);display:flex;align-items:center;justify-content:center;font-size:1.2rem">₿</div>
  <div><div class="mono" style="font-size:0.75rem;color:var(--text)">CoinGecko API</div><div class="mono" style="font-size:0.6rem;color:var(--muted);margin-top:3px">Live BTC price · Free tier · No key</div></div>
</div>
<div style="display:flex;align-items:center;gap:14px;padding:14px;background:var(--bg2);border:1px solid rgba(0,245,255,0.1);border-radius:10px">
  <div style="width:40px;height:40px;border-radius:8px;background:rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.2);display:flex;align-items:center;justify-content:center;font-size:1.2rem">🔁</div>
  <div><div class="mono" style="font-size:0.75rem;color:var(--text)">Auto-Refresh Engine</div><div class="mono" style="font-size:0.6rem;color:var(--muted);margin-top:3px">Background thread · Every 5 minutes</div></div>
</div>
"""

def build_page(template, active, **kwargs):
    nav = NAV_TEMPLATE.replace('{d}', 'active' if active=='d' else '') \
                      .replace('{p}', 'active' if active=='p' else '') \
                      .replace('{s}', 'active' if active=='s' else '') \
                      .replace('{a}', 'active' if active=='a' else '')
    result = template.replace('{css}', SHARED_CSS).replace('{nav}', nav)
    for k, v in kwargs.items():
        result = result.replace('{' + k + '}', str(v))
    return result


@app.route("/")
def dashboard():
    return render_template_string(build_page(DASHBOARD_HTML, 'd'))

@app.route("/posts")
def posts():
    return render_template_string(build_page(POSTS_HTML, 'p'))

@app.route("/sources")
def sources():
    return render_template_string(build_page(SOURCES_HTML, 's', src_cards=SRC_CARDS))

@app.route("/about")
def about():
    return render_template_string(build_page(ABOUT_HTML, 'a', cards=ABOUT_CARDS, tech=TECH_CHIPS))

@app.route("/api/sentiment")
def api_sentiment():
    return jsonify({"loading":cache["loading"],"last_updated":cache["last_updated"],"data":cache["data"]})

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    if not cache["loading"]:
        threading.Thread(target=run_analysis, daemon=True).start()
    return jsonify({"status":"started"})

if __name__ == "__main__":
    threading.Thread(target=bg_loop, daemon=True).start()
    print("\n🚀 BTC Pulse → http://localhost:5000\n")
    app.run(debug=False, port=5000)
