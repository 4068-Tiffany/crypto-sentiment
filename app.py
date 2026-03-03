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
HEADERS = {"User-Agent": "crypto-pulse/3.0"}

# ═══════════════════════════════════════════════════════════
#  COIN CONFIG
# ═══════════════════════════════════════════════════════════
COINS = {
    "bitcoin": {
        "symbol": "BTC", "label": "Bitcoin", "emoji": "₿",
        "color": "var(--orange)", "glow": "rgba(255,140,0,0.4)",
        "subreddits": ["Bitcoin","BitcoinMarkets"],
        "coingecko_id": "bitcoin",
        "keywords_bull": ["moon","hodl","ath","accumulate","halving","laser eyes","stack sats","orange coin"],
        "keywords_bear": ["crash","dump","bear","sell","liquidation","rug","ponzi","dead"],
    },
    "ethereum": {
        "symbol": "ETH", "label": "Ethereum", "emoji": "Ξ",
        "color": "var(--purple)", "glow": "rgba(192,132,252,0.4)",
        "subreddits": ["ethereum","ethfinance"],
        "coingecko_id": "ethereum",
        "keywords_bull": ["merge","staking","l2","rollup","defi","nft","flippening","ultrasound"],
        "keywords_bear": ["gas fees","congestion","slow","dump","bear","liquidation","rug"],
    },
    "solana": {
        "symbol": "SOL", "label": "Solana", "emoji": "◎",
        "color": "var(--green)", "glow": "rgba(0,255,136,0.4)",
        "subreddits": ["solana","solanaNFT"],
        "coingecko_id": "solana",
        "keywords_bull": ["fast","cheap","nft","defi","validator","season","pump","meme"],
        "keywords_bear": ["outage","down","hack","dump","bear","centralized","slow"],
    },
    "tether": {
        "symbol": "USDT", "label": "Tether", "emoji": "₮",
        "color": "var(--accent4)", "glow": "rgba(107,203,255,0.4)",
        "subreddits": ["Tether","CryptoCurrency"],
        "coingecko_id": "tether",
        "keywords_bull": ["stable","peg","trusted","reserve","backed","safe haven"],
        "keywords_bear": ["depeg","unbacked","fraud","lawsuit","investigation","fud","collapse"],
    },
    "binancecoin": {
        "symbol": "BNB", "label": "BNB", "emoji": "⬡",
        "color": "var(--yellow)", "glow": "rgba(255,229,0,0.4)",
        "subreddits": ["binance","BinanceSmartChain"],
        "coingecko_id": "binancecoin",
        "keywords_bull": ["burn","bnb chain","launchpad","cz","defi","ecosystem","stake"],
        "keywords_bear": ["sec","lawsuit","regulated","dump","bear","centralized","hack"],
    },
}

SUBREDDITS_GENERAL = ["CryptoCurrency","CryptoMarkets"]
POSTS_PER_SUB = 20
POST_SORT     = "new"
REFRESH_SECS  = 300

# ═══════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════
def fetch_sub(sub):
    try:
        r = requests.get(
            f"https://www.reddit.com/r/{sub}/{POST_SORT}.json?limit={POSTS_PER_SUB}",
            headers=HEADERS, timeout=10)
        return [{"source": f"r/{sub}",
                 "title": c["data"].get("title",""),
                 "text": (c["data"].get("title","")+" "+c["data"].get("selftext","")).strip(),
                 "upvotes": c["data"].get("score",0)}
                for c in r.json()["data"]["children"]]
    except: return []

def get_prices():
    ids = ",".join(c["coingecko_id"] for c in COINS.values())
    try:
        data = requests.get(
            f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true&include_7d_change=true",
            timeout=10).json()
        result = {}
        for coin_key, cfg in COINS.items():
            cg = cfg["coingecko_id"]
            if cg in data:
                result[coin_key] = {
                    "price":    data[cg].get("usd"),
                    "change24": round(data[cg].get("usd_24h_change", 0), 2),
                    "change7d": round(data[cg].get("usd_7d_change", 0) if "usd_7d_change" in data[cg] else 0, 2),
                }
        return result
    except Exception as e:
        print("Price fetch error:", e)
        return {}

def classify(s):
    return "BULLISH" if s >= 0.05 else "BEARISH" if s <= -0.05 else "NEUTRAL"

# ═══════════════════════════════════════════════════════════
#  PREDICTION ENGINE
#  Combines: sentiment score + 24h momentum + 7d trend
#  Returns a directional signal + confidence + target range
# ═══════════════════════════════════════════════════════════
def predict_price(coin_key, sentiment_score, price_data):
    cfg  = COINS[coin_key]
    sym  = cfg["symbol"]
    price = price_data.get("price")
    if not price:
        return None

    ch24  = price_data.get("change24", 0)
    ch7d  = price_data.get("change7d", 0)

    # Weighted signal: 50% sentiment, 30% 24h momentum, 20% 7d trend
    sent_signal     = sentiment_score * 100        # -100 to +100
    momentum_signal = max(-100, min(100, ch24 * 5))
    trend_signal    = max(-100, min(100, ch7d * 2))

    combined = (sent_signal * 0.50) + (momentum_signal * 0.30) + (trend_signal * 0.20)

    # Confidence: how aligned are the three signals?
    signs = [1 if x > 0 else -1 if x < 0 else 0 for x in [sent_signal, momentum_signal, trend_signal]]
    agreement = abs(sum(signs))   # 0,1,2,3
    confidence_map = {0: 30, 1: 45, 2: 65, 3: 82}
    confidence = confidence_map[agreement]

    # Price targets (24h)
    base_move_pct = abs(combined) / 1000   # e.g. combined=50 → 5% max move scaled to 0.05
    max_pct  = min(base_move_pct * 1.5, 0.12)
    min_pct  = base_move_pct * 0.5

    if combined > 2:
        direction = "UP"
        target_hi = round(price * (1 + max_pct), 2)
        target_lo = round(price * (1 + min_pct), 2)
    elif combined < -2:
        direction = "DOWN"
        target_hi = round(price * (1 - min_pct), 2)
        target_lo = round(price * (1 - max_pct), 2)
    else:
        direction = "SIDEWAYS"
        target_hi = round(price * 1.005, 2)
        target_lo = round(price * 0.995, 2)

    return {
        "direction":   direction,
        "confidence":  confidence,
        "target_lo":   target_lo,
        "target_hi":   target_hi,
        "combined":    round(combined, 1),
        "sent_signal": round(sent_signal, 1),
        "mom_signal":  round(momentum_signal, 1),
        "trend_signal":round(trend_signal, 1),
        "current_price": price,
    }

# ═══════════════════════════════════════════════════════════
#  ANALYSIS LOOP
# ═══════════════════════════════════════════════════════════
def run_analysis():
    cache["loading"] = True
    prices = get_prices()
    coin_results = {}

    for coin_key, cfg in COINS.items():
        subs = cfg["subreddits"] + SUBREDDITS_GENERAL
        posts = []
        for sub in subs:
            posts.extend(fetch_sub(sub))
            time.sleep(0.6)

        # Filter posts relevant to this coin
        coin_terms = [cfg["symbol"].lower(), cfg["label"].lower()] + \
                     [k.lower() for k in cfg["keywords_bull"] + cfg["keywords_bear"]]
        relevant = [p for p in posts if any(t in p["text"].lower() for t in coin_terms)]
        if not relevant:
            relevant = posts[:30]  # fallback: use general posts

        results  = analyze_posts(relevant)
        summary  = summarize(results)
        pdata    = prices.get(coin_key, {})
        prediction = predict_price(coin_key, summary["avg_score"], pdata)

        sub_stats = {}
        for sub in subs:
            subs_r = [r for r in results if r["source"] == f"r/{sub}"]
            if subs_r:
                sc = {"BULLISH":0,"BEARISH":0,"NEUTRAL":0}
                for r in subs_r: sc[r["label"]] += 1
                sub_stats[f"r/{sub}"] = {k: round(v/len(subs_r)*100,1) for k,v in sc.items()}

        coin_results[coin_key] = {
            "symbol":      cfg["symbol"],
            "label":       cfg["label"],
            "emoji":       cfg["emoji"],
            "color":       cfg["color"],
            "total":       summary["total"],
            "counts":      summary["counts"],
            "percentages": summary["percentages"],
            "avg_score":   summary["avg_score"],
            "overall":     summary["overall"],
            "method":      summary["method"],
            "price":       pdata.get("price"),
            "change24":    pdata.get("change24"),
            "change7d":    pdata.get("change7d"),
            "prediction":  prediction,
            "sub_stats":   sub_stats,
            "posts":       [{"source":r["source"],"title":r["title"],"label":r["label"],
                             "compound":r["compound"],"upvotes":r["upvotes"]} for r in results],
        }
        print(f"✅ {cfg['symbol']}: {summary['total']} posts — {summary['overall']}")

    # Legacy BTC cache key for backward compat
    btc = coin_results.get("bitcoin", {})
    cache["data"] = {
        # legacy flat keys (dashboard page uses these)
        "total":       btc.get("total"),
        "counts":      btc.get("counts"),
        "percentages": btc.get("percentages"),
        "avg_score":   btc.get("avg_score"),
        "overall":     btc.get("overall"),
        "btc_price":   btc.get("price"),
        "btc_change":  btc.get("change24"),
        "sub_stats":   btc.get("sub_stats"),
        "method":      btc.get("method"),
        "posts":       btc.get("posts"),
        # new multi-coin data
        "coins":       coin_results,
    }
    cache["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cache["loading"] = False

def bg_loop():
    while True: run_analysis(); time.sleep(REFRESH_SECS)

# ═══════════════════════════════════════════════════════════
#  SHARED CSS + NAV  (unchanged from original)
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
  --purple:   #c084fc;
  --accent4:  #6bcbff;
  --text:     #b0d0f0;
  --muted:    #2a4060;
  --border:   rgba(0,245,255,0.08);
  --border2:  rgba(0,245,255,0.18);
}

html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'Exo 2',sans-serif;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;z-index:0;background-image:linear-gradient(rgba(0,245,255,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,245,255,0.025) 1px,transparent 1px);background-size:60px 60px;pointer-events:none}
body::after{content:'';position:fixed;inset:0;z-index:9998;background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.04) 3px,rgba(0,0,0,0.04) 4px);pointer-events:none}

.orbs{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.orb{position:absolute;border-radius:50%;filter:blur(140px);animation:orb 14s ease-in-out infinite}
.orb1{width:700px;height:700px;background:rgba(0,245,255,0.06);top:-200px;left:-200px;animation-delay:0s}
.orb2{width:600px;height:600px;background:rgba(255,0,255,0.05);bottom:-200px;right:-150px;animation-delay:-7s}
.orb3{width:400px;height:400px;background:rgba(0,255,136,0.04);top:50%;left:50%;transform:translate(-50%,-50%);animation-delay:-3.5s}
@keyframes orb{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(40px,-50px) scale(1.1)}66%{transform:translate(-30px,40px) scale(0.9)}}

nav{position:sticky;top:0;z-index:100;height:64px;display:flex;align-items:center;justify-content:space-between;padding:0 48px;background:rgba(1,5,15,0.85);backdrop-filter:blur(24px);border-bottom:1px solid var(--border2);animation:navIn 0.5s ease both}
@keyframes navIn{from{transform:translateY(-100%);opacity:0}to{transform:translateY(0);opacity:1}}
.nav-logo{font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.4rem;letter-spacing:4px;color:white;display:flex;align-items:center;gap:12px;text-decoration:none}
.logo-gem{width:36px;height:36px;border-radius:8px;background:linear-gradient(135deg,var(--cyan),var(--magenta));display:flex;align-items:center;justify-content:center;font-size:1.1rem;box-shadow:0 0 24px rgba(0,245,255,0.5),0 0 60px rgba(255,0,255,0.2);animation:gemPulse 2.5s ease-in-out infinite}
@keyframes gemPulse{0%,100%{box-shadow:0 0 24px rgba(0,245,255,0.5),0 0 60px rgba(255,0,255,0.2)}50%{box-shadow:0 0 40px rgba(0,245,255,0.8),0 0 100px rgba(255,0,255,0.4)}}
.nav-links{display:flex;align-items:center;gap:4px}
.nav-link{font-family:'Share Tech Mono',monospace;font-size:0.68rem;letter-spacing:2px;color:var(--muted);text-decoration:none;padding:8px 18px;border-radius:6px;transition:all 0.2s;border:1px solid transparent}
.nav-link:hover{color:var(--cyan);border-color:rgba(0,245,255,0.2);background:rgba(0,245,255,0.05)}
.nav-link.active{color:var(--cyan);border-color:rgba(0,245,255,0.3);background:rgba(0,245,255,0.08);text-shadow:0 0 12px var(--cyan)}
.nav-right{display:flex;align-items:center;gap:12px}
.live-pip{display:flex;align-items:center;gap:7px;font-family:'Share Tech Mono',monospace;font-size:0.6rem;letter-spacing:2px;color:var(--green);background:rgba(0,255,136,0.07);border:1px solid rgba(0,255,136,0.2);padding:5px 12px;border-radius:20px}
.pip-dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.2}}
.btn{font-family:'Share Tech Mono',monospace;font-size:0.6rem;letter-spacing:2px;border:1px solid rgba(0,245,255,0.3);background:rgba(0,245,255,0.06);color:var(--cyan);padding:7px 18px;border-radius:6px;cursor:pointer;transition:all 0.2s}
.btn:hover{background:rgba(0,245,255,0.15);box-shadow:0 0 20px rgba(0,245,255,0.3);transform:translateY(-1px)}
.btn:disabled{opacity:0.3;transform:none;cursor:not-allowed}

.page{position:relative;z-index:1;max-width:1480px;margin:0 auto;padding:40px 48px 80px}
.section-title{font-family:'Rajdhani',sans-serif;font-weight:600;font-size:0.65rem;letter-spacing:5px;text-transform:uppercase;color:var(--muted);margin-bottom:16px;display:flex;align-items:center;gap:12px}
.section-title::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border2),transparent)}
.card{background:var(--surface);border:1px solid var(--border2);border-radius:18px;position:relative;overflow:hidden;transition:transform 0.3s,box-shadow 0.3s}
.card:hover{transform:translateY(-3px)}
.card::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(255,255,255,0.02) 0%,transparent 50%);pointer-events:none}

.glow-cyan{box-shadow:0 0 0 1px rgba(0,245,255,0.2);border-color:rgba(0,245,255,0.25)}
.glow-cyan:hover{box-shadow:0 24px 60px rgba(0,245,255,0.12),0 0 0 1px rgba(0,245,255,0.35)}
.glow-mag{box-shadow:0 0 0 1px rgba(255,0,255,0.2);border-color:rgba(255,0,255,0.25)}
.glow-mag:hover{box-shadow:0 24px 60px rgba(255,0,255,0.12),0 0 0 1px rgba(255,0,255,0.35)}
.glow-green{box-shadow:0 0 0 1px rgba(0,255,136,0.2);border-color:rgba(0,255,136,0.25)}
.glow-green:hover{box-shadow:0 24px 60px rgba(0,255,136,0.12),0 0 0 1px rgba(0,255,136,0.35)}
.glow-red{box-shadow:0 0 0 1px rgba(255,34,85,0.2);border-color:rgba(255,34,85,0.25)}
.glow-red:hover{box-shadow:0 24px 60px rgba(255,34,85,0.12),0 0 0 1px rgba(255,34,85,0.35)}

.edge-cyan::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent)}
.edge-mag::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--magenta),transparent)}
.edge-green::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--green),transparent)}
.edge-red::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--red),transparent)}
.edge-yellow::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--yellow),transparent)}
.edge-purple::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--purple),transparent)}

.mono{font-family:'Share Tech Mono',monospace}
.loader{position:fixed;inset:0;z-index:9999;background:var(--bg);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:32px;transition:opacity 0.6s}
.loader.gone{opacity:0;pointer-events:none}
.loader-title{font-family:'Rajdhani',monospace;font-weight:700;font-size:3rem;letter-spacing:12px;background:linear-gradient(90deg,var(--cyan),var(--magenta),var(--cyan));background-size:200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:gradShift 2s linear infinite}
@keyframes gradShift{0%{background-position:0%}100%{background-position:200%}}
.loader-track{width:320px;height:2px;background:var(--muted);border-radius:1px;overflow:hidden}
.loader-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));border-radius:1px;animation:sweep 2s ease-in-out infinite}
@keyframes sweep{0%{width:0%;margin-left:0%}50%{width:70%;margin-left:15%}100%{width:0%;margin-left:100%}}
.loader-sub{font-family:'Share Tech Mono',monospace;font-size:0.65rem;letter-spacing:4px;color:var(--muted);animation:textBlink 2s ease-in-out infinite}
@keyframes textBlink{0%,100%{opacity:0.3}50%{opacity:1}}
@keyframes fadeUp{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}
.anim{animation:fadeUp 0.6s ease both}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--muted);border-radius:2px}

@media(max-width:768px){
  nav{padding:0 16px;height:56px}
  .nav-links{display:none}
  .hamburger{display:flex!important}
  .mobile-menu{display:none;position:fixed;top:56px;left:0;right:0;background:rgba(1,5,15,0.98);border-bottom:1px solid var(--border2);padding:16px;flex-direction:column;gap:8px;z-index:99;backdrop-filter:blur(20px)}
  .mobile-menu.open{display:flex}
  .mobile-menu a{font-family:'Share Tech Mono',monospace;font-size:0.75rem;letter-spacing:3px;color:var(--muted);text-decoration:none;padding:14px 16px;border-radius:8px;border:1px solid var(--border);transition:all 0.2s}
  .mobile-menu a:hover,.mobile-menu a.active{color:var(--cyan);border-color:rgba(0,245,255,0.3);background:rgba(0,245,255,0.06)}
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
  <div class="loader-title">CRYPTO PULSE</div>
  <div class="loader-track"><div class="loader-fill"></div></div>
  <div class="loader-sub">SCANNING REDDIT · ANALYZING SENTIMENT</div>
</div>
<nav>
  <a href="/" class="nav-logo"><div class="logo-gem">₿</div>CRYPTO PULSE</a>
  <div class="nav-links">
    <a href="/" class="nav-link {d}">◈ DASHBOARD</a>
    <a href="/coins" class="nav-link {co}">◈ COINS</a>
    <a href="/predict" class="nav-link {pr}">◈ PREDICT</a>
    <a href="/posts" class="nav-link {p}">◈ POSTS</a>
    <a href="/about" class="nav-link {a}">◈ ABOUT</a>
  </div>
  <div class="nav-right">
    <div class="live-pip"><div class="pip-dot"></div>LIVE</div>
    <button class="btn" id="refreshBtn" onclick="manualRefresh()">↻ REFRESH</button>
    <button class="hamburger" onclick="toggleMenu()" style="display:none;background:transparent;border:1px solid var(--border2);color:var(--cyan);padding:7px 12px;border-radius:6px;cursor:pointer;font-size:1.1rem">☰</button>
  </div>
</nav>
<div class="mobile-menu" id="mobileMenu">
  <a href="/" class="{d}">◈ DASHBOARD</a>
  <a href="/coins" class="{co}">◈ COINS</a>
  <a href="/predict" class="{pr}">◈ PREDICT</a>
  <a href="/posts" class="{p}">◈ POSTS</a>
  <a href="/about" class="{a}">◈ ABOUT</a>
</div>
<script>function toggleMenu(){{document.getElementById('mobileMenu').classList.toggle('open')}}</script>
"""

# ═══════════════════════════════════════════════════════════
#  PAGE — DASHBOARD (BTC focus, unchanged feel)
# ═══════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Pulse — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>{css}</style>
</head><body>
{nav}
<div class="page">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px;margin-bottom:24px" class="anim">
    <div class="card glow-cyan edge-cyan" style="padding:28px 32px">
      <div class="section-title" style="margin-bottom:18px">Bitcoin Price</div>
      <div id="priceVal" class="mono" style="font-size:2.8rem;font-weight:700;color:var(--orange);text-shadow:0 0 40px rgba(255,140,0,0.5);letter-spacing:-1px">$—</div>
      <div id="priceChg" class="mono" style="margin-top:10px;font-size:0.9rem;display:inline-flex;align-items:center;gap:6px;padding:5px 14px;border-radius:8px;background:rgba(0,255,136,0.08);color:var(--green);border:1px solid rgba(0,255,136,0.2)">—</div>
      <div class="mono" style="margin-top:14px;font-size:0.6rem;letter-spacing:2px;color:var(--muted)">COINGECKO · LIVE</div>
    </div>
    <div class="card glow-mag edge-mag" style="padding:28px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center">
      <div class="section-title" style="justify-content:center;margin-bottom:14px">BTC Sentiment</div>
      <div id="overallEmoji" style="font-size:2.5rem;margin-bottom:10px">⚡</div>
      <div id="overallWord" class="mono" style="font-size:1.3rem;letter-spacing:4px;font-weight:700">LOADING</div>
    </div>
    <div class="card glow-green edge-green" style="padding:28px">
      <div class="section-title" style="margin-bottom:18px">Posts Scanned</div>
      <div id="totalVal" class="mono" style="font-size:2.8rem;font-weight:700;color:var(--cyan);text-shadow:0 0 30px rgba(0,245,255,0.4)">—</div>
      <div class="mono" style="margin-top:6px;font-size:0.6rem;letter-spacing:2px;color:var(--muted)">BTC SUBREDDITS</div>
      <div style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <div style="background:rgba(0,255,136,0.07);border:1px solid rgba(0,255,136,0.15);border-radius:8px;padding:10px"><div id="bullStat" class="mono" style="font-size:1.2rem;color:var(--green)">—</div><div class="mono" style="font-size:0.55rem;letter-spacing:2px;color:var(--muted);margin-top:2px">BULLISH</div></div>
        <div style="background:rgba(255,34,85,0.07);border:1px solid rgba(255,34,85,0.15);border-radius:8px;padding:10px"><div id="bearStat" class="mono" style="font-size:1.2rem;color:var(--red)">—</div><div class="mono" style="font-size:0.55rem;letter-spacing:2px;color:var(--muted);margin-top:2px">BEARISH</div></div>
      </div>
    </div>
    <div class="card glow-cyan edge-yellow" style="padding:28px">
      <div class="section-title" style="margin-bottom:18px">Avg Score</div>
      <div id="avgVal" class="mono" style="font-size:2.8rem;font-weight:700;color:var(--yellow);text-shadow:0 0 30px rgba(255,229,0,0.4)">—</div>
      <div class="mono" style="margin-top:6px;font-size:0.6rem;letter-spacing:2px;color:var(--muted)">VADER COMPOUND</div>
      <div style="margin-top:20px"><div id="tsVal" class="mono" style="font-size:0.65rem;color:var(--muted);letter-spacing:1px">LAST UPDATED: —</div></div>
    </div>
  </div>

  <!-- Quick coin ticker -->
  <div class="section-title anim" style="animation-delay:0.05s">All Coins — Quick View</div>
  <div id="coinTicker" style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px" class="anim" style="animation-delay:0.1s"></div>

  <div class="section-title anim" style="animation-delay:0.1s">Sentiment Breakdown</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:18px;margin-bottom:24px" class="anim" style="animation-delay:0.15s">
    <div class="card edge-green" style="padding:32px;border-color:rgba(0,255,136,0.2)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
        <span class="mono" style="font-size:0.6rem;letter-spacing:3px;color:var(--green)">● BULLISH</span>
        <span id="bullCnt" class="mono" style="font-size:0.65rem;padding:3px 10px;border-radius:6px;background:rgba(0,255,136,0.1);color:var(--green);border:1px solid rgba(0,255,136,0.2)">—</span>
      </div>
      <div class="mono" style="font-size:5rem;font-weight:700;line-height:1;color:var(--green);text-shadow:0 0 60px rgba(0,255,136,0.5)" id="bullPct">—%</div>
      <div style="margin-top:20px;height:4px;background:var(--muted);border-radius:2px;overflow:visible;position:relative">
        <div id="bullBar" style="height:100%;width:0%;border-radius:2px;background:linear-gradient(90deg,var(--green2),var(--green));box-shadow:0 0 14px var(--green);transition:width 1.2s cubic-bezier(0.4,0,0.2,1);position:relative">
          <div style="position:absolute;right:-1px;top:-5px;width:14px;height:14px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green);border:2px solid var(--bg)"></div>
        </div>
      </div>
    </div>
    <div class="card edge-red" style="padding:32px;border-color:rgba(255,34,85,0.2)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
        <span class="mono" style="font-size:0.6rem;letter-spacing:3px;color:var(--red)">● BEARISH</span>
        <span id="bearCnt" class="mono" style="font-size:0.65rem;padding:3px 10px;border-radius:6px;background:rgba(255,34,85,0.1);color:var(--red);border:1px solid rgba(255,34,85,0.2)">—</span>
      </div>
      <div class="mono" style="font-size:5rem;font-weight:700;line-height:1;color:var(--red);text-shadow:0 0 60px rgba(255,34,85,0.5)" id="bearPct">—%</div>
      <div style="margin-top:20px;height:4px;background:var(--muted);border-radius:2px;overflow:visible;position:relative">
        <div id="bearBar" style="height:100%;width:0%;border-radius:2px;background:linear-gradient(90deg,var(--red2),var(--red));box-shadow:0 0 14px var(--red);transition:width 1.2s cubic-bezier(0.4,0,0.2,1);position:relative">
          <div style="position:absolute;right:-1px;top:-5px;width:14px;height:14px;border-radius:50%;background:var(--red);box-shadow:0 0 12px var(--red);border:2px solid var(--bg)"></div>
        </div>
      </div>
    </div>
    <div class="card edge-yellow" style="padding:32px;border-color:rgba(255,229,0,0.2)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
        <span class="mono" style="font-size:0.6rem;letter-spacing:3px;color:var(--yellow)">● NEUTRAL</span>
        <span id="neutCnt" class="mono" style="font-size:0.65rem;padding:3px 10px;border-radius:6px;background:rgba(255,229,0,0.1);color:var(--yellow);border:1px solid rgba(255,229,0,0.2)">—</span>
      </div>
      <div class="mono" style="font-size:5rem;font-weight:700;line-height:1;color:var(--yellow);text-shadow:0 0 60px rgba(255,229,0,0.5)" id="neutPct">—%</div>
      <div style="margin-top:20px;height:4px;background:var(--muted);border-radius:2px;overflow:visible;position:relative">
        <div id="neutBar" style="height:100%;width:0%;border-radius:2px;background:linear-gradient(90deg,var(--yellow2),var(--yellow));box-shadow:0 0 14px var(--yellow);transition:width 1.2s cubic-bezier(0.4,0,0.2,1);position:relative">
          <div style="position:absolute;right:-1px;top:-5px;width:14px;height:14px;border-radius:50%;background:var(--yellow);box-shadow:0 0 12px var(--yellow);border:2px solid var(--bg)"></div>
        </div>
      </div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px" class="anim" style="animation-delay:0.2s">
    <div class="card glow-cyan edge-cyan" style="padding:28px">
      <div class="section-title" style="margin-bottom:20px">BTC Distribution</div>
      <div style="position:relative;height:280px"><canvas id="donut"></canvas></div>
    </div>
    <div class="card glow-mag edge-mag" style="padding:28px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
        <div class="section-title" style="margin-bottom:0">Latest Posts</div>
        <a href="/posts" style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;letter-spacing:2px;color:var(--cyan);text-decoration:none;border:1px solid rgba(0,245,255,0.2);padding:5px 12px;border-radius:6px">VIEW ALL →</a>
      </div>
      <div id="previewList" style="display:flex;flex-direction:column;gap:8px;max-height:300px;overflow-y:auto;padding-right:4px">
        <div class="mono" style="color:var(--muted);text-align:center;padding:40px;font-size:0.7rem;letter-spacing:2px">LOADING...</div>
      </div>
    </div>
  </div>
</div>

<script>
let chart=null;
function initChart(b,bear,n){
  const ctx=document.getElementById('donut').getContext('2d');
  if(chart)chart.destroy();
  chart=new Chart(ctx,{type:'doughnut',data:{labels:['Bullish','Bearish','Neutral'],datasets:[{data:[b,bear,n],backgroundColor:['rgba(0,255,136,0.75)','rgba(255,34,85,0.75)','rgba(255,229,0,0.75)'],borderColor:['#00ff88','#ff2255','#ffe500'],borderWidth:2,hoverOffset:14}]},options:{responsive:true,maintainAspectRatio:false,cutout:'70%',plugins:{legend:{position:'bottom',labels:{color:'#2a4060',font:{family:'Share Tech Mono',size:10},padding:20,usePointStyle:true}},tooltip:{backgroundColor:'rgba(3,10,24,0.95)',borderColor:'rgba(0,245,255,0.2)',borderWidth:1,callbacks:{label:c=>'  '+c.parsed+'%'}}}}});
}

function renderTicker(coins){
  const wrap=document.getElementById('coinTicker');
  wrap.innerHTML='';
  Object.entries(coins).forEach(([k,c])=>{
    const up=(c.change24||0)>=0;
    const el=document.createElement('a');
    el.href='/coins#'+k;
    el.style.cssText='display:flex;align-items:center;gap:10px;background:var(--surface);border:1px solid var(--border2);border-radius:12px;padding:12px 18px;text-decoration:none;transition:all 0.2s;cursor:pointer';
    el.onmouseover=()=>{el.style.borderColor='rgba(0,245,255,0.25)';el.style.transform='translateY(-2px)'};
    el.onmouseout=()=>{el.style.borderColor='var(--border2)';el.style.transform='translateY(0)'};
    el.innerHTML=`<span style="font-size:1.3rem">${c.emoji}</span><div><div class="mono" style="font-size:0.75rem;color:white;font-weight:700">${c.symbol}</div><div class="mono" style="font-size:0.6rem;color:var(--muted)">${c.label}</div></div><div style="text-align:right"><div class="mono" style="font-size:0.8rem;color:var(--text)">${c.price?'$'+c.price.toLocaleString('en-US',{maximumFractionDigits:c.price>100?2:4}):'—'}</div><div class="mono" style="font-size:0.65rem;color:${up?'var(--green)':'var(--red)'}">${up?'▲ +':'▼ '}${c.change24||0}%</div></div>`;
    wrap.appendChild(el);
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
  setTimeout(()=>{document.getElementById('bullBar').style.width=p.BULLISH+'%';document.getElementById('bearBar').style.width=p.BEARISH+'%';document.getElementById('neutBar').style.width=p.NEUTRAL+'%'},200);
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
  document.getElementById('overallEmoji').textContent=emojis[o];
  document.getElementById('overallWord').textContent=o;
  document.getElementById('overallWord').style.color=colors[o];
  document.getElementById('totalVal').textContent=d.total;
  document.getElementById('avgVal').textContent=d.avg_score;
  document.getElementById('bullStat').textContent=c.BULLISH;
  document.getElementById('bearStat').textContent=c.BEARISH;
  document.getElementById('tsVal').textContent='LAST UPDATED: '+(d.last_updated||'—');
  initChart(p.BULLISH,p.BEARISH,p.NEUTRAL);
  if(d.coins)renderTicker(d.coins);
  const list=document.getElementById('previewList');
  list.innerHTML='';
  (d.posts||[]).slice(0,8).forEach((p,i)=>{
    const sc=p.compound>0?'var(--green)':p.compound<0?'var(--red)':'var(--yellow)';
    const tc={BULLISH:'var(--green)',BEARISH:'var(--red)',NEUTRAL:'var(--yellow)'};
    const bg={BULLISH:'rgba(0,255,136,0.07)',BEARISH:'rgba(255,34,85,0.07)',NEUTRAL:'rgba(255,229,0,0.07)'};
    const bc={BULLISH:'rgba(0,255,136,0.15)',BEARISH:'rgba(255,34,85,0.15)',NEUTRAL:'rgba(255,229,0,0.15)'};
    const el=document.createElement('div');
    el.style.cssText='display:flex;align-items:flex-start;gap:10px;background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:10px 12px;animation:fadeUp 0.4s ease '+i*0.04+'s both';
    el.innerHTML='<span style="font-family:Share Tech Mono,monospace;font-size:0.55rem;padding:2px 8px;border-radius:4px;background:'+bg[p.label]+';color:'+tc[p.label]+';border:1px solid '+bc[p.label]+';white-space:nowrap;margin-top:2px">'+p.label+'</span><div style="flex:1;min-width:0"><div style="font-size:0.8rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'+p.title+'</div><div style="display:flex;gap:8px;margin-top:4px"><span style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:var(--muted)">'+p.source+'</span><span style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:'+sc+'">'+(p.compound>0?'+':'')+p.compound.toFixed(3)+'</span></div></div>';
    list.appendChild(el);
  });
}
async function load(showLoader=false){
  if(showLoader)document.getElementById('loader').classList.remove('gone');
  document.getElementById('refreshBtn').disabled=true;
  try{
    let d=await(await fetch('/api/sentiment')).json();
    while(d.loading){await new Promise(r=>setTimeout(r,2500));d=await(await fetch('/api/sentiment')).json()}
    if(d.data)updateUI({...d.data,last_updated:d.last_updated});
  }catch(e){console.error(e)}
  document.getElementById('loader').classList.add('gone');
  document.getElementById('refreshBtn').disabled=false;
}
async function manualRefresh(){document.getElementById('refreshBtn').disabled=true;await fetch('/api/refresh',{method:'POST'});await load(true)}
setInterval(()=>load(),300000);
load(true);
</script>
</body></html>"""

# ═══════════════════════════════════════════════════════════
#  PAGE — COINS (all coins sentiment grid)
# ═══════════════════════════════════════════════════════════
COINS_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Pulse — Coins</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>{css}
.coin-card{{background:var(--surface);border:1px solid var(--border2);border-radius:18px;padding:28px;position:relative;overflow:hidden;transition:all 0.3s;cursor:pointer}}
.coin-card:hover{{transform:translateY(-4px)}}
.coin-card::before{{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(255,255,255,0.02) 0%,transparent 50%);pointer-events:none}}
</style>
</head><body>
{nav}
<div class="page">
  <div class="anim" style="margin-bottom:32px">
    <div class="mono" style="font-size:0.6rem;letter-spacing:4px;color:var(--muted);margin-bottom:8px">◈ ALL COINS</div>
    <h1 style="font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:white;letter-spacing:2px">Multi-Coin <span style="color:var(--cyan)">Sentiment</span></h1>
  </div>
  <div id="coinsGrid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px"></div>
</div>
<script>
async function load(){{
  document.getElementById('loader').classList.remove('gone');
  try{{
    let d=await(await fetch('/api/sentiment')).json();
    while(d.loading){{await new Promise(r=>setTimeout(r,2500));d=await(await fetch('/api/sentiment')).json()}}
    if(d.data&&d.data.coins)renderCoins(d.data.coins);
  }}catch(e){{console.error(e)}}
  document.getElementById('loader').classList.add('gone');
}}
function renderCoins(coins){{
  const grid=document.getElementById('coinsGrid');
  grid.innerHTML='';
  Object.entries(coins).forEach(([k,c],i)=>{{
    const up=(c.change24||0)>=0;
    const oc={{BULLISH:'var(--green)',BEARISH:'var(--red)',NEUTRAL:'var(--yellow)'}};
    const bgc={{BULLISH:'rgba(0,255,136,0.07)',BEARISH:'rgba(255,34,85,0.07)',NEUTRAL:'rgba(255,229,0,0.07)'}};
    const bdc={{BULLISH:'rgba(0,255,136,0.2)',BEARISH:'rgba(255,34,85,0.2)',NEUTRAL:'rgba(255,229,0,0.2)'}};
    const pdata=c.prediction;
    const dirColor={{UP:'var(--green)',DOWN:'var(--red)',SIDEWAYS:'var(--yellow)'}}[pdata?pdata.direction:'SIDEWAYS']||'var(--yellow)';
    const dirEmoji={{UP:'📈',DOWN:'📉',SIDEWAYS:'➡️'}}[pdata?pdata.direction:'SIDEWAYS']||'➡️';
    const card=document.createElement('div');
    card.id='coin-'+k;
    card.className='coin-card anim';
    card.style.animationDelay=(i*0.07)+'s';
    card.innerHTML=`
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px">
        <div style="display:flex;align-items:center;gap:12px">
          <div style="width:48px;height:48px;border-radius:12px;background:rgba(255,255,255,0.04);border:1px solid var(--border2);display:flex;align-items:center;justify-content:center;font-size:1.5rem">${{c.emoji}}</div>
          <div>
            <div style="font-family:Rajdhani,sans-serif;font-size:1.3rem;font-weight:700;color:white">${{c.symbol}}</div>
            <div class="mono" style="font-size:0.6rem;color:var(--muted);letter-spacing:1px">${{c.label}}</div>
          </div>
        </div>
        <div style="text-align:right">
          <div class="mono" style="font-size:1.1rem;color:var(--text)">${{c.price?'$'+c.price.toLocaleString('en-US',{{maximumFractionDigits:c.price>100?2:4}}):'—'}}</div>
          <div class="mono" style="font-size:0.7rem;color:${{up?'var(--green)':'var(--red)'}}">${{up?'▲ +':'▼ '}}${{c.change24||0}}%</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
        <div style="font-size:0.75rem;padding:5px 14px;border-radius:8px;background:${{bgc[c.overall]}};color:${{oc[c.overall]}};border:1px solid ${{bdc[c.overall]}};font-family:Share Tech Mono,monospace;letter-spacing:2px">${{c.overall}}</div>
        ${{pdata?`<div style="display:flex;align-items:center;gap:6px"><span style="font-size:1.1rem">${{dirEmoji}}</span><span class="mono" style="font-size:0.65rem;color:${{dirColor}}">${{pdata.direction}} · ${{pdata.confidence}}% conf</span></div>`:'<div class="mono" style="font-size:0.6rem;color:var(--muted)">—</div>'}}
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:16px">
        <div style="background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.12);border-radius:8px;padding:10px;text-align:center">
          <div class="mono" style="font-size:1rem;color:var(--green)">${{(c.percentages||{{}}).BULLISH||0}}%</div>
          <div class="mono" style="font-size:0.5rem;color:var(--muted);margin-top:3px">BULL</div>
        </div>
        <div style="background:rgba(255,34,85,0.06);border:1px solid rgba(255,34,85,0.12);border-radius:8px;padding:10px;text-align:center">
          <div class="mono" style="font-size:1rem;color:var(--red)">${{(c.percentages||{{}}).BEARISH||0}}%</div>
          <div class="mono" style="font-size:0.5rem;color:var(--muted);margin-top:3px">BEAR</div>
        </div>
        <div style="background:rgba(255,229,0,0.06);border:1px solid rgba(255,229,0,0.12);border-radius:8px;padding:10px;text-align:center">
          <div class="mono" style="font-size:1rem;color:var(--yellow)">${{(c.percentages||{{}}).NEUTRAL||0}}%</div>
          <div class="mono" style="font-size:0.5rem;color:var(--muted);margin-top:3px">NEUTRAL</div>
        </div>
      </div>
      <div style="height:3px;background:var(--muted);border-radius:2px;overflow:hidden">
        <div style="display:flex;height:100%">
          <div style="width:${{(c.percentages||{{}}).BULLISH||0}}%;background:var(--green);transition:width 1s ease"></div>
          <div style="width:${{(c.percentages||{{}}).BEARISH||0}}%;background:var(--red);transition:width 1s ease"></div>
          <div style="width:${{(c.percentages||{{}}).NEUTRAL||0}}%;background:var(--yellow);transition:width 1s ease"></div>
        </div>
      </div>
      <div class="mono" style="font-size:0.6rem;color:var(--muted);margin-top:10px">${{c.total||0}} posts analyzed</div>
    `;
    grid.appendChild(card);
  }});
  // scroll to hash
  if(location.hash){{const el=document.querySelector('#coin-'+location.hash.slice(1));if(el)el.scrollIntoView({{behavior:'smooth'}})}}
}}
async function manualRefresh(){{document.getElementById('refreshBtn').disabled=true;await fetch('/api/refresh',{{method:'POST'}});await load();document.getElementById('refreshBtn').disabled=false}}
load();
</script>
</body></html>"""

# ═══════════════════════════════════════════════════════════
#  PAGE — PREDICT
# ═══════════════════════════════════════════════════════════
PREDICT_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Pulse — Predict</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>{css}
.pred-card{{background:var(--surface);border:1px solid var(--border2);border-radius:18px;padding:28px 32px;position:relative;overflow:hidden;transition:all 0.3s}}
.pred-card:hover{{transform:translateY(-3px)}}
.pred-card::before{{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(255,255,255,0.02),transparent 50%);pointer-events:none}}
.signal-bar{{height:8px;border-radius:4px;transition:width 1s cubic-bezier(0.4,0,0.2,1)}}
</style>
</head><body>
{nav}
<div class="page">
  <div class="anim" style="margin-bottom:8px">
    <div class="mono" style="font-size:0.6rem;letter-spacing:4px;color:var(--muted);margin-bottom:8px">◈ AI PREDICTIONS</div>
    <h1 style="font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:white;letter-spacing:2px">24h Price <span style="color:var(--magenta)">Forecast</span></h1>
  </div>
  <div class="card anim" style="padding:14px 20px;margin-bottom:28px;border-color:rgba(255,229,0,0.15);background:rgba(255,229,0,0.03)">
    <div class="mono" style="font-size:0.65rem;color:var(--yellow);letter-spacing:1px">⚠ DISCLAIMER — These predictions are generated from Reddit sentiment + price momentum only. Not financial advice. Crypto is highly volatile. Always DYOR.</div>
  </div>

  <div id="predGrid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px;margin-bottom:32px"></div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px" class="anim" style="animation-delay:0.3s">
    <div class="card glow-cyan edge-cyan" style="padding:28px">
      <div class="section-title" style="margin-bottom:20px">Signal Confidence Comparison</div>
      <div style="position:relative;height:280px"><canvas id="confChart"></canvas></div>
    </div>
    <div class="card glow-mag edge-mag" style="padding:28px">
      <div class="section-title" style="margin-bottom:20px">How Predictions Work</div>
      <div style="display:flex;flex-direction:column;gap:14px;font-size:0.88rem;line-height:1.7;color:var(--text);opacity:0.85">
        <div style="display:flex;gap:12px;align-items:flex-start">
          <div style="width:32px;height:32px;border-radius:8px;background:rgba(0,245,255,0.1);border:1px solid rgba(0,245,255,0.2);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:0.9rem">📊</div>
          <div><strong style="color:var(--cyan)">50% Sentiment Signal</strong><br>VADER compound score from Reddit posts — more bullish posts = higher signal.</div>
        </div>
        <div style="display:flex;gap:12px;align-items:flex-start">
          <div style="width:32px;height:32px;border-radius:8px;background:rgba(255,0,255,0.1);border:1px solid rgba(255,0,255,0.2);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:0.9rem">⚡</div>
          <div><strong style="color:var(--magenta)">30% Momentum Signal</strong><br>24-hour price change amplified ×5 — captures short-term market momentum.</div>
        </div>
        <div style="display:flex;gap:12px;align-items:flex-start">
          <div style="width:32px;height:32px;border-radius:8px;background:rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.2);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:0.9rem">📈</div>
          <div><strong style="color:var(--green)">20% Trend Signal</strong><br>7-day price direction ×2 — filters out noise and captures the broader trend.</div>
        </div>
        <div style="display:flex;gap:12px;align-items:flex-start">
          <div style="width:32px;height:32px;border-radius:8px;background:rgba(255,229,0,0.1);border:1px solid rgba(255,229,0,0.2);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:0.9rem">🎯</div>
          <div><strong style="color:var(--yellow)">Confidence Score</strong><br>How much the three signals agree. All aligned = high confidence. Mixed signals = lower confidence.</div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
let confChart=null;
async function load(){{
  document.getElementById('loader').classList.remove('gone');
  try{{
    let d=await(await fetch('/api/sentiment')).json();
    while(d.loading){{await new Promise(r=>setTimeout(r,2500));d=await(await fetch('/api/sentiment')).json()}}
    if(d.data&&d.data.coins)renderPredictions(d.data.coins);
  }}catch(e){{console.error(e)}}
  document.getElementById('loader').classList.add('gone');
}}

function fmt(price){{
  if(!price)return'—';
  return'$'+price.toLocaleString('en-US',{{minimumFractionDigits:price>100?2:4,maximumFractionDigits:price>100?2:4}});
}}

function renderPredictions(coins){{
  const grid=document.getElementById('predGrid');
  grid.innerHTML='';
  const confLabels=[], confData=[], confColors=[];

  Object.entries(coins).forEach(([k,c],i)=>{{
    const p=c.prediction;
    if(!p)return;
    const dirColor={{UP:'var(--green)',DOWN:'var(--red)',SIDEWAYS:'var(--yellow)'}}[p.direction];
    const dirEmoji={{UP:'📈',DOWN:'📉',SIDEWAYS:'➡️'}}[p.direction];
    const dirBg={{UP:'rgba(0,255,136,0.08)',DOWN:'rgba(255,34,85,0.08)',SIDEWAYS:'rgba(255,229,0,0.08)'}}[p.direction];
    const dirBorder={{UP:'rgba(0,255,136,0.2)',DOWN:'rgba(255,34,85,0.2)',SIDEWAYS:'rgba(255,229,0,0.2)'}}[p.direction];

    confLabels.push(c.symbol);
    confData.push(p.confidence);
    confColors.push({{UP:'rgba(0,255,136,0.7)',DOWN:'rgba(255,34,85,0.7)',SIDEWAYS:'rgba(255,229,0,0.7)'}}[p.direction]);

    const sentBar=Math.max(0,Math.min(100,(p.sent_signal+100)/2));
    const momBar=Math.max(0,Math.min(100,(p.mom_signal+100)/2));
    const trendBar=Math.max(0,Math.min(100,(p.trend_signal+100)/2));
    const sentColor=p.sent_signal>0?'var(--green)':p.sent_signal<0?'var(--red)':'var(--yellow)';
    const momColor=p.mom_signal>0?'var(--green)':p.mom_signal<0?'var(--red)':'var(--yellow)';
    const trendColor=p.trend_signal>0?'var(--green)':p.trend_signal<0?'var(--red)':'var(--yellow)';

    const card=document.createElement('div');
    card.className='pred-card anim';
    card.style.animationDelay=(i*0.07)+'s';
    card.innerHTML=`
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
        <div style="display:flex;align-items:center;gap:12px">
          <span style="font-size:1.6rem">${{c.emoji}}</span>
          <div>
            <div style="font-family:Rajdhani,sans-serif;font-size:1.3rem;font-weight:700;color:white">${{c.symbol}}</div>
            <div class="mono" style="font-size:0.6rem;color:var(--muted)">${{fmt(p.current_price)}}</div>
          </div>
        </div>
        <div style="text-align:right">
          <div style="display:flex;align-items:center;gap:8px;justify-content:flex-end;margin-bottom:6px">
            <span style="font-size:1.3rem">${{dirEmoji}}</span>
            <span class="mono" style="font-size:1rem;font-weight:700;color:${{dirColor}};letter-spacing:2px">${{p.direction}}</span>
          </div>
          <div class="mono" style="font-size:0.65rem;color:var(--muted)">${{p.confidence}}% confidence</div>
        </div>
      </div>

      <!-- confidence bar -->
      <div style="margin-bottom:20px">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
          <span class="mono" style="font-size:0.6rem;letter-spacing:2px;color:var(--muted)">CONFIDENCE</span>
          <span class="mono" style="font-size:0.6rem;color:${{dirColor}}">${{p.confidence}}%</span>
        </div>
        <div style="height:6px;background:var(--muted);border-radius:3px;overflow:hidden">
          <div style="height:100%;width:${{p.confidence}}%;background:${{dirColor}};border-radius:3px;box-shadow:0 0 10px ${{dirColor}};transition:width 1s ease"></div>
        </div>
      </div>

      <!-- target range -->
      <div style="background:${{dirBg}};border:1px solid ${{dirBorder}};border-radius:12px;padding:16px;margin-bottom:20px;text-align:center">
        <div class="mono" style="font-size:0.55rem;letter-spacing:3px;color:var(--muted);margin-bottom:10px">24H TARGET RANGE</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:12px">
          <div>
            <div class="mono" style="font-size:0.55rem;color:var(--muted);margin-bottom:3px">LOW</div>
            <div class="mono" style="font-size:1.1rem;color:${{dirColor}}">${{fmt(p.target_lo)}}</div>
          </div>
          <div style="color:var(--muted);font-size:1.2rem">→</div>
          <div>
            <div class="mono" style="font-size:0.55rem;color:var(--muted);margin-bottom:3px">HIGH</div>
            <div class="mono" style="font-size:1.1rem;color:${{dirColor}}">${{fmt(p.target_hi)}}</div>
          </div>
        </div>
      </div>

      <!-- signal breakdown -->
      <div style="display:flex;flex-direction:column;gap:10px">
        <div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span class="mono" style="font-size:0.55rem;color:var(--muted);letter-spacing:1px">SENTIMENT (50%)</span>
            <span class="mono" style="font-size:0.55rem;color:${{sentColor}}">${{p.sent_signal>0?'+':''}}${{p.sent_signal}}</span>
          </div>
          <div style="height:4px;background:var(--muted);border-radius:2px;overflow:hidden">
            <div style="height:100%;width:${{sentBar}}%;background:${{sentColor}};border-radius:2px"></div>
          </div>
        </div>
        <div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span class="mono" style="font-size:0.55rem;color:var(--muted);letter-spacing:1px">MOMENTUM (30%)</span>
            <span class="mono" style="font-size:0.55rem;color:${{momColor}}">${{p.mom_signal>0?'+':''}}${{p.mom_signal}}</span>
          </div>
          <div style="height:4px;background:var(--muted);border-radius:2px;overflow:hidden">
            <div style="height:100%;width:${{momBar}}%;background:${{momColor}};border-radius:2px"></div>
          </div>
        </div>
        <div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span class="mono" style="font-size:0.55rem;color:var(--muted);letter-spacing:1px">7D TREND (20%)</span>
            <span class="mono" style="font-size:0.55rem;color:${{trendColor}}">${{p.trend_signal>0?'+':''}}${{p.trend_signal}}</span>
          </div>
          <div style="height:4px;background:var(--muted);border-radius:2px;overflow:hidden">
            <div style="height:100%;width:${{trendBar}}%;background:${{trendColor}};border-radius:2px"></div>
          </div>
        </div>
      </div>
    `;
    grid.appendChild(card);
  }});

  // confidence chart
  const ctx=document.getElementById('confChart').getContext('2d');
  if(confChart)confChart.destroy();
  confChart=new Chart(ctx,{{
    type:'bar',
    data:{{labels:confLabels,datasets:[{{label:'Confidence %',data:confData,backgroundColor:confColors,borderColor:confColors.map(c=>c.replace('0.7','1')),borderWidth:1,borderRadius:6}}]}},
    options:{{responsive:true,maintainAspectRatio:false,animation:{{duration:1000}},scales:{{x:{{ticks:{{color:'#2a4060',font:{{family:'Share Tech Mono',size:11}}}},grid:{{color:'rgba(0,245,255,0.04)'}}}},y:{{min:0,max:100,ticks:{{color:'#2a4060',font:{{family:'Share Tech Mono',size:10}},callback:v=>v+'%'}},grid:{{color:'rgba(0,245,255,0.04)'}}}}}},plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'rgba(3,10,24,0.95)',borderColor:'rgba(0,245,255,0.2)',borderWidth:1,callbacks:{{label:c=>c.parsed.y+'% confidence'}}}}}}}}
  }});
}}
async function manualRefresh(){{document.getElementById('refreshBtn').disabled=true;await fetch('/api/refresh',{{method:'POST'}});await load();document.getElementById('refreshBtn').disabled=false}}
load();
</script>
</body></html>"""

# ═══════════════════════════════════════════════════════════
#  PAGE — POSTS  (original, unchanged)
# ═══════════════════════════════════════════════════════════
POSTS_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Pulse — Posts</title>
<style>{css}</style>
</head><body>
{nav}
<div class="page">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px" class="anim">
    <div>
      <div class="mono" style="font-size:0.6rem;letter-spacing:4px;color:var(--muted);margin-bottom:8px">◈ ALL POSTS</div>
      <h1 style="font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:white;letter-spacing:2px">Reddit Feed <span style="color:var(--cyan)">Analysis</span></h1>
    </div>
    <div style="display:flex;gap:8px" id="filters">
      <button onclick="filter('ALL')" class="fbtn active" data-f="ALL">ALL</button>
      <button onclick="filter('BULLISH')" class="fbtn bull-btn" data-f="BULLISH">🟢 BULLISH</button>
      <button onclick="filter('BEARISH')" class="fbtn bear-btn" data-f="BEARISH">🔴 BEARISH</button>
      <button onclick="filter('NEUTRAL')" class="fbtn neut-btn" data-f="NEUTRAL">⚪ NEUTRAL</button>
    </div>
  </div>
  <div id="postGrid" style="display:flex;flex-direction:column;gap:10px"></div>
</div>
<style>
.fbtn{{font-family:'Share Tech Mono',monospace;font-size:0.6rem;letter-spacing:2px;padding:7px 16px;border-radius:6px;border:1px solid var(--border2);background:transparent;color:var(--muted);cursor:pointer;transition:all 0.2s}}
.fbtn.active,.fbtn:hover{{color:var(--cyan);border-color:rgba(0,245,255,0.3);background:rgba(0,245,255,0.06)}}
.bull-btn.active{{color:var(--green);border-color:rgba(0,255,136,0.3);background:rgba(0,255,136,0.06)}}
.bear-btn.active{{color:var(--red);border-color:rgba(255,34,85,0.3);background:rgba(255,34,85,0.06)}}
.neut-btn.active{{color:var(--yellow);border-color:rgba(255,229,0,0.3);background:rgba(255,229,0,0.06)}}
.post-card{{background:var(--surface);border:1px solid var(--border2);border-radius:14px;padding:18px 22px;display:flex;align-items:flex-start;gap:16px;transition:all 0.25s;animation:fadeUp 0.4s ease both}}
.post-card:hover{{transform:translateX(6px);border-color:rgba(0,245,255,0.25);background:var(--surface2)}}
</style>
<script>
let allPosts=[],currentFilter='ALL';
const tc={{BULLISH:'var(--green)',BEARISH:'var(--red)',NEUTRAL:'var(--yellow)'}};
const bg={{BULLISH:'rgba(0,255,136,0.07)',BEARISH:'rgba(255,34,85,0.07)',NEUTRAL:'rgba(255,229,0,0.07)'}};
const bc={{BULLISH:'rgba(0,255,136,0.15)',BEARISH:'rgba(255,34,85,0.15)',NEUTRAL:'rgba(255,229,0,0.15)'}};
function filter(f){{currentFilter=f;document.querySelectorAll('.fbtn').forEach(b=>{{b.classList.remove('active');if(b.dataset.f===f)b.classList.add('active')}});renderPosts()}}
function renderPosts(){{
  const grid=document.getElementById('postGrid');
  const posts=currentFilter==='ALL'?allPosts:allPosts.filter(p=>p.label===currentFilter);
  grid.innerHTML='';
  posts.forEach((p,i)=>{{
    const sc=p.compound>0?'var(--green)':p.compound<0?'var(--red)':'var(--yellow)';
    const sv=(p.compound>0?'+':'')+p.compound.toFixed(3);
    const div=document.createElement('div');
    div.className='post-card';div.style.animationDelay=(i*0.02)+'s';
    div.innerHTML=`<div style="display:flex;flex-direction:column;align-items:center;gap:6px;flex-shrink:0;width:80px"><span style="font-family:Share Tech Mono,monospace;font-size:0.55rem;padding:3px 8px;border-radius:5px;background:${{bg[p.label]}};color:${{tc[p.label]}};border:1px solid ${{bc[p.label]}};letter-spacing:1px;text-align:center">${{p.label}}</span><span style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:${{sc}}">${{sv}}</span><span style="font-family:Share Tech Mono,monospace;font-size:0.55rem;color:var(--muted)">↑ ${{p.upvotes}}</span></div><div style="flex:1;min-width:0;border-left:1px solid var(--border);padding-left:16px"><div style="font-size:0.92rem;line-height:1.5;margin-bottom:8px">${{p.title}}</div><div style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:var(--muted);letter-spacing:1px">${{p.source}}</div></div>`;
    grid.appendChild(div);
  }});
  if(posts.length===0){{grid.innerHTML='<div class="mono" style="text-align:center;padding:60px;color:var(--muted);font-size:0.7rem;letter-spacing:3px">NO POSTS FOUND</div>'}}
}}
async function load(){{
  document.getElementById('loader').classList.remove('gone');
  try{{
    let d=await(await fetch('/api/sentiment')).json();
    while(d.loading){{await new Promise(r=>setTimeout(r,2500));d=await(await fetch('/api/sentiment')).json()}}
    if(d.data){{allPosts=d.data.posts||[];renderPosts()}}
  }}catch(e){{console.error(e)}}
  document.getElementById('loader').classList.add('gone');
}}
async function manualRefresh(){{document.getElementById('refreshBtn').disabled=true;await fetch('/api/refresh',{{method:'POST'}});await load();document.getElementById('refreshBtn').disabled=false}}
load();
</script>
</body></html>"""

# ═══════════════════════════════════════════════════════════
#  PAGE — ABOUT
# ═══════════════════════════════════════════════════════════
ABOUT_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Pulse — About</title>
<style>{css}</style>
</head><body>
{nav}
<div class="page" style="max-width:900px">
  <div class="anim" style="margin-bottom:48px">
    <div class="mono" style="font-size:0.6rem;letter-spacing:4px;color:var(--muted);margin-bottom:8px">◈ ABOUT</div>
    <h1 style="font-family:'Rajdhani',sans-serif;font-size:2.5rem;font-weight:700;color:white;letter-spacing:2px;line-height:1.2">How <span style="color:var(--cyan)">Crypto Pulse</span><br>Works</h1>
  </div>
  <div style="display:flex;flex-direction:column;gap:16px">{cards}</div>
  <div class="card glow-cyan edge-cyan anim" style="padding:32px;margin-top:24px;animation-delay:0.5s">
    <div class="mono" style="font-size:0.6rem;letter-spacing:3px;color:var(--muted);margin-bottom:16px">◈ TECH STACK</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px">{tech}</div>
  </div>
</div>
<script>document.getElementById('loader').classList.add('gone');async function manualRefresh(){{document.getElementById('refreshBtn').disabled=true;await fetch('/api/refresh',{{method:'POST'}});document.getElementById('refreshBtn').disabled=false}}</script>
</body></html>"""

def make_about_card(icon, title, body, color, delay):
    colors = {'cyan':('rgba(0,245,255,0.2)','rgba(0,245,255,0.07)','var(--cyan)'),
              'mag': ('rgba(255,0,255,0.2)','rgba(255,0,255,0.07)','var(--magenta)'),
              'green':('rgba(0,255,136,0.2)','rgba(0,255,136,0.07)','var(--green)'),
              'yellow':('rgba(255,229,0,0.2)','rgba(255,229,0,0.07)','var(--yellow)')}
    bc,bg,tc = colors[color]
    return f"""<div class="card anim" style="padding:28px;display:flex;gap:20px;align-items:flex-start;border-color:{bc};box-shadow:0 0 0 1px {bc};animation-delay:{delay}s">
      <div style="width:48px;height:48px;border-radius:12px;background:{bg};border:1px solid {bc};display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0">{icon}</div>
      <div><div style="font-family:Rajdhani,sans-serif;font-size:1.1rem;font-weight:700;color:{tc};letter-spacing:2px;margin-bottom:8px">{title}</div>
      <div style="font-size:0.88rem;color:var(--text);line-height:1.7;opacity:0.8">{body}</div></div></div>"""

def make_tech_chip(name, color):
    colors = {'cyan':('rgba(0,245,255,0.1)','rgba(0,245,255,0.25)','var(--cyan)'),
              'mag': ('rgba(255,0,255,0.1)','rgba(255,0,255,0.25)','var(--magenta)'),
              'green':('rgba(0,255,136,0.1)','rgba(0,255,136,0.25)','var(--green)'),
              'yellow':('rgba(255,229,0,0.1)','rgba(255,229,0,0.25)','var(--yellow)'),
              'orange':('rgba(255,140,0,0.1)','rgba(255,140,0,0.25)','var(--orange)')}
    bg,bc,tc = colors[color]
    return f'<div style="background:{bg};border:1px solid {bc};border-radius:8px;padding:12px;text-align:center"><div class="mono" style="font-size:0.7rem;color:{tc};letter-spacing:1px">{name}</div></div>'

ABOUT_CARDS = (
    make_about_card('📡','Data Collection',"We scrape 5 crypto subreddits (BTC, ETH, SOL, BNB, USDT) plus r/CryptoCurrency and r/CryptoMarkets using Reddit's public JSON API. No API key required.",'cyan',0.05)+
    make_about_card('🧠','Sentiment Analysis',"Each post is scored by VADER NLP (Valence Aware Dictionary and sEntiment Reasoner), tuned for social media. Score range: -1 (most negative) to +1 (most positive).",'mag',0.1)+
    make_about_card('🎯','Crypto Keywords',"A custom keyword layer boosts VADER's score for crypto-specific language. 'Moon', 'hodl', 'halving' push bullish; 'crash', 'rug', 'liquidation' push bearish.",'green',0.15)+
    make_about_card('🔮','Price Prediction',"Predictions combine 50% sentiment signal + 30% 24h momentum + 20% 7d trend. Confidence reflects how much all three signals agree on direction.",'yellow',0.2)
)
TECH_CHIPS = (make_tech_chip('Python','cyan')+make_tech_chip('Flask','mag')+
              make_tech_chip('VADER NLP','green')+make_tech_chip('Chart.js','yellow')+
              make_tech_chip('Reddit JSON','orange')+make_tech_chip('CoinGecko','cyan')+
              make_tech_chip('5 Coins','mag')+make_tech_chip('Threading','green'))

def build_page(template, active, **kwargs):
    nav = (NAV_TEMPLATE
           .replace('{d}',  'active' if active=='d'  else '')
           .replace('{co}', 'active' if active=='co' else '')
           .replace('{pr}', 'active' if active=='pr' else '')
           .replace('{p}',  'active' if active=='p'  else '')
           .replace('{a}',  'active' if active=='a'  else ''))
    result = template.replace('{css}', SHARED_CSS).replace('{nav}', nav)
    for k, v in kwargs.items():
        result = result.replace('{' + k + '}', str(v))
    return result

# ═══════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════
@app.route("/")
def dashboard(): return render_template_string(build_page(DASHBOARD_HTML, 'd'))

@app.route("/coins")
def coins_page(): return render_template_string(build_page(COINS_HTML, 'co'))

@app.route("/predict")
def predict_page(): return render_template_string(build_page(PREDICT_HTML, 'pr'))

@app.route("/posts")
def posts(): return render_template_string(build_page(POSTS_HTML, 'p'))

@app.route("/about")
def about(): return render_template_string(build_page(ABOUT_HTML, 'a', cards=ABOUT_CARDS, tech=TECH_CHIPS))

@app.route("/api/sentiment")
def api_sentiment():
    return jsonify({"loading": cache["loading"], "last_updated": cache["last_updated"], "data": cache["data"]})

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    if not cache["loading"]:
        threading.Thread(target=run_analysis, daemon=True).start()
    return jsonify({"status": "started"})

import os
if __name__ == "__main__":
    threading.Thread(target=bg_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
