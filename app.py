import subprocess, sys

def install_deps():
    for dep in ["flask", "requests", "vaderSentiment"]:
        try: __import__(dep.replace("-","_"))
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable,"-m","pip","install",dep,"-q"])

install_deps()

from flask import Flask, jsonify, Response, redirect
import requests, time, threading
from datetime import datetime
from sentiment import analyze_posts, summarize, _pipeline, USE_BERT

app     = Flask(__name__)
cache   = {"data": None, "last_updated": None, "loading": False}
HEADERS = {"User-Agent": "crypto-pulse/3.0"}

COINS = {
    "bitcoin":     {"symbol":"BTC","label":"Bitcoin",  "emoji":"₿","coingecko_id":"bitcoin",     "h":"#F7931A","h2":"#FFC366","subreddits":["Bitcoin","BitcoinMarkets"],"keywords_bull":["moon","hodl","ath","accumulate","halving","laser eyes","stack sats"],"keywords_bear":["crash","dump","bear","sell","liquidation","rug","ponzi","dead"]},
    "ethereum":    {"symbol":"ETH","label":"Ethereum", "emoji":"Ξ","coingecko_id":"ethereum",    "h":"#627EEA","h2":"#A0B4FF","subreddits":["ethereum","ethfinance"],    "keywords_bull":["merge","staking","l2","rollup","defi","nft","flippening","ultrasound"],"keywords_bear":["gas fees","congestion","dump","bear","liquidation","rug"]},
    "solana":      {"symbol":"SOL","label":"Solana",   "emoji":"◎","coingecko_id":"solana",      "h":"#9945FF","h2":"#C893FF","subreddits":["solana","solanaNFT"],        "keywords_bull":["fast","cheap","nft","defi","validator","pump","meme"],"keywords_bear":["outage","down","hack","dump","bear","centralized"]},
    "tether":      {"symbol":"USDT","label":"Tether",  "emoji":"₮","coingecko_id":"tether",      "h":"#26A17B","h2":"#4ECCA3","subreddits":["Tether","CryptoCurrency"],   "keywords_bull":["stable","peg","trusted","reserve","backed","safe haven"],"keywords_bear":["depeg","unbacked","fraud","lawsuit","investigation","fud","collapse"]},
    "binancecoin": {"symbol":"BNB","label":"BNB",      "emoji":"⬡","coingecko_id":"binancecoin", "h":"#F3BA2F","h2":"#FFE08A","subreddits":["binance","BinanceSmartChain"],"keywords_bull":["burn","bnb chain","launchpad","cz","defi","ecosystem","stake"],"keywords_bear":["sec","lawsuit","regulated","dump","bear","centralized","hack"]},
}

SUBREDDITS_GENERAL = ["CryptoCurrency","CryptoMarkets"]
POSTS_PER_SUB = 15
POST_SORT     = "new"
REFRESH_SECS  = 300

def fetch_sub(sub):
    try:
        r = requests.get(f"https://www.reddit.com/r/{sub}/{POST_SORT}.json?limit={POSTS_PER_SUB}", headers=HEADERS, timeout=10)
        return [{"source":f"r/{sub}","title":c["data"].get("title",""),"text":(c["data"].get("title","")+" "+c["data"].get("selftext","")).strip(),"upvotes":c["data"].get("score",0)} for c in r.json()["data"]["children"]]
    except: return []

def get_prices():
    ids = ",".join(c["coingecko_id"] for c in COINS.values())
    try:
        data = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true&include_7d_change=true", timeout=10).json()
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

def predict_price(coin_key, sentiment_score, price_data):
    cfg   = COINS[coin_key]
    price = price_data.get("price")
    if not price: return None
    ch24  = price_data.get("change24", 0)
    ch7d  = price_data.get("change7d", 0)
    sent_signal     = sentiment_score * 100
    momentum_signal = max(-100, min(100, ch24 * 5))
    trend_signal    = max(-100, min(100, ch7d * 2))
    combined = (sent_signal * 0.50) + (momentum_signal * 0.30) + (trend_signal * 0.20)
    signs = [1 if x > 0 else -1 if x < 0 else 0 for x in [sent_signal, momentum_signal, trend_signal]]
    agreement = abs(sum(signs))
    confidence = {0:30,1:45,2:65,3:82}[agreement]
    base_move_pct = abs(combined) / 1000
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
    return {"direction":direction,"confidence":confidence,"target_lo":target_lo,"target_hi":target_hi,"combined":round(combined,1),"sent_signal":round(sent_signal,1),"mom_signal":round(momentum_signal,1),"trend_signal":round(trend_signal,1),"current_price":price}

def run_analysis():
    cache["loading"] = True
    prices = get_prices()
    coin_results = {}
    for coin_key, cfg in COINS.items():
        subs = cfg["subreddits"] + SUBREDDITS_GENERAL
        posts = []
        for sub in subs:
            posts.extend(fetch_sub(sub))
            time.sleep(0.5)
        coin_terms = [cfg["symbol"].lower(), cfg["label"].lower()] + [k.lower() for k in cfg["keywords_bull"] + cfg["keywords_bear"]]
        relevant = [p for p in posts if any(t in p["text"].lower() for t in coin_terms)]
        if not relevant: relevant = posts[:30]
        results  = analyze_posts(relevant)
        summary  = summarize(results)
        pdata    = prices.get(coin_key, {})
        prediction = predict_price(coin_key, summary["avg_score"], pdata)
        coin_results[coin_key] = {
            "symbol":cfg["symbol"],"label":cfg["label"],"emoji":cfg["emoji"],
            "color":cfg["h"],"color2":cfg["h2"],
            "total":summary["total"],"counts":summary["counts"],"percentages":summary["percentages"],
            "avg_score":summary["avg_score"],"overall":summary["overall"],"method":summary["method"],
            "price":pdata.get("price"),"change24":pdata.get("change24"),"change7d":pdata.get("change7d"),
            "prediction":prediction,
            "posts":[{"source":r["source"],"title":r["title"],"label":r["label"],"compound":r["compound"],"upvotes":r["upvotes"]} for r in results],
        }
        print(f"✅ {cfg['symbol']}: {summary['total']} posts — {summary['overall']}")
    btc = coin_results.get("bitcoin", {})
    cache["data"] = {
        "total":btc.get("total"),"counts":btc.get("counts"),"percentages":btc.get("percentages"),
        "avg_score":btc.get("avg_score"),"overall":btc.get("overall"),
        "btc_price":btc.get("price"),"btc_change":btc.get("change24"),
        "method":btc.get("method"),"posts":btc.get("posts"),
        "coins":coin_results,
    }
    cache["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cache["loading"] = False

def bg_loop():
    while True: run_analysis(); time.sleep(REFRESH_SECS)

# ═══════════════════════════════════════════
#  SHARED CSS
# ═══════════════════════════════════════════
SHARED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&family=Rajdhani:wght@400;600;700&display=swap');

*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}

:root{
  --bg:      #0a0a0f;
  --bg2:     #111118;
  --surface: #16161f;
  --surface2:#1e1e2a;
  --border:  rgba(255,255,255,0.06);
  --border2: rgba(255,255,255,0.12);
  --text:    #e8e8f0;
  --muted:   #555568;
  --btc:     #F7931A;
  --eth:     #627EEA;
  --sol:     #9945FF;
  --usdt:    #26A17B;
  --bnb:     #F3BA2F;
  --bull:    #22c55e;
  --bear:    #ef4444;
  --neut:    #eab308;
  --radius:  16px;
}

html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'Outfit',sans-serif;min-height:100vh;overflow-x:hidden}

/* mesh background */
body::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse 80% 50% at 20% 0%,rgba(99,126,234,0.07) 0%,transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 100%,rgba(153,69,255,0.07) 0%,transparent 60%),
    radial-gradient(ellipse 50% 60% at 50% 50%,rgba(247,147,26,0.04) 0%,transparent 70%);
}

/* NAV */
nav{
  position:sticky;top:0;z-index:100;
  height:68px;display:flex;align-items:center;justify-content:space-between;
  padding:0 40px;
  background:rgba(10,10,15,0.8);
  backdrop-filter:blur(20px);
  border-bottom:1px solid var(--border2);
}

.nav-logo{
  display:flex;align-items:center;gap:10px;
  text-decoration:none;
  font-weight:800;font-size:1.25rem;letter-spacing:-0.5px;color:var(--text);
}

.logo-mark{
  width:36px;height:36px;border-radius:10px;
  background:linear-gradient(135deg,#F7931A,#9945FF);
  display:flex;align-items:center;justify-content:center;
  font-size:1rem;font-weight:900;color:white;
  box-shadow:0 0 20px rgba(247,147,26,0.3);
}

.nav-links{display:flex;align-items:center;gap:2px}
.nav-link{
  font-size:0.8rem;font-weight:500;letter-spacing:0.3px;
  color:var(--muted);text-decoration:none;
  padding:8px 16px;border-radius:8px;
  transition:all 0.2s;
}
.nav-link:hover{color:var(--text);background:var(--surface2)}
.nav-link.active{color:white;background:var(--surface2);font-weight:600}

.nav-right{display:flex;align-items:center;gap:10px}

.live-badge{
  display:flex;align-items:center;gap:6px;
  font-size:0.7rem;font-weight:600;letter-spacing:1px;
  color:var(--bull);
  background:rgba(34,197,94,0.1);
  border:1px solid rgba(34,197,94,0.25);
  padding:5px 12px;border-radius:20px;
}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--bull);box-shadow:0 0 6px var(--bull);animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.4;transform:scale(0.8)}}

.btn-refresh{
  font-family:'JetBrains Mono',monospace;
  font-size:0.7rem;font-weight:600;
  background:var(--surface2);
  border:1px solid var(--border2);
  color:var(--text);
  padding:8px 18px;border-radius:8px;
  cursor:pointer;transition:all 0.2s;
}
.btn-refresh:hover{background:var(--surface);border-color:rgba(255,255,255,0.2);transform:translateY(-1px)}
.btn-refresh:disabled{opacity:0.4;cursor:not-allowed;transform:none}

/* MOBILE */
.hamburger{display:none;background:var(--surface2);border:1px solid var(--border2);color:var(--text);padding:8px 12px;border-radius:8px;cursor:pointer;font-size:1rem}
.mobile-menu{display:none;position:fixed;top:68px;left:0;right:0;background:rgba(10,10,15,0.98);backdrop-filter:blur(20px);border-bottom:1px solid var(--border2);padding:12px;flex-direction:column;gap:4px;z-index:99}
.mobile-menu.open{display:flex}
.mobile-menu a{font-size:0.85rem;font-weight:500;color:var(--muted);text-decoration:none;padding:12px 16px;border-radius:8px;transition:all 0.2s}
.mobile-menu a:hover,.mobile-menu a.active{color:white;background:var(--surface2)}

@media(max-width:768px){
  nav{padding:0 16px;height:60px}
  .nav-links{display:none}
  .hamburger{display:flex}
  .live-badge{display:none}
  .page{padding:16px 12px 60px!important}
}

/* PAGE */
.page{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:36px 40px 80px}

/* LOADER */
.loader{position:fixed;inset:0;z-index:9999;background:var(--bg);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0;transition:opacity 0.6s;overflow:hidden}
.loader.gone{opacity:0;pointer-events:none}
.loader-bg{position:absolute;inset:0;background:radial-gradient(ellipse 70% 60% at 30% 40%,rgba(247,147,26,0.08) 0%,transparent 60%),radial-gradient(ellipse 60% 50% at 70% 60%,rgba(153,69,255,0.09) 0%,transparent 60%),radial-gradient(ellipse 50% 40% at 50% 20%,rgba(99,126,234,0.07) 0%,transparent 60%);pointer-events:none}
.loader-coins{display:flex;gap:18px;margin-bottom:40px;align-items:center}
.loader-coin{width:52px;height:52px;border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;border:1px solid rgba(255,255,255,0.1);animation:coinFloat 2s ease-in-out infinite}
.loader-coin:nth-child(1){background:rgba(247,147,26,0.15);border-color:rgba(247,147,26,0.3);animation-delay:0s}
.loader-coin:nth-child(2){background:rgba(99,126,234,0.15);border-color:rgba(99,126,234,0.3);animation-delay:0.2s}
.loader-coin:nth-child(3){background:rgba(153,69,255,0.15);border-color:rgba(153,69,255,0.3);animation-delay:0.4s}
.loader-coin:nth-child(4){background:rgba(38,161,123,0.15);border-color:rgba(38,161,123,0.3);animation-delay:0.6s}
.loader-coin:nth-child(5){background:rgba(243,186,47,0.15);border-color:rgba(243,186,47,0.3);animation-delay:0.8s}
@keyframes coinFloat{0%,100%{transform:translateY(0px);opacity:0.7}50%{transform:translateY(-10px);opacity:1}}
.loader-title{font-size:2.2rem;font-weight:900;letter-spacing:-1.5px;color:var(--text);margin-bottom:8px;text-align:center}
.loader-title span{background:linear-gradient(90deg,#F7931A,#9945FF,#627EEA,#26A17B);background-size:200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:gradMove 3s linear infinite}
@keyframes gradMove{0%{background-position:0%}100%{background-position:200%}}
.loader-sub{font-size:0.72rem;font-weight:600;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-bottom:36px;text-align:center}
.loader-steps{display:flex;flex-direction:column;gap:10px;width:320px;margin-bottom:32px}
.loader-step{display:flex;align-items:center;gap:12px;padding:10px 14px;background:var(--surface);border:1px solid var(--border2);border-radius:10px;animation:stepIn 0.5s ease both}
.loader-step:nth-child(1){animation-delay:0.2s}
.loader-step:nth-child(2){animation-delay:0.7s}
.loader-step:nth-child(3){animation-delay:1.2s}
@keyframes stepIn{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:translateX(0)}}
.step-icon{font-size:1rem;width:28px;text-align:center;flex-shrink:0}
.step-text{font-size:0.72rem;font-weight:600;color:var(--muted);letter-spacing:0.5px}
.step-dot{width:6px;height:6px;border-radius:50%;background:var(--bull);box-shadow:0 0 8px var(--bull);animation:pulse 1.5s infinite;margin-left:auto;flex-shrink:0}
.loader-bar-wrap{width:320px}
.loader-bar-label{display:flex;justify-content:space-between;margin-bottom:8px}
.loader-bar-label span{font-size:0.65rem;font-weight:600;letter-spacing:1px;color:var(--muted);text-transform:uppercase}
.loader-bar{width:100%;height:4px;background:var(--surface2);border-radius:2px;overflow:hidden}
.loader-fill{height:100%;background:linear-gradient(90deg,#F7931A,#9945FF,#627EEA);border-radius:2px;animation:bar 2s ease-in-out infinite}
@keyframes bar{0%{width:0%;margin-left:0%}50%{width:65%;margin-left:15%}100%{width:0%;margin-left:100%}}

/* CARDS */
.card{background:var(--surface);border:1px solid var(--border2);border-radius:var(--radius);position:relative;overflow:hidden;transition:transform 0.25s,box-shadow 0.25s}
.card:hover{transform:translateY(-2px)}

/* SECTION LABEL */
.label{font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:14px;display:flex;align-items:center;gap:10px}
.label::after{content:'';flex:1;height:1px;background:var(--border2)}

/* MONO */
.mono{font-family:'JetBrains Mono',monospace}

/* PILL */
.pill{display:inline-flex;align-items:center;gap:5px;font-size:0.7rem;font-weight:700;padding:4px 10px;border-radius:20px;letter-spacing:0.5px}
.pill-bull{background:rgba(34,197,94,0.12);color:var(--bull);border:1px solid rgba(34,197,94,0.25)}
.pill-bear{background:rgba(239,68,68,0.12);color:var(--bear);border:1px solid rgba(239,68,68,0.25)}
.pill-neut{background:rgba(234,179,8,0.12);color:var(--neut);border:1px solid rgba(234,179,8,0.25)}

@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.anim{animation:fadeUp 0.5s ease both}

::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:2px}
"""

NAV_TEMPLATE = """
<div class="loader gone" id="loader">
  <div class="loader-bg"></div>
  <div class="loader-coins">
    <div class="loader-coin">₿</div>
    <div class="loader-coin">Ξ</div>
    <div class="loader-coin">◎</div>
    <div class="loader-coin">₮</div>
    <div class="loader-coin">⬡</div>
  </div>
  <div class="loader-title">Crypto <span>Pulse</span></div>
  <div class="loader-sub">Scanning Reddit · Analyzing Sentiment</div>
  <div class="loader-steps">
    <div class="loader-step"><span class="step-icon">📡</span><span class="step-text">Fetching Reddit posts across 5 coins</span><span class="step-dot"></span></div>
    <div class="loader-step"><span class="step-icon">🧠</span><span class="step-text">Running VADER sentiment analysis</span><span class="step-dot" style="animation-delay:0.5s"></span></div>
    <div class="loader-step"><span class="step-icon">🔮</span><span class="step-text">Generating 24h price predictions</span><span class="step-dot" style="animation-delay:1s"></span></div>
  </div>
  <div class="loader-bar-wrap">
    <div class="loader-bar-label"><span>Processing</span><span>Please wait ~30s</span></div>
    <div class="loader-bar"><div class="loader-fill"></div></div>
  </div>
</div>
<nav>
  <a href="/" class="nav-logo"><div class="logo-mark">⬡</div>Crypto Pulse</a>
  <div class="nav-links">
    <a href="/" class="nav-link {d}">Dashboard</a>
    <a href="/coins" class="nav-link {co}">Coins</a>
    <a href="/predict" class="nav-link {pr}">Predict</a>
    <a href="/posts" class="nav-link {p}">Posts</a>
    <a href="/about" class="nav-link {a}">About</a>
  </div>
  <div class="nav-right">
    <div class="live-badge"><div class="live-dot"></div>LIVE</div>
    <button class="btn-refresh" id="refreshBtn" onclick="manualRefresh()">↻ Refresh</button>
    <button class="hamburger" onclick="document.getElementById('mobileMenu').classList.toggle('open')">☰</button>
  </div>
</nav>
<div class="mobile-menu" id="mobileMenu">
  <a href="/" class="{d}">Dashboard</a>
  <a href="/coins" class="{co}">Coins</a>
  <a href="/predict" class="{pr}">Predict</a>
  <a href="/posts" class="{p}">Posts</a>
  <a href="/about" class="{a}">About</a>
</div>
"""

# ═══════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Pulse — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
{css}

/* ── CYBERPUNK DASHBOARD OVERRIDES ───────────────────── */
:root {
  --neon-cyan:   #00f5ff;
  --neon-pink:   #ff2d78;
  --neon-green:  #00ff9f;
  --neon-yellow: #ffe600;
  --neon-purple: #bf5fff;
  --panel-bg:    rgba(8,12,24,0.85);
  --panel-border:rgba(0,245,255,0.12);
}

.cp-grid-bg {
  position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:
    linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px);
  background-size:48px 48px;
}

.cp-scanline {
  position:fixed;inset:0;z-index:0;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.06) 2px,rgba(0,0,0,0.06) 4px);
}

.cp-glow-orb {
  position:fixed;border-radius:50%;filter:blur(120px);pointer-events:none;z-index:0;
  animation:orbDrift 16s ease-in-out infinite;
}
@keyframes orbDrift{0%,100%{transform:translate(0,0)}33%{transform:translate(30px,-40px)}66%{transform:translate(-20px,30px)}}

/* HEADER BAND */
.cp-header {
  position:relative;z-index:1;
  padding:32px 40px 0;
  display:flex;align-items:flex-end;justify-content:space-between;
  margin-bottom:28px;
}
.cp-title-block {}
.cp-eyebrow {
  font-family:'JetBrains Mono',monospace;
  font-size:0.6rem;letter-spacing:4px;
  color:var(--neon-cyan);opacity:0.7;
  margin-bottom:6px;text-transform:uppercase;
}
.cp-title {
  font-family:'Rajdhani',sans-serif;
  font-size:2.6rem;font-weight:700;line-height:1;
  letter-spacing:2px;color:#fff;
  text-shadow:0 0 40px rgba(0,245,255,0.3);
}
.cp-title em {
  font-style:normal;
  color:var(--neon-cyan);
  text-shadow:0 0 20px var(--neon-cyan),0 0 60px rgba(0,245,255,0.4);
}
.cp-timestamp {
  font-family:'JetBrains Mono',monospace;
  font-size:0.62rem;letter-spacing:2px;
  color:rgba(0,245,255,0.4);
  text-align:right;padding-bottom:6px;
}

/* STAT TICKER ROW */
.cp-ticker {
  display:flex;gap:2px;
  background:rgba(0,245,255,0.04);
  border-top:1px solid var(--neon-cyan);
  border-bottom:1px solid rgba(0,245,255,0.15);
  padding:10px 40px;
  position:relative;z-index:1;
  margin-bottom:28px;
  overflow:hidden;
}
.cp-ticker::before {
  content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
  background:var(--neon-cyan);
  box-shadow:0 0 12px var(--neon-cyan),0 0 30px var(--neon-cyan);
}
.cp-tick {
  display:flex;align-items:center;gap:20px;
  flex:1;padding:0 20px;
  border-right:1px solid rgba(0,245,255,0.08);
}
.cp-tick:last-child{border-right:none}
.cp-tick-label {
  font-family:'JetBrains Mono',monospace;
  font-size:0.55rem;letter-spacing:3px;
  color:rgba(0,245,255,0.4);text-transform:uppercase;
  white-space:nowrap;
}
.cp-tick-val {
  font-family:'Rajdhani',sans-serif;
  font-size:1.5rem;font-weight:700;letter-spacing:1px;
  white-space:nowrap;
}

/* COIN CARDS */
.cp-coin {
  background:var(--panel-bg);
  border:1px solid var(--panel-border);
  border-radius:4px;
  padding:22px 24px;
  position:relative;overflow:hidden;
  transition:transform 0.2s,box-shadow 0.2s;
  clip-path:polygon(0 0,calc(100% - 16px) 0,100% 16px,100% 100%,0 100%);
}
.cp-coin:hover {
  transform:translateY(-4px);
}
.cp-coin::before {
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--coin-color,var(--neon-cyan)),transparent);
}
.cp-coin::after {
  content:'';position:absolute;
  top:0;right:0;width:16px;height:16px;
  background:linear-gradient(225deg,var(--coin-color,var(--neon-cyan)) 50%,transparent 50%);
  opacity:0.6;
}
.cp-coin-glow {
  position:absolute;top:-20px;right:-20px;
  width:120px;height:120px;border-radius:50%;
  filter:blur(50px);opacity:0.12;pointer-events:none;
}

.cp-coin-header {
  display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px;
}
.cp-coin-symbol {
  font-family:'Rajdhani',sans-serif;
  font-size:1.5rem;font-weight:700;letter-spacing:2px;
  line-height:1;
}
.cp-coin-name {
  font-family:'JetBrains Mono',monospace;
  font-size:0.55rem;letter-spacing:2px;
  color:rgba(255,255,255,0.3);margin-top:3px;text-transform:uppercase;
}
.cp-coin-price {
  font-family:'JetBrains Mono',monospace;
  font-size:1.2rem;font-weight:700;letter-spacing:-0.5px;
  text-align:right;line-height:1;
}
.cp-coin-change {
  font-family:'JetBrains Mono',monospace;
  font-size:0.65rem;font-weight:700;
  margin-top:4px;text-align:right;
}

.cp-sent-badge {
  display:inline-flex;align-items:center;gap:5px;
  font-family:'JetBrains Mono',monospace;
  font-size:0.6rem;font-weight:700;letter-spacing:2px;
  padding:3px 10px;border-radius:2px;
  border:1px solid;margin-bottom:14px;
}

.cp-bar-wrap {margin-bottom:14px}
.cp-bar-labels {
  display:flex;justify-content:space-between;margin-bottom:5px;
}
.cp-bar-lbl {
  font-family:'JetBrains Mono',monospace;
  font-size:0.5rem;letter-spacing:2px;color:rgba(255,255,255,0.25);
}
.cp-bar-track {
  height:3px;background:rgba(255,255,255,0.06);
  border-radius:0;overflow:visible;position:relative;
}
.cp-bar-fill {
  height:100%;border-radius:0;
  transition:width 1.2s cubic-bezier(0.4,0,0.2,1);
  position:relative;
}
.cp-bar-fill::after {
  content:'';position:absolute;right:-1px;top:-3px;
  width:8px;height:8px;border-radius:50%;
  background:inherit;
  box-shadow:0 0 8px currentColor;
}

.cp-mini-grid {
  display:grid;grid-template-columns:repeat(3,1fr);gap:6px;
  margin-bottom:12px;
}
.cp-mini-stat {
  padding:7px 6px;text-align:center;
  border:1px solid rgba(255,255,255,0.06);
  border-radius:2px;
  background:rgba(255,255,255,0.02);
}
.cp-mini-val {
  font-family:'JetBrains Mono',monospace;
  font-size:0.85rem;font-weight:700;line-height:1;
}
.cp-mini-lbl {
  font-family:'JetBrains Mono',monospace;
  font-size:0.45rem;letter-spacing:2px;
  color:rgba(255,255,255,0.25);margin-top:3px;
  text-transform:uppercase;
}

.cp-pred {
  display:flex;align-items:center;justify-content:space-between;
  padding:8px 12px;
  background:rgba(255,255,255,0.03);
  border:1px solid rgba(255,255,255,0.07);
  border-radius:2px;
  margin-top:2px;
}
.cp-pred-label {
  font-family:'JetBrains Mono',monospace;
  font-size:0.5rem;letter-spacing:3px;
  color:rgba(255,255,255,0.25);text-transform:uppercase;
}
.cp-pred-dir {
  font-family:'Rajdhani',sans-serif;
  font-size:1rem;font-weight:700;letter-spacing:2px;
  display:flex;align-items:center;gap:6px;
}
.cp-pred-conf {
  font-family:'JetBrains Mono',monospace;
  font-size:0.55rem;color:rgba(255,255,255,0.3);
}

/* BOTTOM PANELS */
.cp-panel {
  background:var(--panel-bg);
  border:1px solid var(--panel-border);
  border-radius:4px;padding:24px;
  position:relative;overflow:hidden;
}
.cp-panel::before {
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--neon-cyan),transparent);
}
.cp-panel-title {
  font-family:'JetBrains Mono',monospace;
  font-size:0.58rem;letter-spacing:4px;
  color:var(--neon-cyan);opacity:0.6;
  text-transform:uppercase;margin-bottom:18px;
  display:flex;align-items:center;gap:10px;
}
.cp-panel-title::after{content:'';flex:1;height:1px;background:rgba(0,245,255,0.1)}

.cp-post {
  display:flex;gap:12px;padding:10px 12px;
  border:1px solid rgba(255,255,255,0.05);
  border-radius:2px;margin-bottom:6px;
  background:rgba(255,255,255,0.02);
  transition:border-color 0.2s,background 0.2s;
}
.cp-post:hover{border-color:rgba(0,245,255,0.2);background:rgba(0,245,255,0.03)}
.cp-post-badge {
  font-family:'JetBrains Mono',monospace;
  font-size:0.5rem;letter-spacing:1px;
  padding:2px 7px;border-radius:2px;
  white-space:nowrap;align-self:flex-start;margin-top:2px;border:1px solid;
}
.cp-post-title {font-size:0.8rem;line-height:1.5;margin-bottom:4px}
.cp-post-meta {font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:rgba(255,255,255,0.25)}

@keyframes cpFadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.cp-anim{animation:cpFadeUp 0.5s ease both}

@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&display=swap');
</style>
</head><body>

<!-- BG LAYERS -->
<div class="cp-grid-bg"></div>
<div class="cp-scanline"></div>
<div class="cp-glow-orb" style="width:600px;height:600px;background:rgba(0,245,255,0.06);top:-100px;left:-100px;animation-delay:0s"></div>
<div class="cp-glow-orb" style="width:500px;height:500px;background:rgba(255,45,120,0.05);bottom:-100px;right:-100px;animation-delay:-8s"></div>
<div class="cp-glow-orb" style="width:400px;height:400px;background:rgba(0,255,159,0.04);top:50%;left:50%;animation-delay:-4s"></div>

{nav}

<div style="position:relative;z-index:1">

  <!-- HEADER -->
  <div class="cp-header cp-anim">
    <div class="cp-title-block">
      <div class="cp-eyebrow">◈ LIVE MARKET INTELLIGENCE</div>
      <div class="cp-title">CRYPTO <em>PULSE</em></div>
    </div>
    <div class="cp-timestamp">
      <div id="liveTime" style="font-size:0.7rem;color:var(--neon-cyan);margin-bottom:4px"></div>
      <div id="lastUpdated" style="color:rgba(0,245,255,0.35)">LAST UPDATED: —</div>
    </div>
  </div>

  <!-- TICKER STRIP -->
  <div class="cp-ticker cp-anim" style="animation-delay:0.05s" id="tickerRow">
    <div class="cp-tick"><div class="cp-tick-label">Total Posts</div><div class="cp-tick-val" id="tkTotal" style="color:var(--neon-cyan)">—</div></div>
    <div class="cp-tick"><div class="cp-tick-label">Avg Bullish</div><div class="cp-tick-val" id="tkBull" style="color:var(--neon-green)">—</div></div>
    <div class="cp-tick"><div class="cp-tick-label">Avg Bearish</div><div class="cp-tick-val" id="tkBear" style="color:var(--neon-pink)">—</div></div>
    <div class="cp-tick"><div class="cp-tick-label">UP Signals</div><div class="cp-tick-val" id="tkPred" style="color:var(--neon-purple)">—</div></div>
    <div class="cp-tick"><div class="cp-tick-label">BTC Price</div><div class="cp-tick-val" id="tkBtc" style="color:#F7931A">—</div></div>
  </div>

  <!-- COIN GRID -->
  <div style="padding:0 40px 28px">
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;letter-spacing:4px;color:rgba(0,245,255,0.4);text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:12px">
      <span>◈ SENTIMENT GRID</span>
      <span style="flex:1;height:1px;background:rgba(0,245,255,0.1);display:block"></span>
      <span id="gridStatus" style="color:rgba(0,245,255,0.3)">5 COINS TRACKED</span>
    </div>
    <div id="coinGrid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px"></div>
  </div>

  <!-- BOTTOM PANELS -->
  <div style="padding:0 40px 60px;display:grid;grid-template-columns:1fr 1.2fr;gap:14px" class="cp-anim" style="animation-delay:0.25s">

    <!-- CHART -->
    <div class="cp-panel">
      <div class="cp-panel-title">Sentiment Distribution</div>
      <div style="position:relative;height:260px"><canvas id="radarChart"></canvas></div>
    </div>

    <!-- POSTS -->
    <div class="cp-panel">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px">
        <div class="cp-panel-title" style="margin-bottom:0">Latest Intel</div>
        <a href="/posts" style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;letter-spacing:2px;color:var(--neon-cyan);opacity:0.6;text-decoration:none;border:1px solid rgba(0,245,255,0.2);padding:4px 10px;border-radius:2px;transition:all 0.2s" onmouseover="this.style.opacity='1'" onmouseout="this.style.opacity='0.6'">VIEW ALL →</a>
      </div>
      <div id="postList" style="display:flex;flex-direction:column;max-height:300px;overflow-y:auto"></div>
    </div>

  </div>
</div>

<script>
let radarChart=null;
const COIN_ORDER=['bitcoin','ethereum','solana','tether','binancecoin'];
const COIN_COLORS={'bitcoin':'#F7931A','ethereum':'#627EEA','solana':'#9945FF','tether':'#26A17B','binancecoin':'#F3BA2F'};
const SENT_COLORS={BULLISH:'#00ff9f',BEARISH:'#ff2d78',NEUTRAL:'#ffe600'};

function fmt(p){
  if(!p)return'—';
  if(p>=1000)return'$'+p.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
  return'$'+p.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:4});
}

// Live clock
function updateClock(){
  const now=new Date();
  document.getElementById('liveTime').textContent=now.toLocaleString()+' (LOCAL)';
}
setInterval(updateClock,1000);updateClock();

function renderCoins(coins){
  const grid=document.getElementById('coinGrid');
  grid.innerHTML='';
  let totalPosts=0,bullSum=0,bearSum=0,upCount=0;

  COIN_ORDER.forEach(function(k,i){
    const c=coins[k];if(!c)return;
    const color=COIN_COLORS[k]||'#00f5ff';
    const up=(c.change24||0)>=0;
    const pct=c.percentages||{};
    const pred=c.prediction;
    totalPosts+=(c.total||0);
    bullSum+=(pct.BULLISH||0);
    bearSum+=(pct.BEARISH||0);
    if(pred&&pred.direction==='UP')upCount++;

    const sc=SENT_COLORS[c.overall]||'#fff';
    const sentBorderColor=c.overall==='BULLISH'?'rgba(0,255,159,0.3)':c.overall==='BEARISH'?'rgba(255,45,120,0.3)':'rgba(255,230,0,0.3)';
    const sentBgColor=c.overall==='BULLISH'?'rgba(0,255,159,0.08)':c.overall==='BEARISH'?'rgba(255,45,120,0.08)':'rgba(255,230,0,0.08)';

    let predHtml='';
    if(pred){
      const dc=pred.direction==='UP'?'#00ff9f':pred.direction==='DOWN'?'#ff2d78':'#ffe600';
      const de=pred.direction==='UP'?'▲ UP':pred.direction==='DOWN'?'▼ DOWN':'→ FLAT';
      predHtml='<div class="cp-pred">'
        +'<span class="cp-pred-label">24H FORECAST</span>'
        +'<span class="cp-pred-dir" style="color:'+dc+'">'+de+'</span>'
        +'<span class="cp-pred-conf">'+pred.confidence+'% CONF</span>'
        +'</div>';
    }

    const card=document.createElement('div');
    card.className='cp-coin cp-anim';
    card.style.setProperty('--coin-color', color);
    card.style.animationDelay=(i*0.07)+'s';
    card.innerHTML=''
      +'<div class="cp-coin-glow" style="background:'+color+'"></div>'
      +'<div class="cp-coin-header">'
        +'<div>'
          +'<div class="cp-coin-symbol" style="color:'+color+';text-shadow:0 0 20px '+color+'66">'+c.symbol+'</div>'
          +'<div class="cp-coin-name">'+c.label+'</div>'
        +'</div>'
        +'<div>'
          +'<div class="cp-coin-price" style="color:'+color+'">'+fmt(c.price)+'</div>'
          +'<div class="cp-coin-change" style="color:'+(up?'#00ff9f':'#ff2d78')+'">'+(up?'▲ +':'▼ ')+(c.change24||0)+'%</div>'
        +'</div>'
      +'</div>'
      +'<div class="cp-sent-badge" style="color:'+sc+';border-color:'+sentBorderColor+';background:'+sentBgColor+'">'+c.overall+'</div>'
      +'<div class="cp-bar-wrap">'
        +'<div class="cp-bar-labels"><span class="cp-bar-lbl">BULL '+(pct.BULLISH||0)+'%</span><span class="cp-bar-lbl">BEAR '+(pct.BEARISH||0)+'%</span></div>'
        +'<div class="cp-bar-track">'
          +'<div id="cpbar-'+k+'" class="cp-bar-fill" style="width:0%;background:linear-gradient(90deg,'+color+'88,'+color+');box-shadow:0 0 8px '+color+'66"></div>'
        +'</div>'
      +'</div>'
      +'<div class="cp-mini-grid">'
        +'<div class="cp-mini-stat"><div class="cp-mini-val" style="color:#00ff9f">'+(pct.BULLISH||0)+'%</div><div class="cp-mini-lbl">Bull</div></div>'
        +'<div class="cp-mini-stat"><div class="cp-mini-val" style="color:#ff2d78">'+(pct.BEARISH||0)+'%</div><div class="cp-mini-lbl">Bear</div></div>'
        +'<div class="cp-mini-stat"><div class="cp-mini-val" style="color:#ffe600">'+(pct.NEUTRAL||0)+'%</div><div class="cp-mini-lbl">Neut</div></div>'
      +'</div>'
      +predHtml;

    grid.appendChild(card);
    setTimeout(function(){
      const bar=document.getElementById('cpbar-'+k);
      if(bar)bar.style.width=(pct.BULLISH||0)+'%';
    },400+i*80);
  });

  // ticker
  const n=COIN_ORDER.filter(function(k){return coins[k]}).length||1;
  document.getElementById('tkTotal').textContent=totalPosts.toLocaleString();
  document.getElementById('tkBull').textContent=Math.round(bullSum/n)+'%';
  document.getElementById('tkBear').textContent=Math.round(bearSum/n)+'%';
  document.getElementById('tkPred').textContent=upCount+' / '+n;
  if(coins.bitcoin&&coins.bitcoin.price)
    document.getElementById('tkBtc').textContent=fmt(coins.bitcoin.price);
}

function renderRadar(coins){
  const ctx=document.getElementById('radarChart').getContext('2d');
  if(radarChart)radarChart.destroy();
  const labels=COIN_ORDER.map(function(k){return coins[k]?coins[k].symbol:''});
  const bull=COIN_ORDER.map(function(k){return coins[k]?(coins[k].percentages||{}).BULLISH||0:0});
  const bear=COIN_ORDER.map(function(k){return coins[k]?(coins[k].percentages||{}).BEARISH||0:0});
  radarChart=new Chart(ctx,{
    type:'radar',
    data:{labels:labels,datasets:[
      {label:'Bullish %',data:bull,backgroundColor:'rgba(0,255,159,0.1)',borderColor:'#00ff9f',borderWidth:2,pointBackgroundColor:'#00ff9f',pointRadius:4,pointBorderColor:'#00ff9f'},
      {label:'Bearish %',data:bear,backgroundColor:'rgba(255,45,120,0.08)',borderColor:'#ff2d78',borderWidth:2,pointBackgroundColor:'#ff2d78',pointRadius:4,pointBorderColor:'#ff2d78'},
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      scales:{r:{
        ticks:{color:'rgba(0,245,255,0.3)',font:{family:'JetBrains Mono',size:9},backdropColor:'transparent',stepSize:25},
        grid:{color:'rgba(0,245,255,0.07)'},
        angleLines:{color:'rgba(0,245,255,0.07)'},
        pointLabels:{color:'rgba(255,255,255,0.6)',font:{family:'Rajdhani',size:13,weight:'600'}}
      }},
      plugins:{
        legend:{labels:{color:'rgba(255,255,255,0.3)',font:{family:'JetBrains Mono',size:10},usePointStyle:true}},
        tooltip:{backgroundColor:'rgba(4,8,20,0.95)',borderColor:'rgba(0,245,255,0.2)',borderWidth:1,titleFont:{family:'JetBrains Mono',size:10},bodyFont:{family:'JetBrains Mono',size:10}}
      }
    }
  });
}

function renderPosts(posts){
  const list=document.getElementById('postList');
  list.innerHTML='';
  (posts||[]).slice(0,8).forEach(function(p,i){
    const sc=p.compound>0?'#00ff9f':p.compound<0?'#ff2d78':'#ffe600';
    const bg={BULLISH:'rgba(0,255,159,0.08)',BEARISH:'rgba(255,45,120,0.08)',NEUTRAL:'rgba(255,230,0,0.08)'}[p.label]||'';
    const bc={BULLISH:'rgba(0,255,159,0.25)',BEARISH:'rgba(255,45,120,0.25)',NEUTRAL:'rgba(255,230,0,0.25)'}[p.label]||'';
    const tc={BULLISH:'#00ff9f',BEARISH:'#ff2d78',NEUTRAL:'#ffe600'}[p.label]||'#fff';
    const el=document.createElement('div');
    el.className='cp-post';
    el.style.animationDelay=(i*0.04)+'s';
    el.innerHTML=''
      +'<span class="cp-post-badge" style="background:'+bg+';color:'+tc+';border-color:'+bc+'">'+p.label+'</span>'
      +'<div style="flex:1;min-width:0">'
        +'<div class="cp-post-title" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'+p.title+'</div>'
        +'<div class="cp-post-meta">'
          +'<span>'+p.source+'</span>'
          +'<span style="margin-left:12px;color:'+sc+'">'+(p.compound>0?'+':'')+p.compound.toFixed(3)+'</span>'
        +'</div>'
      +'</div>';
    list.appendChild(el);
  });
}

async function load(showLoader){
  if(showLoader)document.getElementById('loader').classList.remove('gone');
  document.getElementById('refreshBtn').disabled=true;
  try{
    let d=await(await fetch('/api/sentiment')).json();
    while(d.loading){await new Promise(r=>setTimeout(r,2000));d=await(await fetch('/api/sentiment')).json();}
    if(d.data){
      if(d.data.coins){renderCoins(d.data.coins);renderRadar(d.data.coins);}
      if(d.data.posts)renderPosts(d.data.posts);
      document.getElementById('lastUpdated').textContent='LAST UPDATED: '+(d.last_updated||'—');
    }
  }catch(e){console.error(e)}
  document.getElementById('loader').classList.add('gone');
  document.getElementById('refreshBtn').disabled=false;
}
async function manualRefresh(){document.getElementById('refreshBtn').disabled=true;await fetch('/api/refresh',{method:'POST'});await load(true);}
setInterval(function(){load(false)},300000);
load(true);
</script>
</body></html>"""

# ═══════════════════════════════════════════
#  COINS PAGE
# ═══════════════════════════════════════════
COINS_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Pulse — Coins</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>{css}
.coin-card{background:var(--surface);border:1px solid var(--border2);border-radius:20px;padding:28px;position:relative;overflow:hidden;transition:transform 0.25s}
.coin-card:hover{transform:translateY(-3px)}
</style>
</head><body>
{nav}
<div class="page">
  <div class="anim" style="margin-bottom:28px">
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:8px">All Coins</div>
    <h1 style="font-size:2rem;font-weight:800;letter-spacing:-1px">Multi-Coin <span style="background:linear-gradient(90deg,#F7931A,#9945FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Sentiment</span></h1>
  </div>
  <div id="coinsGrid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px"></div>
</div>
<script>
const COIN_COLORS={'bitcoin':'#F7931A','ethereum':'#627EEA','solana':'#9945FF','tether':'#26A17B','binancecoin':'#F3BA2F'};
const COIN_ORDER=['bitcoin','ethereum','solana','tether','binancecoin'];
function hexToRgb(hex){const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return r+','+g+','+b}
function fmt(p){if(!p)return'—';if(p>=1000)return'$'+p.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});return'$'+p.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:4})}
function sentBg(s){return{BULLISH:'rgba(34,197,94,0.1)',BEARISH:'rgba(239,68,68,0.1)',NEUTRAL:'rgba(234,179,8,0.1)'}[s]||''}
function sentClr(s){return{BULLISH:'#22c55e',BEARISH:'#ef4444',NEUTRAL:'#eab308'}[s]||'#888'}
function dirEmoji(d){return{UP:'📈',DOWN:'📉',SIDEWAYS:'➡️'}[d]||'➡️'}
function dirClr(d){return{UP:'#22c55e',DOWN:'#ef4444',SIDEWAYS:'#eab308'}[d]||'#888'}

function render(coins){
  const grid=document.getElementById('coinsGrid');
  grid.innerHTML='';
  COIN_ORDER.forEach(function(k,i){
    const c=coins[k];
    if(!c)return;
    const color=COIN_COLORS[k]||'#888';
    const rgb=hexToRgb(color);
    const up=(c.change24||0)>=0;
    const pct=c.percentages||{};
    const pred=c.prediction;
    const card=document.createElement('div');
    card.className='coin-card anim';
    card.id='coin-'+k;
    card.style.animationDelay=(i*0.07)+'s';
    card.style.borderColor='rgba('+rgb+',0.2)';
    let predHtml='';
    if(pred){
      predHtml='<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:var(--surface2);border-radius:10px;border:1px solid var(--border);margin-top:14px">'
        +'<span style="font-size:0.7rem;color:var(--muted);font-weight:600">24H FORECAST</span>'
        +'<div style="display:flex;align-items:center;gap:8px">'
        +'<span style="font-size:1.1rem">'+dirEmoji(pred.direction)+'</span>'
        +'<span style="font-weight:700;color:'+dirClr(pred.direction)+'">'+pred.direction+'</span>'
        +'<span style="font-size:0.7rem;color:var(--muted);font-family:JetBrains Mono,monospace">'+pred.confidence+'% conf</span>'
        +'</div></div>';
    }
    card.innerHTML=''
      +'<div style="position:absolute;top:-30px;right:-30px;width:140px;height:140px;border-radius:50%;background:'+color+';filter:blur(60px);opacity:0.1;pointer-events:none"></div>'
      +'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">'
        +'<div style="display:flex;align-items:center;gap:12px">'
          +'<div style="width:48px;height:48px;border-radius:14px;background:rgba('+rgb+',0.15);border:1px solid rgba('+rgb+',0.3);display:flex;align-items:center;justify-content:center;font-size:1.4rem">'+c.emoji+'</div>'
          +'<div><div style="font-weight:800;font-size:1.2rem;letter-spacing:-0.3px">'+c.symbol+'</div>'
          +'<div style="font-size:0.72rem;color:var(--muted);font-weight:500">'+c.label+'</div></div>'
        +'</div>'
        +'<div style="text-align:right">'
          +'<div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;font-weight:700;color:'+color+'">'+fmt(c.price)+'</div>'
          +'<div style="font-size:0.72rem;font-weight:600;color:'+(up?'#22c55e':'#ef4444')+'">'+(up?'▲ +':'▼ ')+(c.change24||0)+'%</div>'
        +'</div>'
      +'</div>'
      +'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">'
        +'<span style="font-size:0.72rem;font-weight:700;padding:4px 12px;border-radius:20px;background:'+sentBg(c.overall)+';color:'+sentClr(c.overall)+';border:1px solid '+sentClr(c.overall)+'33">'+c.overall+'</span>'
        +'<span style="font-size:0.7rem;color:var(--muted);font-family:JetBrains Mono,monospace">'+(c.total||0)+' posts · score '+(c.avg_score||0)+'</span>'
      +'</div>'
      +'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px">'
        +'<div style="text-align:center;padding:10px;background:rgba(34,197,94,0.07);border-radius:10px;border:1px solid rgba(34,197,94,0.15)"><div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;font-weight:700;color:#22c55e">'+(pct.BULLISH||0)+'%</div><div style="font-size:0.55rem;color:var(--muted);margin-top:3px;font-weight:600;letter-spacing:1px">BULLISH</div></div>'
        +'<div style="text-align:center;padding:10px;background:rgba(239,68,68,0.07);border-radius:10px;border:1px solid rgba(239,68,68,0.15)"><div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;font-weight:700;color:#ef4444">'+(pct.BEARISH||0)+'%</div><div style="font-size:0.55rem;color:var(--muted);margin-top:3px;font-weight:600;letter-spacing:1px">BEARISH</div></div>'
        +'<div style="text-align:center;padding:10px;background:rgba(234,179,8,0.07);border-radius:10px;border:1px solid rgba(234,179,8,0.15)"><div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;font-weight:700;color:#eab308">'+(pct.NEUTRAL||0)+'%</div><div style="font-size:0.55rem;color:var(--muted);margin-top:3px;font-weight:600;letter-spacing:1px">NEUTRAL</div></div>'
      +'</div>'
      +'<div style="height:4px;background:var(--surface2);border-radius:2px;overflow:hidden">'
        +'<div style="display:flex;height:100%">'
          +'<div style="width:'+(pct.BULLISH||0)+'%;background:#22c55e;transition:width 1s ease"></div>'
          +'<div style="width:'+(pct.BEARISH||0)+'%;background:#ef4444;transition:width 1s ease"></div>'
          +'<div style="width:'+(pct.NEUTRAL||0)+'%;background:#eab308;transition:width 1s ease"></div>'
        +'</div>'
      +'</div>'
      +predHtml;
    grid.appendChild(card);
  });
  if(location.hash){var el=document.querySelector('#coin-'+location.hash.slice(1));if(el)setTimeout(function(){el.scrollIntoView({behavior:'smooth'})},400)}
}

async function load(){
  try{
    let d=await(await fetch('/api/sentiment')).json();
    if(d.data&&d.data.coins){render(d.data.coins);document.getElementById('loader').classList.add('gone');return;}
    document.getElementById('loader').classList.remove('gone');
    let attempts=0;
    while(d.loading&&attempts<15){await new Promise(r=>setTimeout(r,2000));d=await(await fetch('/api/sentiment')).json();attempts++;}
    if(d.data&&d.data.coins)render(d.data.coins);
  }catch(e){console.error(e)}
  document.getElementById('loader').classList.add('gone');
}
async function manualRefresh(){document.getElementById('refreshBtn').disabled=true;await fetch('/api/refresh',{method:'POST'});await load();document.getElementById('refreshBtn').disabled=false}
load();
</script>
</body></html>"""

# ═══════════════════════════════════════════
#  PREDICT PAGE
# ═══════════════════════════════════════════
PREDICT_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Pulse — Predict</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>{css}
.pred-card{background:var(--surface);border:1px solid var(--border2);border-radius:20px;padding:28px;position:relative;overflow:hidden;transition:transform 0.25s}
.pred-card:hover{transform:translateY(-3px)}
</style>
</head><body>
{nav}
<div class="page">
  <div class="anim" style="margin-bottom:12px">
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:8px">AI Predictions</div>
    <h1 style="font-size:2rem;font-weight:800;letter-spacing:-1px">24h Price <span style="background:linear-gradient(90deg,#9945FF,#627EEA);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Forecast</span></h1>
  </div>
  <div style="background:rgba(234,179,8,0.08);border:1px solid rgba(234,179,8,0.2);border-radius:12px;padding:12px 18px;margin-bottom:28px" class="anim">
    <span style="font-size:0.72rem;color:#eab308;font-weight:500">⚠ Predictions are based on Reddit sentiment + price momentum only. Not financial advice. Always DYOR.</span>
  </div>
  <div id="predGrid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px;margin-bottom:32px"></div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px" class="anim">
    <div class="card" style="padding:28px">
      <div class="label" style="margin-bottom:20px">Confidence by Coin</div>
      <div style="position:relative;height:260px"><canvas id="confChart"></canvas></div>
    </div>
    <div class="card" style="padding:28px">
      <div class="label" style="margin-bottom:20px">How It Works</div>
      <div style="display:flex;flex-direction:column;gap:16px">
        <div style="display:flex;gap:12px;align-items:flex-start"><div style="width:36px;height:36px;border-radius:10px;background:#F7931A18;border:1px solid #F7931A33;display:flex;align-items:center;justify-content:center;flex-shrink:0">📊</div><div><div style="font-weight:700;font-size:0.85rem;color:#F7931A;margin-bottom:3px">Sentiment Signal (50%)</div><div style="font-size:0.78rem;color:var(--muted);line-height:1.5">VADER compound score from Reddit posts. More bullish posts = higher signal.</div></div></div>
        <div style="display:flex;gap:12px;align-items:flex-start"><div style="width:36px;height:36px;border-radius:10px;background:#9945FF18;border:1px solid #9945FF33;display:flex;align-items:center;justify-content:center;flex-shrink:0">⚡</div><div><div style="font-weight:700;font-size:0.85rem;color:#9945FF;margin-bottom:3px">Momentum Signal (30%)</div><div style="font-size:0.78rem;color:var(--muted);line-height:1.5">24-hour price change amplified ×5. Captures short-term momentum.</div></div></div>
        <div style="display:flex;gap:12px;align-items:flex-start"><div style="width:36px;height:36px;border-radius:10px;background:#627EEA18;border:1px solid #627EEA33;display:flex;align-items:center;justify-content:center;flex-shrink:0">📈</div><div><div style="font-weight:700;font-size:0.85rem;color:#627EEA;margin-bottom:3px">Trend Signal (20%)</div><div style="font-size:0.78rem;color:var(--muted);line-height:1.5">7-day price direction ×2. Filters noise, captures broader trend.</div></div></div>
        <div style="display:flex;gap:12px;align-items:flex-start"><div style="width:36px;height:36px;border-radius:10px;background:#26A17B18;border:1px solid #26A17B33;display:flex;align-items:center;justify-content:center;flex-shrink:0">🎯</div><div><div style="font-weight:700;font-size:0.85rem;color:#26A17B;margin-bottom:3px">Confidence Score</div><div style="font-size:0.78rem;color:var(--muted);line-height:1.5">How aligned are all 3 signals. Full agreement = 82% confidence.</div></div></div>
      </div>
    </div>
  </div>
</div>
<script>
const COIN_COLORS={'bitcoin':'#F7931A','ethereum':'#627EEA','solana':'#9945FF','tether':'#26A17B','binancecoin':'#F3BA2F'};
const COIN_ORDER=['bitcoin','ethereum','solana','tether','binancecoin'];
let confChart=null;
function fmt(p){if(!p)return'—';if(p>=1000)return'$'+p.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});return'$'+p.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:4})}
function dirEmoji(d){return{UP:'📈',DOWN:'📉',SIDEWAYS:'➡️'}[d]||'➡️'}
function dirClr(d){return{UP:'#22c55e',DOWN:'#ef4444',SIDEWAYS:'#eab308'}[d]||'#888'}

function render(coins){
  const grid=document.getElementById('predGrid');
  grid.innerHTML='';
  const confLabels=[],confData=[],confBgColors=[];
  COIN_ORDER.forEach(function(k,i){
    const c=coins[k];
    if(!c||!c.prediction)return;
    const p=c.prediction;
    const color=COIN_COLORS[k]||'#888';
    const dc=dirClr(p.direction);
    confLabels.push(c.symbol);confData.push(p.confidence);confBgColors.push(color+'bb');
    const sBar=Math.max(0,Math.min(100,(p.sent_signal+100)/2));
    const mBar=Math.max(0,Math.min(100,(p.mom_signal+100)/2));
    const tBar=Math.max(0,Math.min(100,(p.trend_signal+100)/2));
    const sClr=p.sent_signal>0?'#22c55e':p.sent_signal<0?'#ef4444':'#eab308';
    const mClr=p.mom_signal>0?'#22c55e':p.mom_signal<0?'#ef4444':'#eab308';
    const tClr=p.trend_signal>0?'#22c55e':p.trend_signal<0?'#ef4444':'#eab308';
    const card=document.createElement('div');
    card.className='pred-card anim';
    card.style.animationDelay=(i*0.07)+'s';
    card.style.borderColor=color+'33';
    card.innerHTML=''
      +'<div style="position:absolute;top:-20px;right:-20px;width:100px;height:100px;border-radius:50%;background:'+color+';filter:blur(50px);opacity:0.12;pointer-events:none"></div>'
      +'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">'
        +'<div style="display:flex;align-items:center;gap:10px">'
          +'<span style="font-size:1.5rem">'+c.emoji+'</span>'
          +'<div><div style="font-weight:800;font-size:1.1rem">'+c.symbol+'</div>'
          +'<div style="font-size:0.7rem;color:var(--muted);font-family:JetBrains Mono,monospace">'+fmt(p.current_price)+'</div></div>'
        +'</div>'
        +'<div style="text-align:right">'
          +'<div style="display:flex;align-items:center;gap:6px;justify-content:flex-end">'
            +'<span style="font-size:1.2rem">'+dirEmoji(p.direction)+'</span>'
            +'<span style="font-size:1rem;font-weight:800;color:'+dc+'">'+p.direction+'</span>'
          +'</div>'
          +'<div style="font-size:0.65rem;color:var(--muted);font-family:JetBrains Mono,monospace">'+p.confidence+'% confidence</div>'
        +'</div>'
      +'</div>'
      +'<div style="margin-bottom:18px">'
        +'<div style="display:flex;justify-content:space-between;margin-bottom:6px"><span style="font-size:0.65rem;font-weight:600;color:var(--muted);letter-spacing:1px">CONFIDENCE</span><span style="font-size:0.65rem;font-family:JetBrains Mono,monospace;color:'+dc+'">'+p.confidence+'%</span></div>'
        +'<div style="height:6px;background:var(--surface2);border-radius:3px;overflow:hidden"><div style="height:100%;width:'+p.confidence+'%;background:'+color+';border-radius:3px;transition:width 1s ease;box-shadow:0 0 8px '+color+'66"></div></div>'
      +'</div>'
      +'<div style="background:'+dc+'11;border:1px solid '+dc+'33;border-radius:12px;padding:14px;margin-bottom:18px;text-align:center">'
        +'<div style="font-size:0.6rem;font-weight:700;letter-spacing:2px;color:var(--muted);margin-bottom:8px">24H TARGET RANGE</div>'
        +'<div style="display:flex;align-items:center;justify-content:center;gap:16px">'
          +'<div><div style="font-size:0.6rem;color:var(--muted);margin-bottom:3px">LOW</div><div style="font-family:JetBrains Mono,monospace;font-size:1rem;font-weight:700;color:'+dc+'">'+fmt(p.target_lo)+'</div></div>'
          +'<div style="color:var(--muted);font-size:1.2rem">→</div>'
          +'<div><div style="font-size:0.6rem;color:var(--muted);margin-bottom:3px">HIGH</div><div style="font-family:JetBrains Mono,monospace;font-size:1rem;font-weight:700;color:'+dc+'">'+fmt(p.target_hi)+'</div></div>'
        +'</div>'
      +'</div>'
      +'<div style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;margin-bottom:4px"><span style="font-size:0.6rem;color:var(--muted);font-weight:600;letter-spacing:1px">SENTIMENT (50%)</span><span style="font-size:0.6rem;font-family:JetBrains Mono,monospace;color:'+sClr+'">'+(p.sent_signal>0?'+':'')+p.sent_signal+'</span></div><div style="height:4px;background:var(--surface2);border-radius:2px;overflow:hidden"><div style="height:100%;width:'+sBar+'%;background:'+sClr+';border-radius:2px"></div></div></div>'
      +'<div style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;margin-bottom:4px"><span style="font-size:0.6rem;color:var(--muted);font-weight:600;letter-spacing:1px">MOMENTUM (30%)</span><span style="font-size:0.6rem;font-family:JetBrains Mono,monospace;color:'+mClr+'">'+(p.mom_signal>0?'+':'')+p.mom_signal+'</span></div><div style="height:4px;background:var(--surface2);border-radius:2px;overflow:hidden"><div style="height:100%;width:'+mBar+'%;background:'+mClr+';border-radius:2px"></div></div></div>'
      +'<div><div style="display:flex;justify-content:space-between;margin-bottom:4px"><span style="font-size:0.6rem;color:var(--muted);font-weight:600;letter-spacing:1px">7D TREND (20%)</span><span style="font-size:0.6rem;font-family:JetBrains Mono,monospace;color:'+tClr+'">'+(p.trend_signal>0?'+':'')+p.trend_signal+'</span></div><div style="height:4px;background:var(--surface2);border-radius:2px;overflow:hidden"><div style="height:100%;width:'+tBar+'%;background:'+tClr+';border-radius:2px"></div></div></div>';
    grid.appendChild(card);
  });
  const ctx=document.getElementById('confChart').getContext('2d');
  if(confChart)confChart.destroy();
  confChart=new Chart(ctx,{type:'bar',data:{labels:confLabels,datasets:[{label:'Confidence %',data:confData,backgroundColor:confBgColors,borderColor:confBgColors,borderWidth:1,borderRadius:8}]},options:{responsive:true,maintainAspectRatio:false,scales:{x:{ticks:{color:'#555568',font:{family:'Outfit',size:11,weight:'600'}},grid:{color:'rgba(255,255,255,0.04)'}},y:{min:0,max:100,ticks:{color:'#555568',font:{family:'JetBrains Mono',size:10},callback:function(v){return v+'%'}},grid:{color:'rgba(255,255,255,0.04)'}}},plugins:{legend:{display:false},tooltip:{backgroundColor:'rgba(22,22,31,0.95)',borderColor:'rgba(255,255,255,0.1)',borderWidth:1}}}});
}

async function load(){
  try{
    let d=await(await fetch('/api/sentiment')).json();
    if(d.data&&d.data.coins){render(d.data.coins);document.getElementById('loader').classList.add('gone');return;}
    document.getElementById('loader').classList.remove('gone');
    let attempts=0;
    while(d.loading&&attempts<15){await new Promise(r=>setTimeout(r,2000));d=await(await fetch('/api/sentiment')).json();attempts++;}
    if(d.data&&d.data.coins)render(d.data.coins);
  }catch(e){console.error(e)}
  document.getElementById('loader').classList.add('gone');
}
async function manualRefresh(){document.getElementById('refreshBtn').disabled=true;await fetch('/api/refresh',{method:'POST'});await load();document.getElementById('refreshBtn').disabled=false}
load();
</script>
</body></html>"""

# ═══════════════════════════════════════════
#  POSTS PAGE
# ═══════════════════════════════════════════
POSTS_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Pulse — Posts</title>
<style>{css}
.fbtn{font-size:0.75rem;font-weight:600;padding:7px 16px;border-radius:8px;border:1px solid var(--border2);background:transparent;color:var(--muted);cursor:pointer;transition:all 0.2s}
.fbtn:hover,.fbtn.active{color:white;background:var(--surface2);border-color:rgba(255,255,255,0.2)}
.fbtn.f-bull.active{color:#22c55e;border-color:rgba(34,197,94,0.3);background:rgba(34,197,94,0.08)}
.fbtn.f-bear.active{color:#ef4444;border-color:rgba(239,68,68,0.3);background:rgba(239,68,68,0.08)}
.fbtn.f-neut.active{color:#eab308;border-color:rgba(234,179,8,0.3);background:rgba(234,179,8,0.08)}
.post-row{background:var(--surface);border:1px solid var(--border2);border-radius:14px;padding:16px 20px;display:flex;align-items:flex-start;gap:14px;transition:all 0.2s;animation:fadeUp 0.35s ease both}
.post-row:hover{transform:translateX(5px);border-color:rgba(255,255,255,0.15);background:var(--surface2)}
</style>
</head><body>
{nav}
<div class="page">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;flex-wrap:wrap;gap:12px" class="anim">
    <div>
      <div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:8px">Feed</div>
      <h1 style="font-size:2rem;font-weight:800;letter-spacing:-1px">Reddit <span style="background:linear-gradient(90deg,#F7931A,#627EEA);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Analysis</span></h1>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <button onclick="filter('ALL')" class="fbtn active" data-f="ALL">All</button>
      <button onclick="filter('BULLISH')" class="fbtn f-bull" data-f="BULLISH">🟢 Bullish</button>
      <button onclick="filter('BEARISH')" class="fbtn f-bear" data-f="BEARISH">🔴 Bearish</button>
      <button onclick="filter('NEUTRAL')" class="fbtn f-neut" data-f="NEUTRAL">⚪ Neutral</button>
    </div>
  </div>
  <div id="postGrid" style="display:flex;flex-direction:column;gap:10px"></div>
</div>
<script>
let allPosts=[],cur='ALL';
function sentBg(s){return{BULLISH:'rgba(34,197,94,0.1)',BEARISH:'rgba(239,68,68,0.1)',NEUTRAL:'rgba(234,179,8,0.1)'}[s]||''}
function sentClr(s){return{BULLISH:'#22c55e',BEARISH:'#ef4444',NEUTRAL:'#eab308'}[s]||'#888'}

function filter(f){
  cur=f;
  document.querySelectorAll('.fbtn').forEach(function(b){b.classList.remove('active');if(b.dataset.f===f)b.classList.add('active')});
  render();
}

function render(){
  const grid=document.getElementById('postGrid');
  const posts=cur==='ALL'?allPosts:allPosts.filter(function(p){return p.label===cur});
  grid.innerHTML='';
  if(!posts.length){grid.innerHTML='<div style="text-align:center;padding:60px;color:var(--muted);font-size:0.85rem">No posts found</div>';return}
  posts.forEach(function(p,i){
    const sc=p.compound>0?'#22c55e':p.compound<0?'#ef4444':'#eab308';
    const div=document.createElement('div');
    div.className='post-row';
    div.style.animationDelay=(i*0.02)+'s';
    div.innerHTML=''
      +'<div style="display:flex;flex-direction:column;align-items:center;gap:5px;flex-shrink:0;width:76px">'
        +'<span style="font-size:0.6rem;font-weight:700;padding:2px 8px;border-radius:6px;background:'+sentBg(p.label)+';color:'+sentClr(p.label)+';letter-spacing:0.5px;text-align:center">'+p.label+'</span>'
        +'<span style="font-size:0.7rem;font-family:JetBrains Mono,monospace;color:'+sc+'">'+(p.compound>0?'+':'')+p.compound.toFixed(3)+'</span>'
        +'<span style="font-size:0.6rem;color:var(--muted)">↑ '+p.upvotes+'</span>'
      +'</div>'
      +'<div style="flex:1;min-width:0;border-left:1px solid var(--border);padding-left:14px">'
        +'<div style="font-size:0.88rem;font-weight:500;line-height:1.5;margin-bottom:5px">'+p.title+'</div>'
        +'<div style="font-size:0.65rem;color:var(--muted);font-family:JetBrains Mono,monospace">'+p.source+'</div>'
      +'</div>';
    grid.appendChild(div);
  });
}

async function load(){
  try{
    let d=await(await fetch('/api/sentiment')).json();
    if(d.data){allPosts=d.data.posts||[];render();document.getElementById('loader').classList.add('gone');return;}
    document.getElementById('loader').classList.remove('gone');
    let attempts=0;
    while(d.loading&&attempts<15){await new Promise(r=>setTimeout(r,2000));d=await(await fetch('/api/sentiment')).json();attempts++;}
    if(d.data){allPosts=d.data.posts||[];render()}
  }catch(e){console.error(e)}
  document.getElementById('loader').classList.add('gone');
}
async function manualRefresh(){document.getElementById('refreshBtn').disabled=true;await fetch('/api/refresh',{method:'POST'});await load();document.getElementById('refreshBtn').disabled=false}
load();
</script>
</body></html>"""

# ═══════════════════════════════════════════
#  ABOUT PAGE
# ═══════════════════════════════════════════
ABOUT_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Pulse — About</title>
<style>{css}
.about-hero{{background:linear-gradient(135deg,rgba(247,147,26,0.08),rgba(153,69,255,0.08));border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:40px;margin-bottom:28px;position:relative;overflow:hidden}}
.about-hero::before{{content:'';position:absolute;top:-40px;right:-40px;width:200px;height:200px;border-radius:50%;background:radial-gradient(circle,rgba(247,147,26,0.15),transparent 70%);pointer-events:none}}
.about-section{{background:var(--surface);border:1px solid var(--border2);border-radius:20px;padding:32px;margin-bottom:16px}}
.about-section h2{{font-size:1.1rem;font-weight:700;margin-bottom:12px;display:flex;align-items:center;gap:10px}}
.about-section p{{font-size:0.88rem;color:#9090a8;line-height:1.8;margin-bottom:12px}}
.about-section p:last-child{{margin-bottom:0}}
.about-section strong{{color:var(--text);font-weight:600}}
.coin-pills{{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}}
.coin-pill{{display:flex;align-items:center;gap:7px;padding:6px 14px;border-radius:20px;font-size:0.78rem;font-weight:600;border:1px solid}}
.tech-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:16px}}
.tech-item{{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px 16px}}
.tech-item-name{{font-size:0.8rem;font-weight:700;color:var(--text);margin-bottom:3px}}
.tech-item-desc{{font-size:0.68rem;color:var(--muted)}}
</style>
</head><body>
{nav}
<div class="page" style="max-width:860px">

  <!-- HERO -->
  <div class="about-hero anim">
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:10px">About Crypto Pulse</div>
    <h1 style="font-size:2rem;font-weight:900;letter-spacing:-1px;line-height:1.3;margin-bottom:16px">Your real-time window into<br><span style="background:linear-gradient(90deg,#F7931A,#9945FF,#627EEA);-webkit-background-clip:text;-webkit-text-fill-color:transparent">crypto market sentiment</span></h1>
    <p style="font-size:0.92rem;color:#9090a8;line-height:1.8;max-width:600px">Crypto Pulse tracks what people are actually saying about crypto on Reddit — right now. It turns thousands of social media posts into clear, actionable sentiment signals so you can understand the mood of the market at a glance.</p>
  </div>

  <!-- WHAT IT DOES -->
  <div class="about-section anim" style="animation-delay:0.05s">
    <h2><span style="font-size:1.3rem">📡</span> <span style="color:#F7931A">Where the data comes from</span></h2>
    <p>Every few minutes, Crypto Pulse scans the most active crypto communities on Reddit. We pull the freshest posts from subreddits dedicated to each coin — places like r/Bitcoin, r/ethereum, r/solana, r/binance, and more — as well as general crypto communities like r/CryptoCurrency and r/CryptoMarkets.</p>
    <p>We focus on <strong>new posts</strong> rather than trending ones, because we want to capture what people are talking about right now, not what was popular yesterday. For each coin we collect around 15 posts per subreddit, giving us a solid sample of current community sentiment.</p>
    <div class="coin-pills">
      <div class="coin-pill" style="background:rgba(247,147,26,0.1);color:#F7931A;border-color:rgba(247,147,26,0.25)">₿ Bitcoin</div>
      <div class="coin-pill" style="background:rgba(99,126,234,0.1);color:#627EEA;border-color:rgba(99,126,234,0.25)">Ξ Ethereum</div>
      <div class="coin-pill" style="background:rgba(153,69,255,0.1);color:#9945FF;border-color:rgba(153,69,255,0.25)">◎ Solana</div>
      <div class="coin-pill" style="background:rgba(38,161,123,0.1);color:#26A17B;border-color:rgba(38,161,123,0.25)">₮ Tether</div>
      <div class="coin-pill" style="background:rgba(243,186,47,0.1);color:#F3BA2F;border-color:rgba(243,186,47,0.25)">⬡ BNB</div>
    </div>
  </div>

  <!-- HOW SENTIMENT WORKS -->
  <div class="about-section anim" style="animation-delay:0.1s">
    <h2><span style="font-size:1.3rem">🧠</span> <span style="color:#9945FF">How we measure sentiment</span></h2>
    <p>Each post is scored using <strong>VADER</strong> — a natural language processing model built specifically for social media text. VADER reads each post and assigns it a compound score between <strong>-1</strong> (extremely negative) and <strong>+1</strong> (extremely positive).</p>
    <p>But crypto Twitter and Reddit have their own language. Words like <strong>"moon"</strong>, <strong>"hodl"</strong>, and <strong>"ath"</strong> mean very different things in crypto than in everyday English. So on top of VADER, we layer a custom crypto keyword system that understands the community's language — boosting scores for bullish terms and reducing them for bearish ones like <strong>"rug"</strong>, <strong>"liquidation"</strong>, or <strong>"depeg"</strong>.</p>
    <p>A post scoring above <strong>+0.05</strong> is classified as Bullish. Below <strong>-0.05</strong> is Bearish. Everything in between is Neutral. The overall market mood for each coin is determined by averaging the scores of all its recent posts.</p>
  </div>

  <!-- USDT SPECIAL -->
  <div class="about-section anim" style="animation-delay:0.15s">
    <h2><span style="font-size:1.3rem">₮</span> <span style="color:#26A17B">Tether is different</span></h2>
    <p>USDT is a stablecoin — it's not supposed to go up or down in price. So measuring whether people are "bullish" on USDT doesn't make much sense. Instead, Crypto Pulse measures something more useful: <strong>peg health</strong>.</p>
    <p>For Tether, a Bullish sentiment means the community trusts the peg — people are talking about reserves, transparency, and stability. A Bearish sentiment means depeg anxiety is rising — discussions of lawsuits, insolvency fears, or FUD are dominating. This gives you an early warning signal before a potential peg event.</p>
  </div>

  <!-- PREDICTIONS -->
  <div class="about-section anim" style="animation-delay:0.2s">
    <h2><span style="font-size:1.3rem">🔮</span> <span style="color:#627EEA">How predictions are made</span></h2>
    <p>The 24-hour price forecast combines three signals into a single directional prediction. <strong>Sentiment (50%)</strong> — what Reddit is saying right now. <strong>Momentum (30%)</strong> — whether the price has been moving up or down in the last 24 hours. <strong>Trend (20%)</strong> — the broader 7-day price direction.</p>
    <p>When all three signals agree, confidence is high. When they point in different directions — for example, Reddit is bullish but the price has been dropping — confidence is lower and the prediction is treated with more caution.</p>
    <p style="color:#eab308;font-size:0.82rem">⚠ These predictions are indicators, not guarantees. Crypto is highly volatile and no model can reliably predict short-term price moves. Always do your own research before making any trading decisions.</p>
  </div>

  <!-- TECH -->
  <div class="about-section anim" style="animation-delay:0.25s">
    <h2><span style="font-size:1.3rem">⚙️</span> <span style="color:var(--muted)">Built with</span></h2>
    <div class="tech-grid">
      <div class="tech-item"><div class="tech-item-name" style="color:#F7931A">Python</div><div class="tech-item-desc">Core backend language</div></div>
      <div class="tech-item"><div class="tech-item-name" style="color:#9945FF">Flask</div><div class="tech-item-desc">Web server framework</div></div>
      <div class="tech-item"><div class="tech-item-name" style="color:#627EEA">VADER NLP</div><div class="tech-item-desc">Sentiment analysis engine</div></div>
      <div class="tech-item"><div class="tech-item-name" style="color:#26A17B">Chart.js</div><div class="tech-item-desc">Data visualizations</div></div>
      <div class="tech-item"><div class="tech-item-name" style="color:#F3BA2F">Reddit JSON API</div><div class="tech-item-desc">No API key needed</div></div>
      <div class="tech-item"><div class="tech-item-name" style="color:#F7931A">CoinGecko API</div><div class="tech-item-desc">Live price data</div></div>
    </div>
  </div>

</div>
<script>document.getElementById('loader').classList.add('gone');async function manualRefresh(){{document.getElementById('refreshBtn').disabled=true;await fetch('/api/refresh',{{method:'POST'}});document.getElementById('refreshBtn').disabled=false}}</script>
</body></html>"""

def make_card(icon, title, body, color, delay):
    return f"""<div class="card anim" style="padding:24px;display:flex;gap:16px;align-items:flex-start;border-color:{color}22;animation-delay:{delay}s">
      <div style="width:44px;height:44px;border-radius:12px;background:{color}15;border:1px solid {color}33;display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0">{icon}</div>
      <div><div style="font-weight:700;font-size:0.95rem;color:{color};margin-bottom:6px">{title}</div><div style="font-size:0.83rem;color:var(--muted);line-height:1.65">{body}</div></div></div>"""

def make_chip(name, color):
    return f'<div style="background:{color}12;border:1px solid {color}25;border-radius:10px;padding:10px;text-align:center"><div style="font-size:0.72rem;font-weight:700;color:{color};letter-spacing:0.5px">{name}</div></div>'

ABOUT_CARDS = (
    make_card('📡','Data Collection',"Scrapes 5 crypto subreddits (Bitcoin, Ethereum, Solana, Tether, BNB) plus r/CryptoCurrency and r/CryptoMarkets via Reddit's public JSON API. No API key needed.",'#F7931A',0.05)+
    make_card('🧠','Sentiment Analysis',"Each post is scored using VADER NLP with a crypto keyword boost layer. 'Moon', 'hodl', 'halving' push bullish; 'crash', 'rug', 'liquidation' push bearish.",'#9945FF',0.1)+
    make_card('🔮','Price Prediction',"Predictions combine 50% Reddit sentiment + 30% 24h price momentum + 20% 7-day trend. Confidence reflects how aligned all three signals are.",'#627EEA',0.15)+
    make_card('⚡','Auto Refresh',"Background thread re-scrapes and re-analyzes all 5 coins every 5 minutes automatically. Prices from CoinGecko free API update on every page load.",'#26A17B',0.2)
)
TECH_CHIPS = (make_chip('Python','#F7931A')+make_chip('Flask','#9945FF')+make_chip('VADER NLP','#627EEA')+make_chip('Chart.js','#26A17B')+make_chip('Reddit JSON','#F3BA2F')+make_chip('CoinGecko','#F7931A')+make_chip('5 Coins','#9945FF')+make_chip('Threading','#627EEA'))

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

# ═══════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════
@app.route("/")
def dashboard(): return Response(build_page(DASHBOARD_HTML, 'd'), mimetype='text/html')

@app.route("/coins")
def coins_page(): return Response(build_page(COINS_HTML, 'co'), mimetype='text/html')

@app.route("/predict")
def predict_page(): return Response(build_page(PREDICT_HTML, 'pr'), mimetype='text/html')

@app.route("/posts")
def posts(): return Response(build_page(POSTS_HTML, 'p'), mimetype='text/html')

@app.route("/about")
def about(): return Response(build_page(ABOUT_HTML, 'a', cards=ABOUT_CARDS, tech=TECH_CHIPS), mimetype='text/html')

@app.route("/api/sentiment")
def api_sentiment():
    return Response(jsonify({"loading":cache["loading"],"last_updated":cache["last_updated"],"data":cache["data"]}).get_data(), mimetype='application/json')

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    if not cache["loading"]:
        threading.Thread(target=run_analysis, daemon=True).start()
    return Response(jsonify({"status":"started"}).get_data(), mimetype='application/json')

import os
if __name__ == "__main__":
    print("🚀 Running initial analysis...")
    run_analysis()
    threading.Thread(target=bg_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
