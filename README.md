# ⬡ Crypto Pulse

> Real-time cryptocurrency sentiment tracker powered by Reddit NLP analysis

Crypto Pulse scans Reddit every 5 minutes across 5 major cryptocurrencies, runs natural language processing on thousands of posts, and delivers live sentiment signals so you can see what the crypto community is actually feeling right now.

**[🔴 Live Demo →](https://crypto-sentiment-fxp6.onrender.com)**

---

## What It Does

- **Live Sentiment Analysis** — Scrapes Reddit posts for BTC, ETH, SOL, USDT and BNB, scores each post using VADER NLP with a custom crypto keyword layer, and classifies the overall mood as Bullish, Bearish or Neutral
- **24h Price Predictions** — Combines Reddit sentiment (50%), 24h price momentum (30%) and 7-day trend (20%) into a directional forecast with a confidence score
- **USDT Peg Health** — Instead of price sentiment, Tether tracks community trust signals reserve backing, transparency discussions, and depeg FUD — giving you an early warning system for stablecoin risk
- **Auto Refresh** — Runs a background analysis cycle every 5 minutes automatically, no manual intervention needed
- **Live Prices** — Real-time price and 24h change data pulled from CoinGecko on every page load

---

## Pages

| Page | Description |
|------|-------------|
| `/` | Cyberpunk dashboard — all 5 coins in an equal grid with sentiment, price and prediction |
| `/coins` | Detailed per-coin breakdown with full sentiment percentages |
| `/predict` | 24h forecasts with confidence bars and signal breakdowns |
| `/posts` | Full Reddit post feed filterable by Bullish / Bearish / Neutral |
| `/about` | Plain-English explanation of how everything works |

---

## How Sentiment Works

Each post is scored using **VADER** (Valence Aware Dictionary and sEntiment Reasoner), a model built for social media text. On top of that, a custom crypto keyword layer boosts scores for community language — words like `moon`, `hodl`, `ath`, `halving` push the score bullish, while `crash`, `rug`, `liquidation`, `depeg` push it bearish.

- Score `>= +0.05` → **BULLISH**
- Score `<= -0.05` → **BEARISH**
- Everything else → **NEUTRAL**

---

## Tech Stack

- **Python** — Core backend
- **Flask** — Web framework
- **VADER NLP** — Sentiment analysis engine (`vaderSentiment`)
- **Reddit JSON API** — No API key required
- **CoinGecko API** — Free tier live price data
- **Chart.js** — Radar and bar chart visualizations
- **Threading** — Background refresh loop runs independently of the web server

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/4068-tiffany/crypto-sentiment.git
cd crypto-sentiment

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install flask requests vaderSentiment

# 4. Run
python app.py
```

Then open `http://localhost:5000` in your browser. The first load takes ~30 seconds while it scrapes and analyzes all 5 coins.

---

## Project Structure

```
crypto-sentiment/
├── app.py          # Main Flask app — routes, HTML templates, prediction engine
├── sentiment.py    # VADER analysis engine with optional CryptoBERT upgrade
└── README.md
```

---

## Upgrading to CryptoBERT (Optional)

`sentiment.py` includes support for [CryptoBERT](https://huggingface.co/kk08/CryptoBERT), a transformer model fine-tuned on crypto social media. To enable it:

```python
# In sentiment.py, change:
USE_BERT = False
# to:
USE_BERT = True
```

On first run it downloads ~500MB. Accuracy improves significantly but requires more RAM and slower analysis cycles.

---

## Accuracy

| Signal | Estimated Accuracy |
|--------|--------------------|
| Sentiment classification | ~65–70% |
| 24h price direction | ~52–55% |

Sentiment is a social signal, not a financial oracle. Use it as one input among many. The predictions page includes a disclaimer for this reason.

> ⚠ Not financial advice. Always do your own research before making any trading decisions.

---

## Deployment

Deployed on **Render** (free tier). Any push to `main` triggers an automatic redeploy.

Start command:
```
python app.py
```

Environment variable required on Render:
```
PORT=10000   (set automatically by Render)
```

---

## License

MIT
