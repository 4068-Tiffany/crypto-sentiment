# ============================================================
#  CONFIG — edit these to change coin or sources
# ============================================================

COIN = "Bitcoin"
COIN_SYMBOL = "BTC"

SUBREDDITS = [
    "Bitcoin",
    "CryptoCurrency",
    "BitcoinMarkets",
    "CryptoMarkets",
]

POSTS_PER_SUB = 25          # 25 x 4 subs = 100 posts total
POST_SORT = "new"           # new | hot | top | rising

# Extra crypto keywords to boost VADER scoring
BULLISH_KEYWORDS = [
    "moon", "bullish", "buy", "pump", "ath", "breakout",
    "hodl", "accumulate", "rally", "surge", "rocket", "🚀", "💎", "📈"
]

BEARISH_KEYWORDS = [
    "crash", "bearish", "sell", "dump", "rug", "scam", "dead",
    "panic", "drop", "fall", "short", "fear", "liquidation", "📉", "💀"
]
