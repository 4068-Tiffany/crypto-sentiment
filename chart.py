import requests
import json
import os
from datetime import datetime

HISTORY_FILE = "sentiment_history.json"


def get_btc_price():
    """Fetch live BTC price from CoinGecko (free, no API key)."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
        r = requests.get(url, timeout=10)
        data = r.json()
        price = data["bitcoin"]["usd"]
        change = data["bitcoin"]["usd_24h_change"]
        return price, round(change, 2)
    except Exception as e:
        print("  ⚠️  Could not fetch BTC price: " + str(e))
        return None, None


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def record_snapshot(summary, price, change):
    """Save a snapshot of sentiment + price to history."""
    history = load_history()
    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bullish_pct": summary["percentages"]["BULLISH"],
        "bearish_pct": summary["percentages"]["BEARISH"],
        "neutral_pct": summary["percentages"]["NEUTRAL"],
        "avg_score": summary["avg_score"],
        "overall": summary["overall"],
        "btc_price": price,
        "btc_change_24h": change,
    })
    # Keep last 100 snapshots
    history = history[-100:]
    save_history(history)
    return history


def print_price_panel(price, change):
    if price is None:
        return
    arrow = "▲" if change >= 0 else "▼"
    color = "\033[92m" if change >= 0 else "\033[91m"
    reset = "\033[0m"
    bold = "\033[1m"
    print()
    print(bold + "=" * 56 + reset)
    print(bold + "  💰 LIVE BTC PRICE" + reset)
    print("  $" + "{:,.2f}".format(price) + "  " + color + arrow + " " + str(change) + "% (24h)" + reset)
    print(bold + "=" * 56 + reset)


def plot_chart(history):
    """Plot sentiment vs price in the terminal using matplotlib."""
    if len(history) < 2:
        print("\n  📈 Not enough data yet for chart (need 2+ snapshots). Run again later!")
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime as dt
    except ImportError:
        print("  ⚠️  matplotlib not installed. Run: pip install matplotlib")
        return

    timestamps = [dt.strptime(h["timestamp"], "%Y-%m-%d %H:%M:%S") for h in history]
    bullish = [h["bullish_pct"] for h in history]
    bearish = [h["bearish_pct"] for h in history]
    prices  = [h["btc_price"] for h in history if h["btc_price"] is not None]
    price_ts = [dt.strptime(h["timestamp"], "%Y-%m-%d %H:%M:%S") for h in history if h["btc_price"] is not None]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor("#0d1117")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["top"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["right"].set_color("#30363d")

    # Sentiment chart
    ax1.plot(timestamps, bullish, color="#4caf50", linewidth=2, label="Bullish %", marker="o", markersize=4)
    ax1.plot(timestamps, bearish, color="#f44336", linewidth=2, label="Bearish %", marker="o", markersize=4)
    ax1.fill_between(timestamps, bullish, alpha=0.15, color="#4caf50")
    ax1.fill_between(timestamps, bearish, alpha=0.15, color="#f44336")
    ax1.set_ylabel("Sentiment %", color="white")
    ax1.set_ylim(0, 100)
    ax1.legend(facecolor="#1f2937", labelcolor="white")
    ax1.set_title("BTC Sentiment vs Price", color="white", fontsize=14, pad=12)

    # Price chart
    if prices:
        ax2.plot(price_ts, prices, color="#f7931a", linewidth=2, label="BTC Price (USD)", marker="o", markersize=4)
        ax2.fill_between(price_ts, prices, alpha=0.15, color="#f7931a")
        ax2.set_ylabel("BTC Price (USD)", color="white")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "$" + "{:,.0f}".format(x)))
        ax2.legend(facecolor="#1f2937", labelcolor="white")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(color="white", rotation=30)
    plt.tight_layout()

    chart_file = "btc_sentiment_chart.png"
    plt.savefig(chart_file, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("\n  📈 Chart saved: " + chart_file)
