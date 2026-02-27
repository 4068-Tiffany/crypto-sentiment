from config import COIN, COIN_SYMBOL
from datetime import datetime

ICONS = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}
COLORS = {
    "BULLISH": "\033[92m",
    "BEARISH": "\033[91m",
    "NEUTRAL": "\033[93m",
    "RESET":   "\033[0m",
    "BOLD":    "\033[1m",
    "CYAN":    "\033[96m",
    "DIM":     "\033[2m",
}


def c(color, text):
    return COLORS[color] + text + COLORS["RESET"]


def print_header():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print(c("BOLD", "=" * 56))
    print(c("BOLD", c("CYAN", "   B  CRYPTO SENTIMENT ANALYZER — " + COIN + " (" + COIN_SYMBOL + ")")))
    print(c("DIM", "   " + now))
    print(c("BOLD", "=" * 56))


def print_summary(summary):
    pct = summary["percentages"]
    cnt = summary["counts"]
    overall = summary["overall"]
    icon = ICONS[overall]

    bullish_str = str(cnt["BULLISH"]).rjust(3) + " posts  (" + str(pct["BULLISH"]) + "%)"
    bearish_str = str(cnt["BEARISH"]).rjust(3) + " posts  (" + str(pct["BEARISH"]) + "%)"
    neutral_str = str(cnt["NEUTRAL"]).rjust(3) + " posts  (" + str(pct["NEUTRAL"]) + "%)"

    print()
    print(c("BOLD", "  RESULTS SUMMARY"))
    print("  " + "-" * 40)
    print("  " + ICONS["BULLISH"] + " Bullish : " + c("BULLISH", bullish_str))
    print("  " + ICONS["BEARISH"] + " Bearish : " + c("BEARISH", bearish_str))
    print("  " + ICONS["NEUTRAL"] + " Neutral : " + c("NEUTRAL", neutral_str))
    print("  " + "-" * 40)
    print("  Total posts  : " + str(summary["total"]))
    print("  Avg score    : " + str(summary["avg_score"]))
    print()
    print(c("BOLD", "  Overall Sentiment: " + icon + " " + c(overall, overall)))
    print(c("BOLD", "=" * 56))


def print_bar_chart(summary):
    pct = summary["percentages"]
    print()
    print(c("BOLD", "  SENTIMENT BAR CHART"))
    print()
    for label in ["BULLISH", "BEARISH", "NEUTRAL"]:
        bar_len = int(pct[label] / 2)
        bar = "█" * bar_len
        print("  " + ICONS[label] + " " + label.ljust(8) + " " + c(label, bar) + " " + str(pct[label]) + "%")
    print()


def print_posts(results, limit=10):
    print(c("BOLD", "  SAMPLE POSTS (top " + str(limit) + ")"))
    print("  " + "-" * 52)
    for r in results[:limit]:
        icon = ICONS[r["label"]]
        title = r["title"][:60] + "..." if len(r["title"]) > 60 else r["title"]
        score_str = "{:+.3f}".format(r["compound"])
        print("  " + icon + " [" + score_str + "] " + title)
        print(c("DIM", "       " + r["source"]))
        print()
    print(c("BOLD", "=" * 56))


def display(results, summary):
    print_header()
    print_summary(summary)
    print_bar_chart(summary)
    print_posts(results)
