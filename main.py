import subprocess
import sys
import time


def install_deps():
    deps = ["requests", "vaderSentiment", "openpyxl", "matplotlib"]
    for dep in deps:
        pkg = dep.replace("-", "_").split(".")[0]
        try:
            __import__(pkg)
        except ImportError:
            print("📦 Installing " + dep + "...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])


install_deps()

from scraper import fetch_all_posts
from sentiment import analyze_posts, summarize
from display import display
from csv_exporter import export_all
from chart import get_btc_price, record_snapshot, plot_chart, print_price_panel

# ── CONFIG ───────────────────────────────────────────────────
REFRESH_MINUTES = 5       # how often to re-run (set to 0 to run once)
SHOW_CHART      = True    # show price vs sentiment chart
EXPORT          = True    # save CSV + Excel after each run
# ─────────────────────────────────────────────────────────────


def run_once():
    posts = fetch_all_posts()
    if not posts:
        print("No posts fetched. Check your internet connection.")
        return None

    print("🧠 Running sentiment analysis...")
    results = analyze_posts(posts)
    summary = summarize(results)

    price, change = get_btc_price()
    print_price_panel(price, change)

    display(results, summary)

    history = record_snapshot(summary, price, change)

    if EXPORT:
        export_all(results, summary)

    if SHOW_CHART:
        plot_chart(history)

    return summary


def main():
    if REFRESH_MINUTES <= 0:
        run_once()
    else:
        run_count = 1
        while True:
            print("\n🔄 Run #" + str(run_count))
            run_once()
            run_count += 1
            wait = REFRESH_MINUTES * 60
            print("\n⏰ Next refresh in " + str(REFRESH_MINUTES) + " minutes... (Ctrl+C to stop)\n")
            try:
                time.sleep(wait)
            except KeyboardInterrupt:
                print("\n👋 Stopped. Goodbye!")
                break


if __name__ == "__main__":
    main()
