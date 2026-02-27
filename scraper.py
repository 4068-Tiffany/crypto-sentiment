import requests
import time
from config import SUBREDDITS, POSTS_PER_SUB, POST_SORT, COIN, COIN_SYMBOL

HEADERS = {"User-Agent": "crypto-sentiment-analyzer/1.0"}


def fetch_subreddit(subreddit: str) -> list[dict]:
    """Fetch posts from a subreddit using the public JSON endpoint."""
    url = f"https://www.reddit.com/r/{subreddit}/{POST_SORT}.json?limit={POSTS_PER_SUB}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        posts = []
        for child in data["data"]["children"]:
            p = child["data"]
            text = f"{p.get('title', '')} {p.get('selftext', '')}".strip()
            posts.append({
                "source": f"r/{subreddit}",
                "title": p.get("title", ""),
                "text": text,
                "score": p.get("score", 0),
                "url": f"https://reddit.com{p.get('permalink', '')}",
            })
        return posts
    except Exception as e:
        print(f"  ⚠️  Could not fetch r/{subreddit}: {e}")
        return []


def fetch_all_posts() -> list[dict]:
    """Fetch posts from all configured subreddits."""
    all_posts = []
    print(f"\n📡 Fetching posts about {COIN} ({COIN_SYMBOL})...\n")
    for sub in SUBREDDITS:
        print(f"  🔍 Scraping r/{sub}...")
        posts = fetch_subreddit(sub)
        print(f"     ✅ Got {len(posts)} posts")
        all_posts.extend(posts)
        time.sleep(1)   # be polite to Reddit's servers
    print(f"\n  📦 Total posts collected: {len(all_posts)}\n")
    return all_posts
