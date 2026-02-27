import subprocess, sys

def install_deps():
    for dep in ["transformers", "torch", "vaderSentiment"]:
        try:
            __import__(dep.replace("-","_"))
        except ImportError:
            print(f"📦 Installing {dep} (this may take a while)...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])

install_deps()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import threading

# ── CONFIG ────────────────────────────────────────────────
MODEL_NAME    = "kk08/CryptoBERT"
BATCH_SIZE    = 8      # process 8 posts at a time (CPU friendly)
MAX_LENGTH    = 128    # truncate long posts
USE_BERT      = True   # set False to force VADER only
# ──────────────────────────────────────────────────────────

vader     = SentimentIntensityAnalyzer()
_pipeline = None
_loading  = False
_lock     = threading.Lock()

BULLISH_KW = ["moon","bullish","buy","pump","ath","breakout","hodl","accumulate","rally","surge","rocket","long","up","green","gains","profit","launch","adoption","institutional"]
BEARISH_KW = ["crash","bearish","sell","dump","rug","scam","dead","panic","drop","fall","short","fear","liquidation","fraud","hack","ban","regulation","lawsuit","red","loss"]


def load_model():
    """Load CryptoBERT in background so app starts instantly."""
    global _pipeline, _loading
    if _pipeline is not None or _loading:
        return
    _loading = True
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        print("🤖 Loading CryptoBERT model (first time ~500MB download)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=MAX_LENGTH,
            device=-1,   # -1 = CPU
        )
        print("✅ CryptoBERT loaded! Higher accuracy mode active.")
    except Exception as e:
        print(f"⚠️  CryptoBERT failed to load ({e}), falling back to VADER.")
        _pipeline = None
    _loading = False


def vader_score(text):
    """VADER compound score with crypto keyword boost."""
    base  = vader.polarity_scores(text)["compound"]
    tl    = text.lower()
    boost = sum(0.05 for w in BULLISH_KW if w in tl) - sum(0.05 for w in BEARISH_KW if w in tl)
    return max(-1.0, min(1.0, base + boost))


def bert_label_to_score(label, score):
    """
    CryptoBERT returns labels like 'Bullish' / 'Bearish'.
    Convert to a -1 to +1 compound score.
    """
    label = label.lower()
    if "bull" in label or "positive" in label:
        return score          # e.g. +0.92
    elif "bear" in label or "negative" in label:
        return -score         # e.g. -0.92
    else:
        return 0.0


def classify(score):
    if score >= 0.05:  return "BULLISH"
    if score <= -0.05: return "BEARISH"
    return "NEUTRAL"


def analyze_posts(posts):
    """
    Analyze a list of posts. Uses CryptoBERT if loaded,
    otherwise falls back to VADER. Always returns instantly.
    """
    results = []

    if USE_BERT and _pipeline is not None:
        # ── BERT MODE ─────────────────────────────────────
        print(f"🤖 Analyzing {len(posts)} posts with CryptoBERT...")
        texts = [p["text"][:512] if p["text"] else p["title"] for p in posts]

        # Process in batches
        all_outputs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            try:
                out = _pipeline(batch, truncation=True, max_length=MAX_LENGTH)
                all_outputs.extend(out)
            except Exception as e:
                # fallback to VADER for this batch
                for t in batch:
                    s = vader_score(t)
                    all_outputs.append({"label": classify(s), "score": abs(s)})

        for post, out in zip(posts, all_outputs):
            compound = bert_label_to_score(out["label"], out["score"])
            # Blend with VADER for robustness: 70% BERT + 30% VADER
            v_score  = vader_score(post["text"])
            blended  = round(0.7 * compound + 0.3 * v_score, 4)
            results.append({
                **post,
                "compound": blended,
                "label":    classify(blended),
                "method":   "CryptoBERT",
            })
        print(f"✅ CryptoBERT analysis complete.")

    else:
        # ── VADER MODE (fallback) ──────────────────────────
        print(f"📊 Analyzing {len(posts)} posts with VADER...")
        for post in posts:
            s = vader_score(post["text"])
            results.append({
                **post,
                "compound": round(s, 4),
                "label":    classify(s),
                "method":   "VADER",
            })

    return results


def summarize(results):
    total  = len(results)
    counts = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
    for r in results: counts[r["label"]] += 1
    pct = {k: round(v/total*100, 1) if total else 0 for k,v in counts.items()}
    avg = round(sum(r["compound"] for r in results)/total, 4) if total else 0
    method = results[0]["method"] if results else "VADER"
    return {
        "total": total, "counts": counts, "percentages": pct,
        "avg_score": avg, "overall": classify(avg), "method": method,
    }


# Start loading CryptoBERT in background immediately
if USE_BERT:
    threading.Thread(target=load_model, daemon=True).start()
