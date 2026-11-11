# Movie Review Analyzer — Sentiment + Sarcasm Pipeline

Give it a movie (or TV-series) name. It crawls real reviews from the web and judges whether the title is **GOOD / BAD / MIXED** by combining a **sentiment** model with a **movie-domain sarcasm** model, so a review that *sounds* positive but is sarcastic is correctly counted as negative.

**Problem:** off-the-shelf sentiment models are biased toward "positive" and miss sarcastic negative reviews.
**Solution:** a custom sarcasm model fine-tuned on movie-subreddit comments, combined with a pre-trained sentiment model and a multi-source review crawler.
**Key result:** sarcasm F1 on held-out movie comments rose from **0.38** (best ready-made model) to **0.72** (our model).

<!-- Add a screenshot or GIF of the Streamlit app here, e.g.:
![App screenshot](docs/screenshot.png) -->

> **Guiding principle:** prefer ready, pre-trained models and only fine-tune when they genuinely fall short, so the numbers reflect what a real app would get, not an over-fitted lab score.

---

## Contributors

This was a group project.

- **<Your name>** — <your part, e.g. sarcasm model, crawler, app>
- **<Teammate 1>** — <their part>
- **<Teammate 2>** — <their part>

---

## Setup

Use **Python 3.11** (newer system Pythons may have no PyTorch wheels):

```bash
git clone https://github.com/Bhr27/IMDB-Sentiment-Classification.git
cd IMDB-Sentiment-Classification
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The crawler's two API sources need free keys, passed as environment variables:

```bash
export TMDB_API_KEY=<your TMDB v3 API key>        # themoviedb.org -> Settings -> API
export TRAKT_CLIENT_ID=<your Trakt app Client ID> # trakt.tv/oauth/applications -> New Application
```

(Windows PowerShell: `$env:TMDB_API_KEY="..."`.) Letterboxd needs no key. Without the keys the crawler still runs on Letterboxd alone; the API sources skip gracefully.

### Data and model files (not stored in this repo)

To keep the repo small, these are **not** committed:

| Item | How to get it |
|------|---------------|
| Reddit movie-sarcasm data (`phase3_sarcasm/data/reddit_movie_*.csv`) | Rebuilt from the SARC corpus, filtered to movie subreddits. See `phase3_sarcasm/DATASET_INSTRUCTIONS.md`. |
| Trained sarcasm model (`phase3_sarcasm/checkpoints/movie_reddit_model`) | Train it yourself (Phase 3 below) or download it from Hugging Face: `<add link>`. |
| Crawler output (`phase4_crawler/gathered_data/`) | Generated when you run the crawler. |

Phases 1, 2 and 4 run without the sarcasm checkpoint. Phases 3 (combined inference) and 5 (app) need it.

---

## Phase 1 — Sanity-check the sentiment model

Model: `fabriceyhc/bert-base-uncased-imdb` (BERT fine-tuned on IMDB).

```bash
python phase1_sanity_check/sanity_check.py
pytest phase1_sanity_check/test_sanity.py -v      # same 8 checks as unit tests
```

8 checks: output shape/softmax, prediction direction, edge cases, confidence, reproducibility, calibration, batch consistency, tokenization.

**Result:** 7/8 pass. The one failure, *probability calibration*, is a real property of the model (it is over-confident, mean confidence 0.993), not a bug; `pytest` reproduces the identical outcome. See `results/phase1/`.

---

## Phase 2 — How well does the ready model do on a *different* dataset?

Dataset: **Rotten Tomatoes** (`cornell-movie-review-data/rotten_tomatoes`).

We **deliberately did not fine-tune.** The goal is to estimate real-world accuracy on freshly crawled reviews, so we evaluate the existing IMDB model on a different-source dataset.

```bash
python phase2_new_dataset/download_and_prepare.py                     # download + stats + splits
python phase2_new_dataset/evaluate.py --model fabriceyhc/bert-base-uncased-imdb
```

> `phase2_new_dataset/finetune.py` is an optional script kept for experimentation. It was **not** used for any reported result.

**Result:** ~**80% accuracy / F1 0.82** (full 10,662-review set: acc 0.8011, F1 0.8185). The model transfers well but is **biased toward positive** (predicts positive ~59% of the time, recall ~0.90, precision ~0.75), so it misses subtle or sarcastic negative reviews. This is what Phase 3 addresses. See `results/phase2/`.

---

## Phase 3 — A sarcasm model that works on movie reviews

Ready sarcasm models exist, but most are trained on **news headlines** and collapse on conversational movie sarcasm. We compared them and ended up fine-tuning our own model on real movie-discussion data.

| Sarcasm model | Domain | F1 on held-out movie test |
|---------------|--------|---------------------------|
| `helinivan/english-sarcasm-detector` | news headlines | 0.11 (collapses) |
| `dnzblgn/Sarcasm-Detection-Customer-Reviews` | customer reviews | 0.38 (best ready-made) |
| **Our BERT, fine-tuned on r/movies Reddit** | **movie comments** | **0.72** |
| RoBERTa, same data | movie comments | 0.71 (no gain, task ceiling) |

Training data: **9,527 movie-subreddit comments** (r/movies, marvelstudios, TrueFilm, ...) from the SARC corpus (`marcbishara/sarcasm-on-reddit`, the Kaggle `danofer/sarcasm` dataset), filtered by subreddit.

```bash
# reproduce the winning model (BERT) and the comparison; RoBERTa via env vars
python results/phase3/train_movie_sarcasm.py
BASE_MODEL=roberta-base OUT_DIR=phase3_sarcasm/checkpoints/movie_reddit_roberta \
    python results/phase3/train_movie_sarcasm.py
```

**Final sarcasm model:** `phase3_sarcasm/checkpoints/movie_reddit_model` (BERT).

Combined sentiment + sarcasm inference (used by the app):

```bash
python phase3_sarcasm/combined_inference.py --test
python phase3_sarcasm/combined_inference.py "Oh sure, BEST movie ever 🙄"
```

Verdict logic: a *Positive + Sarcastic* review becomes "Likely Negative".
See `results/phase3/` (final) and `results/phase3/journey_storytelling/` (how we got there).

---

## Phase 4 — Multi-source review crawler

Most movie sites (IMDb, Rotten Tomatoes, Metacritic, Reddit) are JS-rendered or bot-walled, so we crawl **three reliable sources**:

| Source | Type | Auth |
|--------|------|------|
| **TMDB** | official API (JSON) | free `TMDB_API_KEY` |
| **Letterboxd** | server-rendered HTML (slug-based) | none |
| **Trakt** | official API (JSON) | free `TRAKT_CLIENT_ID` |

```bash
python phase4_crawler/crawler.py "Inception"
python phase4_crawler/test_crawler.py            # 2-movie sanity test
```

Output: `phase4_crawler/gathered_data/<movie>_reviews.xlsx` (columns: `source, review_text, date, rating`).

Robustness: every request **retries up to 3x with growing backoff** and treats 403/429 as rate limits; each source reports a **reason** on failure; TMDB and Trakt fall back from a movie match to a **TV-series** match (so *Game of Thrones* works).
Example, Inception: 146 reviews (TMDB 8 + Letterboxd 66 + Trakt 80). See `results/phase4/`.

---

## Phase 5 — Web app (Streamlit)

```bash
streamlit run phase5_app/app.py                  # needs TMDB_API_KEY + TRAKT_CLIENT_ID and the sarcasm checkpoint
python phase5_app/test_pipeline_e2e.py           # headless end-to-end check (4/4 pass)
```

Enter a movie and the dashboard shows:

1. **Raw sentiment verdict** + confidence (first model only, *before* sarcasm),
2. **Verdict after sarcasm correction** + confidence, side by side, with a banner if sarcasm changed the verdict,
3. top **positive / negative / most-sarcastic** reviews (with their source),
4. per-source counts and a **crawler report** (with the reason per source),
5. a **History tab** with the last 3 analyses of the session.

Example, Inception: **GOOD, 66% confidence**, 20 of 146 reviews flagged sarcastic. See `results/phase5/`.

---

## Limitations

- **Sarcasm domain:** the sarcasm model is trained on Reddit movie comments. It may transfer less well to other review styles (e.g. long-form critic reviews).
- **Text-only sarcasm is hard:** without tone or conversation context there is a ceiling; RoBERTa gave no gain over BERT on the same data.
- **Sentiment model:** over-confident (mean confidence 0.993) and biased toward "positive" on out-of-domain text (~80% accuracy on Rotten Tomatoes).
- **Crawler coverage:** review counts vary by source and title (e.g. TMDB returned only 8 reviews for Inception), and sources can rate-limit.
- **Small samples:** verdicts on titles with few reviews are less reliable.

---

## Order of execution

Phases run in order 1 → 2 → 3 → 4 → 5. Phase 3's combined inference uses the IMDB sentiment model + our movie sarcasm model; Phase 5 imports Phases 3 and 4.

## Where the results live

```
results/
├── phase1/   sanity-check logs + SUMMARY
├── phase2/   evaluation summaries, data source
├── phase3/   final sarcasm comparison + README ; journey_storytelling/ = the process
├── phase4/   crawler summary + source-probe findings
└── phase5/   app summary + e2e test log
```

## Key models & datasets

- Sentiment: `fabriceyhc/bert-base-uncased-imdb` (used as-is)
- Sarcasm: our BERT fine-tuned on r/movies (`phase3_sarcasm/checkpoints/movie_reddit_model`)
- Phase 2 data: `cornell-movie-review-data/rotten_tomatoes`
- Phase 3 data: `marcbishara/sarcasm-on-reddit` (SARC), filtered to movie subreddits

## License

<Add a LICENSE file, e.g. MIT, and name it here.>
