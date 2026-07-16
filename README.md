# How We Automated Content Marketing to Acquire Users At Scale

*Ranking artists and ad creatives for automated user-acquisition campaigns with XGBoost learning-to-rank, orchestrated on Kubeflow Pipelines.*

---

## The problem

A growth team running artist-driven content marketing (think: a music app promoting "Listen to [Artist] now — try free") faces a combinatorial targeting problem long before any ad ever runs. Which artists, out of tens of thousands, are worth featuring this week? And for each one, which of a dozen possible creative treatments — video vs. static, TikTok vs. YouTube, "try free" vs. "listen now" — is most likely to land?

Doing this by hand doesn't scale. This project builds the ranking layer that would sit underneath that decision: two learning-to-rank models — one that ranks **artists** by acquisition potential within their genre, and one that ranks **ad creatives** for a given artist — feeding into a single prioritized campaign list. It's built as a real Kubeflow Pipeline, not a notebook, because that's what makes it re-runnable on a schedule against fresh data.

## Data — and an honest disclosure about what's real and what isn't

**Real data:** track-level audio features, popularity, genre, and playlist-membership for ~32,800 tracks and ~10,700 artists, from Spotify's own [Web API](https://developer.spotify.com/documentation/web-api), released as open data via the [TidyTuesday](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21) project. It's downloaded fresh at pipeline run time — no API key required.

**What's simulated, and why:** Spotify does not publish (and no open dataset contains) real ad-campaign outcomes — no impressions, clicks, installs, or ad-creative metadata exist for these artists. Two things in this project are therefore synthetic, and it's disclosed inline in the code everywhere it happens:

1. **The artist "acquisition value" label.** There's no ground-truth "this artist, if promoted, acquired N users." Instead, each artist's label is a documented formula: a weighted combination of *real* signals a UA team would actually have on hand — current track popularity, playlist reach, release recency, and sonic "ad appeal" (danceability/energy) — plus substantial Gaussian noise standing in for the genuine unpredictability of real campaign performance. The noise is deliberately large relative to the signal specifically so the model has to learn a real pattern rather than just memorizing the formula (see [Model results](#model-results) — this is why test NDCG is ~0.74, not 1.0).
2. **Ad creatives entirely.** Format, hook, CTA, visual style, and placement are sampled from a defined vocabulary; engagement is generated from a small hand-authored affinity table (e.g., "EDM + video + TikTok" gets a bonus) plus noise. This is a stand-in for a real ad-platform performance feed (Meta/TikTok/Google Ads reporting), which is exactly what you'd swap in for a production version of this pipeline.

Net effect: the *inputs* (audio features, popularity, reach, genre) are real Spotify data throughout. The *outcome labels* are simulated but designed to be non-trivial to predict, so the ranking models are solving a genuine (if synthetic) learning-to-rank problem rather than recovering an algebraic identity.

## Approach

**Why learning-to-rank, not regression/classification.** The business question isn't "what's this artist's absolute score" — it's "given this pool of candidates, which order should I work through them in." XGBoost's `rank:ndcg` objective optimizes directly for ranking quality within groups (`qid`), which is the right loss function for that question.

- **Artist ranker** — one row per artist, grouped (`qid`) by genre, relevance-graded into deciles per genre from the acquisition-lift label. Features: track/album counts, playlist reach, popularity stats, release recency, and mean audio features (danceability, energy, valence, tempo, etc.).
- **Creative ranker** — one row per (artist, creative variant), grouped by artist, relevance-graded from the engagement label. Features: the artist's popularity/reach/audio profile plus the creative's categorical attributes (format, hook, CTA, visual style, placement), one-hot encoded.
- Both are trained by the *same* reusable Kubeflow component (`train_ranker`), parametrized by feature columns, group column, and label column — one component, two models, which is the shape you actually want in a production pipeline.

**Why Kubeflow Pipelines.** This needs to re-run regularly against fresh Spotify/ad-platform data, retrain both models, and hand off a fresh recommendation list — a scheduled, reproducible DAG with typed artifacts and per-stage metrics is exactly Kubeflow's job. The pipeline here is written with the KFP v2 SDK as fully self-contained lightweight components (each declares its own `packages_to_install`, no shared-filesystem imports), so it compiles to a portable `pipeline_compiled.yaml` and runs unchanged on a real cluster — the version here just executes locally via KFP's subprocess runner, which is enough to prove the DAG and every component genuinely works without standing up a cluster.

## Pipeline architecture

```
download_spotify_data
        │
        ▼
build_artist_features ──────────────┐
        │                            │
        ▼                            ▼
generate_ad_creatives          train_ranker (artists, qid=genre)
        │                            │
        ▼                            │
train_ranker (creatives, qid=artist) │
        │                            │
        └──────────┬─────────────────┘
                    ▼
        generate_campaign_recommendations
```

| Stage | What it does |
|---|---|
| `download_spotify_data` | Pulls the open Spotify tracks CSV |
| `build_artist_features` | Aggregates ~32.8k tracks → 4,312 artists (≥2 tracks each), builds the acquisition-lift label |
| `generate_ad_creatives` | Synthesizes 8 creative variants for the top 80 artists/genre (480 artists → 3,840 creatives) |
| `train_ranker` ×2 | XGBoost `rank:ndcg`, group-aware train/test split, logs NDCG@5 + feature importances |
| `generate_campaign_recommendations` | Scores every artist and creative, blends both rankers into one prioritized list |

## Model results

| Model | Groups (`qid`) | Train NDCG@5 | Test NDCG@5 | Top features |
|---|---|---|---|---|
| Artist ranker | genre (6) | 0.979 | **0.738** | playlist reach, track count, max popularity, album count |
| Creative ranker | artist (480) | 0.778 | **0.717** | CTA = "try free", format = story ad / video, genre |

Both are trained/tested with a group-aware split (`GroupShuffleSplit`, 80/20) so no artist's or genre's data leaks across the boundary. The train/test gap is intentionally not huge and NDCG isn't pinned at 1.0 — see the [data disclosure](#data--and-an-honest-disclosure-about-what-is-real-and-what-isnt) above for why that matters and how it was tuned (label noise + relevance-grade granularity).

## Sample output — the actual deliverable

`outputs/reports/campaign_recommendations.csv` is the business-facing artifact: top artists blended with their best-fit creative, ranked by a combined priority score (`0.6 × artist score + 0.4 × creative score`, both min-max normalized).

| Rank | Artist | Genre | Creative | Combined score |
|---|---|---|---|---|
| 1 | Dua Lipa | pop | video · trending_challenge · TikTok | 0.874 |
| 2 | Selena Gomez | pop | video · genre_tag · TikTok | 0.868 |
| 3 | Ed Sheeran | latin | video · artist_story · YouTube preroll | 0.821 |
| 4 | Travis Scott | rap | video · artist_story · YouTube preroll | 0.816 |
| 5 | The Chainsmokers | edm | video · genre_tag · TikTok | 0.761 |

Across the full top-25, `video` dominates the recommended format (20/25) and TikTok/Instagram Story dominate placement (17/25) — which tracks with the affinity model favoring short-form video for high-energy genres, and is a sanity-checkable signal that the pipeline is doing something coherent rather than random.

## Project layout

```
content-marketing-automation/
├── pipeline/
│   ├── pipeline.py              # KFP v2 pipeline + all components (source of truth)
│   └── pipeline_compiled.yaml   # compiled pipeline, deployable to a real KFP cluster
├── scripts/
│   └── run_local.py             # runs the pipeline locally via kfp's subprocess runner
├── data/raw/                    # downloaded Spotify CSV (fetched by the pipeline)
├── outputs/
│   ├── models/                  # trained XGBoost rankers (JSON boosters + column maps)
│   └── reports/                 # artist_features.csv, ad_creatives.csv,
│                                 # campaign_recommendations.csv, pipeline_metrics.json
├── requirements.txt
└── README.md
```

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Compile the pipeline to a portable KFP YAML (what you'd upload to a real cluster)
python3 pipeline/pipeline.py

# Run it end-to-end locally (no cluster needed) via KFP's subprocess runner
python3 scripts/run_local.py --top-n-per-genre 80 --creatives-per-artist 8 --top-k-artists 25
```

Outputs land under `local_outputs/<run-id>/<stage>/`, including per-stage metrics (`executor_output.json`) and every intermediate dataset/model artifact — the same layout KFP uses on a real cluster, just on your filesystem instead of object storage.

### Deploying to a real Kubeflow cluster

Because every component is self-contained, `pipeline_compiled.yaml` uploads directly to any KFP-conformant cluster (Kubeflow Pipelines UI → **Upload pipeline**, or `kfp.Client().create_run_from_pipeline_package(...)`) with no code changes. The natural next steps for production: swap the synthetic ad-creative stage for a real ad-platform reporting feed, add a recurring `Schedule` (e.g. daily/weekly retrain), and push `generate_campaign_recommendations`'s output to wherever the UA team's campaign tooling reads from.

## Honest limitations

- **The acquisition-lift and engagement labels are simulated**, as detailed above — this is a methodology demonstration, not a validated production model. Swapping in real UA outcome data and real ad-platform engagement data is the actual path to production.
- **Only 6 genre groups** for the artist ranker means that model's group-level generalization is measured on a very small number of held-out queries; more genre granularity (subgenre-level, or cross-genre features) would give a more robust evaluation.
- **The affinity table for ad creatives is hand-authored**, not learned from real ad performance — it encodes plausible marketing intuition (short-form video does well on TikTok for high-energy genres), not verified fact.
