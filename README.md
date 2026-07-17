# How We Automated Content Marketing to Acquire Users At Scale

Ranking artists and ad creatives for automated user acquisition campaigns, using XGBoost learning to rank, orchestrated with Kubeflow Pipelines.

## Why I built this

Say you're on a growth team and you're running artist driven content marketing. Something like "Listen to [Artist] now, try free." Before any ad actually runs, you have a targeting problem. Out of tens of thousands of artists, which ones are worth featuring this week? And for each one, which creative actually works? Video or static image? TikTok or YouTube? "Try free" or "Listen now"? There are a dozen combinations per artist and you can't manually eyeball your way through that at scale.

So I wanted to build the ranking layer that would sit under that decision. Two learning to rank models. One ranks artists by acquisition potential within their genre. The other ranks ad creatives for a given artist. Both feed into one prioritized campaign list at the end. I built it as an actual Kubeflow Pipeline instead of a notebook because the whole point is that this needs to rerun on a schedule against fresh data, not get run once by hand.

## The data situation, and being upfront about it

I used real Spotify data. Track level audio features, popularity, genre, playlist membership, for about 32,800 tracks and 10,700 artists. This comes from Spotify's own Web API, released as open data through the [TidyTuesday](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21) project. The pipeline downloads it fresh every run, no API key needed.

Here's the thing though. Spotify doesn't publish ad campaign outcomes anywhere, and neither does anyone else as far as I could find. There's no open dataset of impressions, clicks, installs, or ad creative performance tied to these artists. So two parts of this project are simulated, and I want to be straight about it rather than bury it:

1. **The artist "acquisition value" label.** There's no ground truth for "if we promoted this artist, we acquired N users." What I did instead was build a label as a documented formula: a weighted mix of real signals a UA team would actually have on hand (current track popularity, playlist reach, release recency, and some sonic "ad appeal" from danceability and energy), plus a decent amount of Gaussian noise on top to stand in for how unpredictable real campaign performance actually is. I made the noise deliberately large relative to the signal on purpose, because my first pass at this had noise too small and the model was basically just memorizing the label formula back out (test NDCG came out at literally 1.0, which was my tell that something was wrong, more on that below).
2. **The ad creatives, entirely.** Format, hook, CTA, visual style, and placement are all sampled from a vocabulary I defined, and the engagement label comes from a small affinity table I hand wrote (like, EDM plus video plus TikTok gets a bonus) plus noise. In a real version of this you'd replace this whole stage with an actual ad platform reporting feed from Meta or TikTok or Google Ads.

So to be clear: every input feature (audio features, popularity, reach, genre) is real Spotify data. The labels I'm trying to predict are simulated, but built so they're genuinely hard to predict, not just an algebra problem in disguise.

## How I approached the modeling

First decision was learning to rank instead of regression or classification. The actual business question isn't "what's this artist's score," it's "given this whole pool of candidates, what order should I work through them in." XGBoost has a `rank:ndcg` objective built for exactly this, optimizing ranking quality within groups (`qid`), so that's what I used.

Artist ranker: one row per artist, grouped by genre as the qid, relevance graded into deciles per genre based on the acquisition lift label. Features are track and album counts, playlist reach, popularity stats, release recency, and mean audio features like danceability, energy, valence, tempo.

Creative ranker: one row per artist/creative combo, grouped by artist as the qid, relevance graded from the engagement label. Features are the artist's popularity/reach/audio profile plus the creative's own categorical attributes, one hot encoded.

Both models actually get trained by the same reusable Kubeflow component, just parametrized differently (which feature columns, which group column, which label column). One component, two models. That's the shape you actually want if this were a real production pipeline instead of two copy pasted scripts.

As for why Kubeflow specifically: this is the kind of thing that needs to rerun regularly against fresh data, retrain both models, and hand off an updated recommendation list. That's a scheduled, reproducible DAG with typed artifacts and per stage metrics, which is exactly what Kubeflow is for. I wrote every component in the KFP v2 SDK as fully self contained lightweight components (each one declares its own `packages_to_install`, no shared filesystem imports between them), so it compiles down to a portable `pipeline_compiled.yaml` that would run unchanged on a real cluster. For this project I ran it locally through KFP's subprocess runner instead of standing up an actual cluster, which is enough to prove every component and the whole DAG genuinely executes.

## What the pipeline actually looks like

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

Stage by stage:

`download_spotify_data` just pulls the open Spotify tracks CSV.

`build_artist_features` aggregates about 32.8k tracks down to 4,312 artists (I filtered out anyone with fewer than 2 tracks in the dataset since aggregated stats on a single song aren't very meaningful), and builds the acquisition lift label.

`generate_ad_creatives` synthesizes 8 creative variants each for the top 80 artists per genre, so 480 artists times 8 comes out to 3,840 synthetic creatives.

`train_ranker` runs twice, once for artists and once for creatives. XGBoost with `rank:ndcg`, a group aware train/test split so nothing leaks across the boundary, and it logs NDCG@5 plus feature importances each time.

`generate_campaign_recommendations` scores every artist and every creative with the trained models and blends both rankings into one final prioritized list.

## Results, and a bug I want to mention

| Model | Groups (qid) | Train NDCG@5 | Test NDCG@5 | Top features |
|---|---|---|---|---|
| Artist ranker | genre (6) | 0.979 | 0.738 | playlist reach, track count, max popularity, album count |
| Creative ranker | artist (480) | 0.778 | 0.717 | CTA = "try free", format = story ad / video, genre |

Both models are evaluated with a group aware 80/20 split (`GroupShuffleSplit`) so an artist's or a genre's data can't leak across train and test.

I want to flag something because I think it's actually the more interesting part of building this. My first version of the artist ranker came back with test NDCG@5 basically equal to 1.0. Perfect score. That should immediately make you suspicious, and it did, because it meant the model wasn't really learning anything, it was just recovering the label formula I'd written. The problem was that the label was a linear combination of a handful of features (popularity, reach, recency, danceability, energy), and I was feeding those exact same features into the model, with barely any noise on top and coarse 5 bin relevance grades on a genre group with hundreds of artists in it. So the model just had to roughly cluster the top 20% of artists to nail NDCG@5 every time. Not a real prediction task.

To fix it I turned the noise up substantially (0.4 to 0.8 standard deviation, which is bigger than the deterministic signal itself), switched the relevance grading from 5 bins to 10 so the top bucket is a much smaller, harder to hit target, and pulled back the model capacity a bit (fewer trees, shallower depth, some L2 regularization) so it couldn't just brute force overfit its way around the noise. After that, test NDCG dropped to a much more believable 0.74, with train NDCG no longer pinned at 1.0 either. That felt like a real result instead of a magic trick.

## The actual output

`outputs/reports/campaign_recommendations.csv` is the thing a growth team would actually look at. Top artists blended with their best fit creative, ranked by a combined priority score (0.6 times the normalized artist score plus 0.4 times the normalized creative score).

| Rank | Artist | Genre | Creative | Combined score |
|---|---|---|---|---|
| 1 | Dua Lipa | pop | video, trending_challenge, TikTok | 0.874 |
| 2 | Selena Gomez | pop | video, genre_tag, TikTok | 0.868 |
| 3 | Ed Sheeran | latin | video, artist_story, YouTube preroll | 0.821 |
| 4 | Travis Scott | rap | video, artist_story, YouTube preroll | 0.816 |
| 5 | The Chainsmokers | edm | video, genre_tag, TikTok | 0.761 |

Looking across the full top 25, video is the recommended format for 20 out of 25, and TikTok or Instagram Story shows up as the placement for 17 out of 25. That tracks with the affinity table I wrote favoring short form video for higher energy genres, which was a decent sanity check that the pipeline is doing something coherent and not just spitting out noise.

## How the repo is laid out

```
content-marketing-automation/
├── pipeline/
│   ├── pipeline.py              # the KFP v2 pipeline and every component, source of truth
│   └── pipeline_compiled.yaml   # compiled pipeline, ready to deploy to a real KFP cluster
├── scripts/
│   └── run_local.py             # runs the pipeline locally through kfp's subprocess runner
├── data/raw/                    # downloaded Spotify CSV, fetched by the pipeline itself
├── outputs/
│   ├── models/                  # trained XGBoost rankers, JSON boosters plus column maps
│   └── reports/                 # artist_features.csv, ad_creatives.csv,
│                                 # campaign_recommendations.csv, pipeline_metrics.json
├── requirements.txt
└── README.md
```

## Running it yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# compile the pipeline into a portable KFP yaml, this is what you'd upload to a real cluster
python3 pipeline/pipeline.py

# run it end to end locally, no cluster needed, through KFP's subprocess runner
python3 scripts/run_local.py --top-n-per-genre 80 --creatives-per-artist 8 --top-k-artists 25
```

Everything lands under `local_outputs/<run-id>/<stage>/`, including per stage metrics (`executor_output.json`) and every intermediate dataset and model artifact. It's the same layout KFP uses on a real cluster, just sitting on your filesystem instead of object storage.

### If you actually wanted to run this on a real cluster

Since every component is self contained, `pipeline_compiled.yaml` should upload directly to any KFP conformant cluster with no code changes. Kubeflow Pipelines UI, Upload pipeline, or `kfp.Client().create_run_from_pipeline_package(...)` if you're doing it programmatically. From there the real next steps would be swapping the synthetic ad creative stage for an actual ad platform reporting feed, adding a recurring schedule for retraining, and pointing `generate_campaign_recommendations`'s output at wherever the UA team's campaign tooling actually reads from.

## What this isn't

I want to be honest about the limits here instead of overselling it.

The acquisition lift and engagement labels are simulated, as I explained above. This is a demonstration of the methodology, not a validated production model. Getting this to production means swapping in real UA outcome data and a real ad platform engagement feed.

There are only 6 genre groups for the artist ranker, which means that model's generalization is being measured on a pretty small number of held out queries. Going to subgenre level grouping, or adding cross genre features, would give a more robust evaluation.

The affinity table behind the ad creatives is something I hand wrote based on general marketing intuition (short form video does well on TikTok for high energy genres, that kind of thing). It's a reasonable assumption, not something learned from real ad performance data.
