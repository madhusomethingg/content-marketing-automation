"""
Kubeflow Pipelines (KFP v2) definition for the content-marketing ranking system.

Pipeline stages:
  1. download_spotify_data        - pull real, open Spotify track data (TidyTuesday mirror)
  2. build_artist_features        - aggregate to artist level, build a real-signal proxy
                                     label for "user acquisition value" (SIMULATED outcome,
                                     real inputs -- see README for full disclosure)
  3. generate_ad_creatives        - synthesize ad-creative variants per artist with a
                                     documented, hand-authored affinity model (SIMULATED)
  4. train_ranker (x2)            - XGBoost learning-to-rank (rank:ndcg), once for artists
                                     (grouped by genre), once for ad creatives (grouped by
                                     artist)
  5. generate_campaign_recommendations - combine both rankers into a final ranked
                                     artist x creative campaign priority list

Every component is fully self-contained (only stdlib + packages_to_install), which is
what makes it portable to a real Kubeflow cluster unchanged -- lightweight KFP components
get shipped as their own container, so they cannot rely on local sibling imports.
"""

from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics

SPOTIFY_DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2020/2020-01-21/spotify_songs.csv"
)

ARTIST_FEATURE_COLS = ",".join([
    "n_tracks", "n_albums", "playlist_reach", "n_genres", "n_subgenres",
    "avg_popularity", "max_popularity", "std_popularity", "recency_score",
    "avg_danceability", "avg_energy", "avg_loudness", "avg_speechiness",
    "avg_acousticness", "avg_instrumentalness", "avg_liveness", "avg_valence",
    "avg_tempo", "dominant_genre",
])
ARTIST_CATEGORICAL_COLS = "dominant_genre"

CREATIVE_FEATURE_COLS = ",".join([
    "avg_popularity", "playlist_reach", "avg_danceability", "avg_energy",
    "avg_valence", "dominant_genre",
    "creative_format", "hook_type", "cta_type", "visual_style", "placement",
])
CREATIVE_CATEGORICAL_COLS = ",".join([
    "dominant_genre", "creative_format", "hook_type", "cta_type", "visual_style", "placement",
])

BASE_PACKAGES = ["pandas==2.2.2", "numpy==1.26.4"]
ML_PACKAGES = BASE_PACKAGES + ["scikit-learn==1.5.0", "xgboost==2.1.1"]


# --------------------------------------------------------------------------------------
# 1. Data ingestion
# --------------------------------------------------------------------------------------
@dsl.component(base_image="python:3.11", packages_to_install=BASE_PACKAGES)
def download_spotify_data(data_url: str, tracks_csv: Output[Dataset]):
    """Download the open TidyTuesday mirror of Spotify's track/audio-feature data."""
    import urllib.request

    urllib.request.urlretrieve(data_url, tracks_csv.path)


# --------------------------------------------------------------------------------------
# 2. Artist-level feature engineering + acquisition-value proxy label
# --------------------------------------------------------------------------------------
@dsl.component(base_image="python:3.11", packages_to_install=BASE_PACKAGES)
def build_artist_features(
    tracks_csv: Input[Dataset],
    artist_features_csv: Output[Dataset],
    metrics: Output[Metrics],
    min_tracks: int = 2,
    random_seed: int = 42,
):
    """Aggregate raw tracks to one row per artist and build the ranking label.

    All feature columns are real Spotify signals. The *label* (campaign_lift_score /
    relevance_grade) is a SIMULATED proxy for "user-acquisition lift if this artist is
    promoted" -- no public dataset of real UA campaign outcomes exists. It is built as a
    documented, weighted combination of real signals (current popularity, playlist reach,
    release recency, sonic "ad appeal") plus Gaussian noise representing the real-world
    unpredictability of campaign performance. This keeps the ranking task genuinely
    learnable rather than a tautology, while being transparent that it is not ground truth.
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(tracks_csv.path)
    df = df.dropna(subset=["track_artist", "track_album_release_date"]).reset_index(drop=True)
    df["track_album_release_date"] = pd.to_datetime(df["track_album_release_date"], errors="coerce")
    df = df.dropna(subset=["track_album_release_date"])

    audio_cols = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    ]

    def mode_or_first(s):
        m = s.mode()
        return m.iloc[0] if len(m) else s.iloc[0]

    agg = df.groupby("track_artist").agg(
        n_tracks=("track_id", "count"),
        n_albums=("track_album_id", "nunique"),
        playlist_reach=("playlist_id", "nunique"),
        n_genres=("playlist_genre", "nunique"),
        n_subgenres=("playlist_subgenre", "nunique"),
        avg_popularity=("track_popularity", "mean"),
        max_popularity=("track_popularity", "max"),
        std_popularity=("track_popularity", "std"),
        latest_release=("track_album_release_date", "max"),
        dominant_genre=("playlist_genre", mode_or_first),
        dominant_subgenre=("playlist_subgenre", mode_or_first),
        **{f"avg_{c}": (c, "mean") for c in audio_cols},
    ).reset_index()

    agg["std_popularity"] = agg["std_popularity"].fillna(0.0)
    agg = agg[agg["n_tracks"] >= min_tracks].reset_index(drop=True)

    max_date = agg["latest_release"].max()
    agg["days_since_release"] = (max_date - agg["latest_release"]).dt.days
    agg["recency_score"] = 1.0 / (1.0 + agg["days_since_release"] / 365.0)

    def zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0.0

    rng = np.random.default_rng(random_seed)

    known_signal = (
        0.5 * zscore(agg["avg_popularity"])
        + 0.3 * zscore(agg["playlist_reach"])
        + 0.2 * zscore(agg["recency_score"])
    )
    ad_appeal = (
        0.15 * zscore(agg["avg_danceability"])
        + 0.15 * zscore(agg["avg_energy"])
        - 0.10 * zscore(agg["avg_instrumentalness"])
    )
    # Noise is deliberately larger than the deterministic signal: real campaign outcomes
    # are not fully predictable from current metrics alone, and this keeps the ranking
    # task genuinely learnable rather than a tautological recovery of the label formula.
    noise = rng.normal(0, 0.8, size=len(agg))

    # SIMULATED LABEL -- see docstring.
    agg["campaign_lift_score"] = known_signal + ad_appeal + noise

    def bucket_grade(group, n_bins=10):
        try:
            return pd.qcut(group, n_bins, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(0, index=group.index)

    agg["relevance_grade"] = (
        agg.groupby("dominant_genre")["campaign_lift_score"]
        .transform(bucket_grade)
        .fillna(0)
        .astype(int)
    )

    agg["artist_id"] = agg["track_artist"]
    agg.to_csv(artist_features_csv.path, index=False)

    metrics.log_metric("n_artists", float(len(agg)))
    metrics.log_metric("n_genres", float(agg["dominant_genre"].nunique()))
    metrics.log_metric("avg_tracks_per_artist", float(agg["n_tracks"].mean()))
    metrics.log_metric("dropped_below_min_tracks", float(df["track_artist"].nunique() - len(agg)))


# --------------------------------------------------------------------------------------
# 3. Synthetic ad-creative generation
# --------------------------------------------------------------------------------------
@dsl.component(base_image="python:3.11", packages_to_install=BASE_PACKAGES)
def generate_ad_creatives(
    artist_features_csv: Input[Dataset],
    creatives_csv: Output[Dataset],
    metrics: Output[Metrics],
    top_n_per_genre: int = 80,
    creatives_per_artist: int = 8,
    random_seed: int = 42,
):
    """Synthesize ad-creative variants for a subset of artists.

    Spotify does not publish ad-creative or engagement data, so this entire stage is
    SIMULATED: creative attributes are sampled from a defined vocabulary, and the
    engagement label is generated from a hand-authored, documented affinity model (which
    creative attributes plausibly pair well with which genre / format) plus noise. This is
    disclosed as simulated throughout the README and is not real Spotify or ad-platform data.
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(artist_features_csv.path)

    # Keep the creative-ranking stage tractable: focus on the top artists per genre by
    # current popularity, which is who a UA team would realistically shortlist first.
    top = (
        df.sort_values("avg_popularity", ascending=False)
        .groupby("dominant_genre")
        .head(top_n_per_genre)
        .reset_index(drop=True)
    )

    formats = ["video", "carousel", "static_image", "story_ad"]
    hooks = ["genre_tag", "mood_based", "lyric_snippet", "artist_story", "trending_challenge"]
    ctas = ["try_free", "listen_now", "add_to_playlist", "follow_artist"]
    visual_styles = ["dark_moody", "vibrant_colorful", "minimal_clean", "retro_grain"]
    placements = ["instagram_feed", "instagram_story", "tiktok", "youtube_preroll", "display_banner"]

    # Hand-authored domain affinities (SIMULATED assumptions, documented in README).
    genre_format_affinity = {
        "edm": {"video": 0.6, "story_ad": 0.4}, "rap": {"video": 0.5, "story_ad": 0.5},
        "pop": {"video": 0.5, "carousel": 0.3}, "latin": {"video": 0.4, "story_ad": 0.4},
        "r&b": {"carousel": 0.4, "static_image": 0.3}, "rock": {"carousel": 0.4, "static_image": 0.4},
    }
    format_placement_affinity = {
        "video": {"tiktok": 0.5, "youtube_preroll": 0.4, "instagram_story": 0.3},
        "story_ad": {"instagram_story": 0.5, "tiktok": 0.3},
        "carousel": {"instagram_feed": 0.4, "display_banner": 0.2},
        "static_image": {"display_banner": 0.4, "instagram_feed": 0.3},
    }
    genre_hook_affinity = {
        "edm": {"trending_challenge": 0.4, "mood_based": 0.3},
        "rap": {"lyric_snippet": 0.5, "trending_challenge": 0.3},
        "pop": {"trending_challenge": 0.4, "artist_story": 0.2},
        "latin": {"mood_based": 0.4, "trending_challenge": 0.3},
        "r&b": {"mood_based": 0.4, "artist_story": 0.3},
        "rock": {"artist_story": 0.4, "genre_tag": 0.3},
    }

    rng = np.random.default_rng(random_seed)
    combo_pool = [
        (f, h, c, v, p)
        for f in formats for h in hooks for c in ctas for v in visual_styles for p in placements
    ]

    rows = []
    for _, artist in top.iterrows():
        n = min(creatives_per_artist, len(combo_pool))
        chosen_idx = rng.choice(len(combo_pool), size=n, replace=False)
        for idx in chosen_idx:
            fmt, hook, cta, style, placement = combo_pool[idx]
            base = (
                0.5 * artist["avg_popularity"] / 100.0
                + 0.3 * min(artist["playlist_reach"] / 50.0, 1.0)
                + 0.2 * artist["avg_danceability"]
            )
            appeal = 0.1 * artist["avg_energy"] + 0.1 * artist["avg_valence"]
            fmt_bonus = genre_format_affinity.get(artist["dominant_genre"], {}).get(fmt, 0.0)
            hook_bonus = genre_hook_affinity.get(artist["dominant_genre"], {}).get(hook, 0.0)
            placement_bonus = format_placement_affinity.get(fmt, {}).get(placement, 0.0)
            noise = rng.normal(0, 0.45)

            # SIMULATED LABEL -- see docstring.
            engagement_score = base + appeal + fmt_bonus + hook_bonus + placement_bonus + noise

            rows.append({
                "artist_id": artist["artist_id"],
                "dominant_genre": artist["dominant_genre"],
                "avg_popularity": artist["avg_popularity"],
                "playlist_reach": artist["playlist_reach"],
                "avg_danceability": artist["avg_danceability"],
                "avg_energy": artist["avg_energy"],
                "avg_valence": artist["avg_valence"],
                "creative_format": fmt,
                "hook_type": hook,
                "cta_type": cta,
                "visual_style": style,
                "placement": placement,
                "engagement_score": engagement_score,
            })

    creatives = pd.DataFrame(rows)

    def bucket_grade(group, n_bins=5):
        try:
            return pd.qcut(group, n_bins, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(0, index=group.index)

    creatives["relevance_grade"] = (
        creatives.groupby("artist_id")["engagement_score"]
        .transform(bucket_grade)
        .fillna(0)
        .astype(int)
    )

    creatives.to_csv(creatives_csv.path, index=False)

    metrics.log_metric("n_creatives", float(len(creatives)))
    metrics.log_metric("n_artists_with_creatives", float(creatives["artist_id"].nunique()))


# --------------------------------------------------------------------------------------
# 4. Generic XGBoost learning-to-rank trainer (reused for artists and creatives)
# --------------------------------------------------------------------------------------
@dsl.component(base_image="python:3.11", packages_to_install=ML_PACKAGES)
def train_ranker(
    features_csv: Input[Dataset],
    model_out: Output[Model],
    metrics: Output[Metrics],
    feature_cols: str,
    categorical_cols: str,
    label_col: str,
    qid_col: str,
    model_name: str,
    test_size: float = 0.2,
    random_seed: int = 42,
):
    """Train an XGBoost rank:ndcg model, grouped by qid_col, with a held-out group split."""
    import json
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import GroupShuffleSplit

    df = pd.read_csv(features_csv.path).reset_index(drop=True)
    feats = [c for c in feature_cols.split(",") if c]
    cats = [c for c in categorical_cols.split(",") if c]

    X = pd.get_dummies(df[feats], columns=cats, drop_first=False)
    y = df[label_col].astype(int).to_numpy()
    # XGBoost's qid must be numeric (it's packed as uint32 internally), so encode the
    # raw group keys (artist names / genre strings) to integer codes.
    groups = pd.factorize(df[qid_col].astype(str))[0]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    def sort_by_qid(idx):
        order = np.argsort(groups[idx], kind="stable")
        return idx[order]

    train_idx = sort_by_qid(train_idx)
    test_idx = sort_by_qid(test_idx)

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    qid_train, qid_test = groups[train_idx], groups[test_idx]

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric=["ndcg@5"],
        n_estimators=80,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        random_state=random_seed,
    )
    model.fit(
        X_train, y_train, qid=qid_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_qid=[qid_train, qid_test],
        verbose=False,
    )

    results = model.evals_result()
    train_ndcg = results["validation_0"]["ndcg@5"][-1]
    test_ndcg = results["validation_1"]["ndcg@5"][-1]

    model.get_booster().save_model(model_out.path + ".json")
    with open(model_out.path + ".columns.json", "w") as f:
        json.dump({"columns": list(X.columns), "feature_cols": feats, "categorical_cols": cats}, f)

    metrics.log_metric("train_ndcg_at_5", float(train_ndcg))
    metrics.log_metric("test_ndcg_at_5", float(test_ndcg))
    metrics.log_metric("n_train", float(len(X_train)))
    metrics.log_metric("n_test", float(len(X_test)))
    metrics.log_metric("n_groups", float(pd.Series(groups).nunique()))

    importances = model.feature_importances_
    for i in np.argsort(importances)[::-1][:5]:
        metrics.log_metric(f"top_feature__{X.columns[i]}", float(importances[i]))


# --------------------------------------------------------------------------------------
# 5. Combine both rankers into a final campaign recommendation list
# --------------------------------------------------------------------------------------
@dsl.component(base_image="python:3.11", packages_to_install=ML_PACKAGES)
def generate_campaign_recommendations(
    artist_features_csv: Input[Dataset],
    artist_model: Input[Model],
    creatives_csv: Input[Dataset],
    creative_model: Input[Model],
    recommendations_csv: Output[Dataset],
    metrics: Output[Metrics],
    artist_feature_cols: str,
    artist_categorical_cols: str,
    creative_feature_cols: str,
    creative_categorical_cols: str,
    top_k_artists: int = 25,
    top_creatives_per_artist: int = 1,
):
    import json
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    def score(df, feats, cats, model_path):
        X = pd.get_dummies(df[feats], columns=cats, drop_first=False)
        with open(model_path + ".columns.json") as f:
            meta = json.load(f)
        X = X.reindex(columns=meta["columns"], fill_value=0)
        booster = xgb.Booster()
        booster.load_model(model_path + ".json")
        return booster.predict(xgb.DMatrix(X))

    artists = pd.read_csv(artist_features_csv.path).reset_index(drop=True)
    creatives = pd.read_csv(creatives_csv.path).reset_index(drop=True)

    artist_feats = [c for c in artist_feature_cols.split(",") if c]
    artist_cats = [c for c in artist_categorical_cols.split(",") if c]
    creative_feats = [c for c in creative_feature_cols.split(",") if c]
    creative_cats = [c for c in creative_categorical_cols.split(",") if c]

    artists["artist_rank_score"] = score(artists, artist_feats, artist_cats, artist_model.path)
    creatives["creative_rank_score"] = score(creatives, creative_feats, creative_cats, creative_model.path)

    # Only artists that also have generated creative variants can receive a full
    # artist+creative recommendation.
    eligible_artists = set(creatives["artist_id"].unique())
    ranked_artists = (
        artists[artists["artist_id"].isin(eligible_artists)]
        .sort_values("artist_rank_score", ascending=False)
        .head(top_k_artists)
    )

    best_creatives = (
        creatives.sort_values("creative_rank_score", ascending=False)
        .groupby("artist_id")
        .head(top_creatives_per_artist)
    )

    merged = ranked_artists.merge(best_creatives, on="artist_id", suffixes=("", "_creative"))

    def normalize(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0.0

    merged["combined_priority_score"] = (
        0.6 * normalize(merged["artist_rank_score"]) + 0.4 * normalize(merged["creative_rank_score"])
    )
    merged = merged.sort_values("combined_priority_score", ascending=False).reset_index(drop=True)
    merged.insert(0, "campaign_rank", merged.index + 1)

    out_cols = [
        "campaign_rank", "artist_id", "dominant_genre", "avg_popularity", "playlist_reach",
        "artist_rank_score", "creative_format", "hook_type", "cta_type", "visual_style",
        "placement", "creative_rank_score", "combined_priority_score",
    ]
    merged[out_cols].to_csv(recommendations_csv.path, index=False)

    metrics.log_metric("n_recommendations", float(len(merged)))
    metrics.log_metric("avg_combined_priority_score", float(merged["combined_priority_score"].mean()))


# --------------------------------------------------------------------------------------
# Pipeline DAG
# --------------------------------------------------------------------------------------
@dsl.pipeline(
    name="content-marketing-ranking-pipeline",
    description="Rank artists and ad creatives for automated user-acquisition campaigns using XGBoost learning-to-rank.",
)
def content_marketing_pipeline(
    data_url: str = SPOTIFY_DATA_URL,
    min_tracks_per_artist: int = 2,
    top_n_per_genre: int = 80,
    creatives_per_artist: int = 8,
    top_k_artists: int = 25,
    random_seed: int = 42,
):
    download_task = download_spotify_data(data_url=data_url)

    artist_feat_task = build_artist_features(
        tracks_csv=download_task.outputs["tracks_csv"],
        min_tracks=min_tracks_per_artist,
        random_seed=random_seed,
    ).set_display_name("build-artist-features")

    creatives_task = generate_ad_creatives(
        artist_features_csv=artist_feat_task.outputs["artist_features_csv"],
        top_n_per_genre=top_n_per_genre,
        creatives_per_artist=creatives_per_artist,
        random_seed=random_seed,
    ).set_display_name("generate-ad-creatives")

    artist_ranker_task = train_ranker(
        features_csv=artist_feat_task.outputs["artist_features_csv"],
        feature_cols=ARTIST_FEATURE_COLS,
        categorical_cols=ARTIST_CATEGORICAL_COLS,
        label_col="relevance_grade",
        qid_col="dominant_genre",
        model_name="artist_ranker",
        random_seed=random_seed,
    ).set_display_name("train-artist-ranker")

    creative_ranker_task = train_ranker(
        features_csv=creatives_task.outputs["creatives_csv"],
        feature_cols=CREATIVE_FEATURE_COLS,
        categorical_cols=CREATIVE_CATEGORICAL_COLS,
        label_col="relevance_grade",
        qid_col="artist_id",
        model_name="creative_ranker",
        random_seed=random_seed,
    ).set_display_name("train-creative-ranker")

    generate_campaign_recommendations(
        artist_features_csv=artist_feat_task.outputs["artist_features_csv"],
        artist_model=artist_ranker_task.outputs["model_out"],
        creatives_csv=creatives_task.outputs["creatives_csv"],
        creative_model=creative_ranker_task.outputs["model_out"],
        artist_feature_cols=ARTIST_FEATURE_COLS,
        artist_categorical_cols=ARTIST_CATEGORICAL_COLS,
        creative_feature_cols=CREATIVE_FEATURE_COLS,
        creative_categorical_cols=CREATIVE_CATEGORICAL_COLS,
        top_k_artists=top_k_artists,
    ).set_display_name("generate-campaign-recommendations")


if __name__ == "__main__":
    compiler.Compiler().compile(content_marketing_pipeline, __file__.replace(".py", "_compiled.yaml"))
    print("Compiled pipeline to pipeline_compiled.yaml")
