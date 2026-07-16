"""Run the content-marketing ranking pipeline locally via the KFP local (subprocess) runner.

This executes every pipeline component as a real KFP task -- same code path that would
run on a Kubeflow cluster -- just without a Kubernetes control plane. Useful for
development, CI smoke tests, and demoing the pipeline without cluster infrastructure.
"""

import argparse
import sys
from pathlib import Path

from kfp import local

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from pipeline import content_marketing_pipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n-per-genre", type=int, default=80)
    parser.add_argument("--creatives-per-artist", type=int, default=8)
    parser.add_argument("--top-k-artists", type=int, default=25)
    parser.add_argument("--min-tracks-per-artist", type=int, default=2)
    parser.add_argument("--pipeline-root", default="local_outputs")
    args = parser.parse_args()

    local.init(runner=local.SubprocessRunner(), pipeline_root=args.pipeline_root)

    task = content_marketing_pipeline(
        top_n_per_genre=args.top_n_per_genre,
        creatives_per_artist=args.creatives_per_artist,
        top_k_artists=args.top_k_artists,
        min_tracks_per_artist=args.min_tracks_per_artist,
    )
    print("Pipeline finished. Outputs under:", args.pipeline_root)
    return task


if __name__ == "__main__":
    main()
