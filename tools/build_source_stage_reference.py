"""Build source class-stage references for v3.2.1.

This is a reserved command-line hook.  The production implementation should
load a trained source checkpoint and call
``build_reference_from_temporal_features`` after exposing encoder temporal
features.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v3.2.1 source stage references.")
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kmax", type=int, default=6)
    parser.add_argument("--min_stage_len", type=int, default=2)
    args = parser.parse_args()
    raise NotImplementedError(
        "Reference building is scaffolded only. Expose temporal encoder features, "
        "then implement checkpoint/data loading here."
    )


if __name__ == "__main__":
    main()
