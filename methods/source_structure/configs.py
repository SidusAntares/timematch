"""Configuration helpers for source-side structure shaping.

This module keeps source-structure flags separate from active TimeMatch and
future local-shift flags.  It is intentionally small; the current CLI remains
backward compatible through ``train.py``.
"""


def add_source_structure_args(parser):
    """Register source-structure arguments on an argparse parser."""

    parser.add_argument(
        "--source_structure_feature_target",
        type=str,
        default="raw",
        choices=["raw", "reshaped", "both"],
        help="feature stream used by source-side structure losses",
    )
    parser.add_argument(
        "--source_structure_compact_distance",
        type=str,
        default="mse",
        choices=["mse", "cosine"],
        help="distance used by compactness-style source-structure losses",
    )
    return parser
