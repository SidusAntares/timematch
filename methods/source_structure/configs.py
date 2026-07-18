"""Active source-structure configuration for the research baseline."""


def validate_source_structure_config(config):
    """Reject inactive or silently ignored source-structure combinations."""

    mode = str(config.source_structure_mode)
    weight = float(config.source_structure_weight)
    method = getattr(config, "method", None)
    if mode == "raw_global" and method != "source_structure":
        raise ValueError("raw_global requires the source_structure training method")
    if mode == "raw_global" and weight <= 0.0:
        raise ValueError("raw_global requires source_structure_weight > 0")
    if mode == "off" and weight != 0.0:
        raise ValueError("source_structure_weight must be 0 when source_structure_mode=off")


def add_source_structure_args(parser):
    """Register the only two active source-structure settings."""

    parser.add_argument(
        "--source_structure_mode",
        type=str,
        default="off",
        choices=["off", "raw_global"],
        help="source-stage structure loss; off preserves plain source training",
    )
    parser.add_argument(
        "--source_structure_weight",
        type=float,
        default=0.0,
        help="weight multiplying the unweighted raw-global compactness loss",
    )
    return parser
