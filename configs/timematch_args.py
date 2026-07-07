"""Base TimeMatch argparse helpers."""


def add_timematch_args(parser):
    parser.add_argument("--alpha", type=float, default=0.999)
    parser.add_argument("--max_temporal_shift", type=int, default=60)
    parser.add_argument("--timematch_source_structure_trade_off", type=float, default=0.0)
    return parser
