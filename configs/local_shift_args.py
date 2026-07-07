"""v3.2.1 local-shift argparse helpers."""


def add_local_shift_args(parser):
    parser.add_argument("--local_shift_trade_off", type=float, default=0.0)
    parser.add_argument("--local_shift_kmax", type=int, default=6)
    parser.add_argument("--local_shift_min_stage_len", type=int, default=2)
    parser.add_argument("--local_shift_temperature", type=float, default=0.1)
    parser.add_argument("--local_shift_clip", type=float, default=60.0)
    return parser
