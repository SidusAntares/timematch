"""Shared argparse helpers."""


def add_common_runtime_args(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    return parser
