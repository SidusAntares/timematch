"""Source-structure argparse helpers."""


def add_source_structure_args(parser):
    parser.add_argument("--source_structure_intra_trade_off", type=float, default=0.0)
    parser.add_argument("--source_structure_loss_version", type=str, default="compactness")
    return parser
