"""Legacy experimental argparse helpers.

These flags are not active in the base TimeMatch parser.  They document the old
v3.1 stage-contrast CLI surface for archived scripts only.
"""


def add_v31_stage_contrast_args(parser):
    parser.add_argument("--stage_contrast_trade_off", type=float, default=0.0)
    parser.add_argument("--stage_contrast_stage_count", type=int, default=6)
    parser.add_argument("--stage_contrast_temperature", type=float, default=0.1)
    parser.add_argument("--stage_partition_mode", type=str, default="feature_change_topk")
    parser.add_argument("--stage_min_len", type=int, default=2)
    parser.add_argument("--stage_time_radius", type=float, default=30.0)
    parser.add_argument("--stage_time_temperature", type=float, default=10.0)
    parser.add_argument("--stage_contrast_pseudo_threshold", type=float, default=None)
    parser.add_argument("--stage_contrast_feature_kind", type=str, default="spatial")
    parser.add_argument("--stage_contrast_log_path", type=str, default="")
    parser.add_argument("--stage_contrast_debug", action="store_true")
    parser.add_argument("--stage_contrast_backend", type=str, default="class_prototype_fast")
    return parser
