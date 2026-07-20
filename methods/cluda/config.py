from dataclasses import dataclass
from .losses import CLUDALossWeights


def add_cluda_args(parser):
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--cluda_init", choices=("random", "source_weights"), default="random")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=50,
                        help="20 x 50 reproduces the official CLI default of 1,000 total steps")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=.99)
    parser.add_argument("--temperature", type=float, default=.07)
    parser.add_argument("--queue_size", type=int, default=98304)
    parser.add_argument("--num_neighbors", type=int, default=1)
    parser.add_argument("--weight_src_contrastive", type=float, default=1.)
    parser.add_argument("--weight_trg_contrastive", type=float, default=1.)
    parser.add_argument("--weight_cross_domain_nn", type=float, default=1.)
    parser.add_argument("--weight_domain", type=float, default=1.)
    parser.add_argument("--weight_prediction", type=float, default=1.)
    parser.add_argument("--cutout_length", type=int, default=4)
    parser.add_argument("--cutout_prob", type=float, default=.5)
    parser.add_argument("--crop_min_history", type=float, default=.5)
    parser.add_argument("--crop_prob", type=float, default=.5)
    parser.add_argument("--gaussian_std", type=float, default=.1)
    parser.add_argument("--channel_dropout_prob", type=float, default=.1)
    parser.add_argument("--grl_schedule", choices=("official", "training_progress", "constant"), default="official")
    return parser


def add_cluda_model_args(parser):
    """Shared by source-only and full CLUDA so source initialization is shape-compatible."""
    parser.add_argument("--cluda_channels", type=str, default="64-64-64-64-64")
    parser.add_argument("--cluda_hidden_dim", type=int, default=256)
    parser.add_argument("--cluda_kernel_size", type=int, default=3)
    parser.add_argument("--cluda_dilation_factor", type=int, default=2)
    parser.add_argument("--cluda_dropout", type=float, default=0.)
    parser.add_argument("--cluda_max_temporal_shift", type=int, default=100)
    return parser


@dataclass
class CLUDAConfig:
    # Official main/train.py CLI defaults. README sensor overrides are launcher-explicit.
    channels: tuple = (64, 64, 64, 64, 64)
    kernel_size: int = 3
    stride: int = 1
    dilation_factor: int = 2
    dropout: float = 0.0
    hidden_dim: int = 256
    momentum: float = 0.99
    temperature: float = 0.07
    queue_size: int = 98304
    num_neighbors: int = 1
    max_temporal_shift: int = 100
    cutout_length: int = 4
    cutout_prob: float = 0.5
    crop_min_history: float = 0.5
    crop_prob: float = 0.5
    gaussian_std: float = 0.1
    channel_dropout_prob: float = 0.1
    weight_src_contrastive: float = 1.0
    weight_trg_contrastive: float = 1.0
    weight_cross_domain_nn: float = 1.0
    weight_domain: float = 1.0
    weight_prediction: float = 1.0
    grl_schedule: str = "official"

    def __post_init__(self):
        if self.num_neighbors != 1:
            raise ValueError("faithful CLUDA baseline requires num_neighbors=1")

    @property
    def loss_weights(self):
        return CLUDALossWeights(
            self.weight_src_contrastive,
            self.weight_trg_contrastive,
            self.weight_cross_domain_nn,
            self.weight_domain,
            self.weight_prediction,
        )

    @classmethod
    def from_namespace(cls, args):
        names = cls.__dataclass_fields__
        values = {name: getattr(args, name) for name in names if hasattr(args, name)}
        if "cluda_channels" in vars(args):
            values["channels"] = tuple(int(x) for x in args.cluda_channels.split("-"))
        if "cluda_hidden_dim" in vars(args):
            values["hidden_dim"] = args.cluda_hidden_dim
        if "cluda_dropout" in vars(args):
            values["dropout"] = args.cluda_dropout
        if "cluda_kernel_size" in vars(args):
            values["kernel_size"] = args.cluda_kernel_size
        if "cluda_dilation_factor" in vars(args):
            values["dilation_factor"] = args.cluda_dilation_factor
        if "cluda_max_temporal_shift" in vars(args):
            values["max_temporal_shift"] = args.cluda_max_temporal_shift
        return cls(**values)
