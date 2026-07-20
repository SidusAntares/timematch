import torch
from torch import nn
import importlib.util
from pathlib import Path

from methods.cluda.tcn import TemporalConvNet


class ReferenceBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, x):
        b = self.source
        y = torch.nn.functional.conv1d(x, b.conv1.weight, b.conv1.bias, padding=b.padding, dilation=b.dilation)
        y = torch.relu(y[:, :, :-b.padding])
        y = torch.nn.functional.conv1d(y, b.conv2.weight, b.conv2.bias, padding=b.padding, dilation=b.dilation)
        y = torch.relu(y[:, :, :-b.padding])
        residual = x if b.downsample is None else torch.nn.functional.conv1d(x, b.downsample.weight, b.downsample.bias)
        return torch.relu(y + residual)


def test_tcn_matches_official_block_arithmetic():
    torch.manual_seed(7)
    model = TemporalConvNet(3, [4, 4], kernel_size=3, dilation_factor=2, dropout=0).eval()
    x = torch.randn(2, 3, 17)
    expected = x
    for block in model.network:
        expected = ReferenceBlock(block)(expected)
    actual = model(x)
    assert torch.max(torch.abs(actual - expected)).item() < 1e-7


def test_tcn_matches_checked_out_official_implementation():
    upstream_file = Path(__file__).resolve().parents[2] / "_external_refs" / "CLUDA" / "utils" / "tcn_no_norm.py"
    if not upstream_file.exists():
        return
    spec = importlib.util.spec_from_file_location("cluda_upstream_tcn", upstream_file)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    local = TemporalConvNet(3, [4, 4], kernel_size=3, dilation_factor=2, dropout=0).eval()
    official = upstream.TemporalConvNet(3, [4, 4], kernel_size=3, dilation_factor=2, dropout=0).eval()
    official.load_state_dict(local.state_dict())
    x = torch.randn(2, 3, 19)
    error = torch.max(torch.abs(local(x) - official(x))).item()
    print(f"TCN_MAX_ABS_ERROR={error:.12g}")
    assert error < 1e-7


def test_positions_change_classifier_output():
    from methods.cluda.model import CLUDATCNClassifier
    torch.manual_seed(3)
    model = CLUDATCNClassifier(2, 3, channels=(4,), kernel_size=2, dropout=0).eval()
    pixels = torch.randn(2, 5, 2, 3)
    valid = torch.ones(2, 5, 3)
    p1 = torch.arange(5).repeat(2, 1)
    p2 = p1 + 100
    assert not torch.allclose(model(pixels, valid, p1, None), model(pixels, valid, p2, None))
