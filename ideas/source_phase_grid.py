import torch


def make_phase_grid_positions(positions, grid_count):
    if positions.ndim != 2:
        raise ValueError(f"positions must have shape [B, T], got {tuple(positions.shape)}")
    return torch.arange(
        int(grid_count),
        device=positions.device,
        dtype=positions.dtype,
    ).unsqueeze(0).expand(positions.shape[0], -1)


def project_to_phase_grid(
    spatial_feats,
    positions,
    grid_count=5,
    kernel="linear",
    bandwidth=None,
    min_support=1e-6,
):
    """
    Project irregular observed features [B, T, D] to fixed phase slots [B, K, D].

    This is a conservative structure-view helper: it does not change the model
    path, it only builds a dense phase representation for source structure loss.
    Low-support slots are nearest-filled so the trajectory remains defined, while
    the returned support score exposes which slots are weakly observed.
    """
    if spatial_feats.ndim != 3:
        raise ValueError(f"spatial_feats must have shape [B, T, D], got {tuple(spatial_feats.shape)}")
    if positions.ndim != 2:
        raise ValueError(f"positions must have shape [B, T], got {tuple(positions.shape)}")
    if spatial_feats.shape[:2] != positions.shape:
        raise ValueError(
            "spatial_feats and positions must agree on batch/time dimensions, "
            f"got {tuple(spatial_feats.shape[:2])} and {tuple(positions.shape)}"
        )

    grid_count = int(grid_count)
    if grid_count <= 0:
        raise ValueError(f"grid_count must be positive, got {grid_count}")

    positions_float = positions.to(dtype=spatial_feats.dtype)
    min_pos = positions_float.min(dim=1, keepdim=True).values
    max_pos = positions_float.max(dim=1, keepdim=True).values
    span = (max_pos - min_pos).clamp_min(1.0)
    normalized_positions = (positions_float - min_pos) / span

    anchors = torch.linspace(
        0.0,
        1.0,
        grid_count,
        device=spatial_feats.device,
        dtype=spatial_feats.dtype,
    )
    distances = (normalized_positions.unsqueeze(-1) - anchors.view(1, 1, -1)).abs()

    if bandwidth is None:
        bandwidth = 1.0 / max(grid_count - 1, 1)
    bandwidth = max(float(bandwidth), 1e-6)

    kernel = str(kernel).lower()
    if kernel in {"linear", "triangular"}:
        weights = (1.0 - distances / bandwidth).clamp_min(0.0)
    elif kernel == "gaussian":
        weights = torch.exp(-0.5 * (distances / bandwidth).pow(2))
    else:
        raise ValueError(f"Unsupported phase-grid kernel: {kernel}")

    support = weights.sum(dim=1)
    weighted = torch.einsum("btk,btd->bkd", weights, spatial_feats)
    grid = weighted / support.clamp_min(1e-6).unsqueeze(-1)

    weak_mask = support <= float(min_support)
    if bool(weak_mask.any().item()):
        nearest_idx = distances.argmin(dim=1)
        nearest_feats = torch.gather(
            spatial_feats,
            dim=1,
            index=nearest_idx.unsqueeze(-1).expand(-1, -1, spatial_feats.shape[-1]),
        )
        grid = torch.where(weak_mask.unsqueeze(-1), nearest_feats, grid)

    support = support.clamp(0.0, 1.0)
    return grid, support
