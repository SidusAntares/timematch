"""Position utilities for TimeMatch-style temporal encoders."""


def check_temporal_index_range(model, positions, applied_shift, tag):
    """Raise when shifted positions would exceed the positional embedding table."""
    temporal_encoder = model.temporal_encoder
    min_pos = int(positions.min().item())
    max_pos = int(positions.max().item())
    min_idx = min_pos + int(applied_shift) + temporal_encoder.max_temporal_shift
    max_idx = max_pos + int(applied_shift) + temporal_encoder.max_temporal_shift
    table_size = temporal_encoder.positional_enc.num_embeddings
    if min_idx < 0 or max_idx >= table_size:
        raise ValueError(
            f"{tag} temporal indices out of range: "
            f"positions=[{min_pos}, {max_pos}], shift={applied_shift}, "
            f"embedding_indices=[{min_idx}, {max_idx}], table_size={table_size}."
        )

