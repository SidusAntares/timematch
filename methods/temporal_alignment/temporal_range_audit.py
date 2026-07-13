"""Pure statistics for auditing temporal coordinates and legal ranges."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import pickle
from typing import Iterable, Optional, Sequence, Union

import numpy as np


DateValue = Union[str, int, np.integer]


@dataclass(frozen=True)
class EmbeddingRange:
    """Raw-position range implied by an LTAE positional embedding table."""

    table_length: int
    internal_index_offset: int
    legal_raw_min: int
    legal_raw_max: int


@dataclass(frozen=True)
class DomainTemporalData:
    """Metadata-only temporal data for one domain and one explicit split."""

    domain: str
    split: str
    metadata_path: str
    start_date_raw: str
    start_date_parsed: str
    position_sequence: np.ndarray
    sample_count: int


@dataclass(frozen=True)
class DomainTemporalSummary:
    """Temporal-position statistics for one domain/split."""

    domain: str
    split: str
    sample_count: int
    start_date_raw: str
    start_date_parsed: str
    position_dtype: str
    position_min: float
    position_max: float
    negative_position_count: int
    total_position_count: int
    duplicate_position_sample_count: int
    non_increasing_sample_count: int
    sequence_length_min: int
    sequence_length_max: int
    sequence_length_mean: float
    temporal_span_min: float
    temporal_span_max: float
    temporal_span_mean: float
    gap_min: Optional[float]
    gap_max: Optional[float]
    gap_mean: Optional[float]
    embedding_legal_raw_min: int
    embedding_legal_raw_max: int
    embedding_out_of_range_point_count: int
    embedding_out_of_range_sample_count: int


@dataclass(frozen=True)
class TemporalRangeComparison:
    """Raw-coordinate range comparison between source and target summaries.

    ``start_date_difference_days`` is signed as target minus source. For a
    zero-span interval, its overlap ratio is 1 only when its single coordinate
    lies inside the other closed interval; otherwise it is 0.
    """

    same_start_date: bool
    start_date_difference_days: int
    source_range_min: float
    source_range_max: float
    target_range_min: float
    target_range_max: float
    raw_range_overlap_span: float
    raw_range_overlap_ratio_source: float
    raw_range_overlap_ratio_target: float
    union_span: float
    source_midpoint: float
    target_midpoint: float


def parse_compact_date(value: DateValue) -> datetime:
    """Parse the repository's ``YYYYMMDD`` metadata date representation."""

    text = str(value)
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"date must use YYYYMMDD format, got {value!r}")
    return datetime.strptime(text, "%Y%m%d")


def positions_from_dates(start_date: DateValue, dates: Sequence[DateValue]) -> np.ndarray:
    """Reproduce ``PixelSetData.days_after`` in days without reading labels.

    The repository implementation takes the absolute calendar-day difference
    between every acquisition date and the domain metadata ``start_date``.
    """

    parsed_start = parse_compact_date(start_date)
    return np.asarray(
        [abs((parse_compact_date(date) - parsed_start).days) for date in dates],
        dtype=np.int64,
    )


def derive_embedding_range(max_position: int, max_temporal_shift: int) -> EmbeddingRange:
    """Derive legal raw positions from LTAE's embedding expression.

    LTAE allocates ``max_position + 2 * max_temporal_shift`` rows and indexes
    them with ``raw_position + max_temporal_shift``.
    """

    if max_position <= 0 or max_temporal_shift < 0:
        raise ValueError("max_position must be positive and max_temporal_shift nonnegative")
    table_length = max_position + 2 * max_temporal_shift
    return EmbeddingRange(
        table_length=table_length,
        internal_index_offset=max_temporal_shift,
        legal_raw_min=-max_temporal_shift,
        legal_raw_max=table_length - 1 - max_temporal_shift,
    )


def _read_indices(path: Path, split: str) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"indices file does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if split not in payload:
                raise ValueError(f"indices JSON has no split named {split!r}")
            payload = payload[split]
        indices = [int(value) for value in payload]
    else:
        text = path.read_text(encoding="utf-8")
        indices = [int(value) for value in text.replace(",", " ").split()]
    if len(indices) != len(set(indices)):
        raise ValueError("indices must not contain duplicates")
    return indices


def load_domain_temporal_data(
    data_root: Union[str, Path],
    dataset_name: str,
    *,
    split: str = "all",
    indices_path: Optional[Union[str, Path]] = None,
) -> DomainTemporalData:
    """Read temporal metadata from the same path used by ``PixelSetData``.

    Only ``start_date``, ``dates``, and the number of ``parcels`` are used.
    Parcel labels are never read. A split other than ``all`` requires an
    explicit indices file because training splits are randomly generated and
    are not persisted by the repository.
    """

    metadata_path = Path(data_root) / dataset_name / "meta" / "metadata.pkl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata file does not exist: {metadata_path}")
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    start_date = metadata["start_date"]
    dates = metadata["dates"]
    parcel_count = len(metadata["parcels"])

    if split == "all":
        if indices_path is not None:
            raise ValueError("indices_path must be omitted when split='all'")
        sample_count = parcel_count
    else:
        if indices_path is None:
            raise ValueError("an explicit indices file is required for a non-all split")
        indices = _read_indices(Path(indices_path), split)
        if any(index < 0 or index >= parcel_count for index in indices):
            raise ValueError("indices contain a parcel index outside metadata bounds")
        sample_count = len(indices)
    if sample_count == 0:
        raise ValueError(f"empty split: {dataset_name}/{split}")

    parsed_start = parse_compact_date(start_date)
    return DomainTemporalData(
        domain=dataset_name,
        split=split,
        metadata_path=str(metadata_path.resolve()),
        start_date_raw=str(start_date),
        start_date_parsed=parsed_start.date().isoformat(),
        position_sequence=positions_from_dates(start_date, dates),
        sample_count=sample_count,
    )


def _as_real_finite_sequence(sequence: Iterable[float]) -> np.ndarray:
    values = np.asarray(sequence)
    if values.ndim != 1:
        raise ValueError("each position sequence must be one-dimensional")
    if values.size == 0:
        raise ValueError("position sequences must not be empty")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ) or np.issubdtype(values.dtype, np.bool_):
        raise TypeError("position sequences must contain real numeric values")
    if not np.isfinite(values).all():
        raise ValueError("position sequences must contain only finite values")
    return values


def summarize_temporal_positions(
    samples_or_position_sequences: Iterable[Iterable[float]],
    *,
    domain: str,
    split: str,
    start_date: DateValue,
    embedding_legal_raw_min: int,
    embedding_legal_raw_max: int,
) -> DomainTemporalSummary:
    """Summarize position sequences in their input order without mutation."""

    if embedding_legal_raw_min > embedding_legal_raw_max:
        raise ValueError("embedding legal minimum must not exceed maximum")
    sequences = [_as_real_finite_sequence(sequence) for sequence in samples_or_position_sequences]
    if not sequences:
        raise ValueError(f"empty split: {domain}/{split}")

    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    spans = np.asarray(
        [float(sequence.max() - sequence.min()) for sequence in sequences], dtype=np.float64
    )
    all_positions = np.concatenate(sequences)
    gaps = [np.diff(sequence) for sequence in sequences if len(sequence) >= 2]
    all_gaps = np.concatenate(gaps) if gaps else None
    duplicate_samples = 0
    non_increasing_samples = 0
    out_of_range_samples = 0
    out_of_range_points = 0
    for sequence in sequences:
        differences = np.diff(sequence)
        duplicate_samples += int(bool((differences == 0).any()))
        non_increasing_samples += int(bool((differences <= 0).any()))
        out_of_range = (sequence < embedding_legal_raw_min) | (
            sequence > embedding_legal_raw_max
        )
        out_of_range_points += int(out_of_range.sum())
        out_of_range_samples += int(bool(out_of_range.any()))

    dtype_names = sorted({str(sequence.dtype) for sequence in sequences})
    parsed_start = parse_compact_date(start_date)
    return DomainTemporalSummary(
        domain=domain,
        split=split,
        sample_count=len(sequences),
        start_date_raw=str(start_date),
        start_date_parsed=parsed_start.date().isoformat(),
        position_dtype=dtype_names[0] if len(dtype_names) == 1 else f"mixed:{','.join(dtype_names)}",
        position_min=float(all_positions.min()),
        position_max=float(all_positions.max()),
        negative_position_count=int((all_positions < 0).sum()),
        total_position_count=int(all_positions.size),
        duplicate_position_sample_count=duplicate_samples,
        non_increasing_sample_count=non_increasing_samples,
        sequence_length_min=int(lengths.min()),
        sequence_length_max=int(lengths.max()),
        sequence_length_mean=float(lengths.mean()),
        temporal_span_min=float(spans.min()),
        temporal_span_max=float(spans.max()),
        temporal_span_mean=float(spans.mean()),
        gap_min=None if all_gaps is None else float(all_gaps.min()),
        gap_max=None if all_gaps is None else float(all_gaps.max()),
        gap_mean=None if all_gaps is None else float(all_gaps.mean()),
        embedding_legal_raw_min=int(embedding_legal_raw_min),
        embedding_legal_raw_max=int(embedding_legal_raw_max),
        embedding_out_of_range_point_count=out_of_range_points,
        embedding_out_of_range_sample_count=out_of_range_samples,
    )


def _zero_safe_overlap_ratio(
    interval_min: float,
    interval_max: float,
    other_min: float,
    other_max: float,
    overlap_span: float,
) -> float:
    span = interval_max - interval_min
    if span == 0:
        return float(other_min <= interval_min <= other_max)
    return overlap_span / span


def compare_temporal_summaries(
    source_summary: DomainTemporalSummary,
    target_summary: DomainTemporalSummary,
) -> TemporalRangeComparison:
    """Compare raw source/target ranges without selecting a gate or stretch."""

    source_start = datetime.fromisoformat(source_summary.start_date_parsed)
    target_start = datetime.fromisoformat(target_summary.start_date_parsed)
    source_min, source_max = source_summary.position_min, source_summary.position_max
    target_min, target_max = target_summary.position_min, target_summary.position_max
    overlap_span = max(0.0, min(source_max, target_max) - max(source_min, target_min))
    return TemporalRangeComparison(
        same_start_date=source_start == target_start,
        start_date_difference_days=(target_start - source_start).days,
        source_range_min=source_min,
        source_range_max=source_max,
        target_range_min=target_min,
        target_range_max=target_max,
        raw_range_overlap_span=overlap_span,
        raw_range_overlap_ratio_source=_zero_safe_overlap_ratio(
            source_min, source_max, target_min, target_max, overlap_span
        ),
        raw_range_overlap_ratio_target=_zero_safe_overlap_ratio(
            target_min, target_max, source_min, source_max, overlap_span
        ),
        union_span=max(source_max, target_max) - min(source_min, target_min),
        source_midpoint=(source_min + source_max) / 2,
        target_midpoint=(target_min + target_max) / 2,
    )
