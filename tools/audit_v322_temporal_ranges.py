"""Audit real TimeMatch temporal metadata without loading labels or training."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.temporal_alignment.temporal_range_audit import (  # noqa: E402
    compare_temporal_summaries,
    derive_embedding_range,
    load_domain_temporal_data,
    summarize_temporal_positions,
)
from tools.v28_audit_utils import file_sha256  # noqa: E402


SCHEMA_VERSION = "v322-temporal-range-audit-v1"
POSITION_DEFINITION = (
    "absolute calendar-day difference between each metadata date and the "
    "domain metadata start_date, matching PixelSetData.days_after"
)
TSV_FIELDS = [
    "domain_role",
    "domain",
    "split",
    "sample_count",
    "start_date_raw",
    "start_date_parsed",
    "position_dtype",
    "position_min",
    "position_max",
    "negative_position_count",
    "total_position_count",
    "duplicate_position_sample_count",
    "non_increasing_sample_count",
    "sequence_length_min",
    "sequence_length_max",
    "sequence_length_mean",
    "temporal_span_min",
    "temporal_span_max",
    "temporal_span_mean",
    "gap_min",
    "gap_max",
    "gap_mean",
    "embedding_legal_raw_min",
    "embedding_legal_raw_max",
    "embedding_out_of_range_point_count",
    "embedding_out_of_range_sample_count",
]
FLOAT_FIELDS = {
    "sequence_length_mean",
    "temporal_span_min",
    "temporal_span_max",
    "temporal_span_mean",
    "gap_min",
    "gap_max",
    "gap_mean",
}


def _repository_identity(
    repository_root: Path,
    explicit_branch: Optional[str] = None,
    explicit_commit: Optional[str] = None,
) -> dict:
    """Resolve repository identity from CLI values first, then local Git."""

    def git_value(*arguments: str) -> Optional[str]:
        try:
            value = subprocess.check_output(
                ["git", *arguments], cwd=repository_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
        return value or None

    if explicit_branch is not None:
        branch = explicit_branch.strip()
        if not branch:
            raise ValueError("repository branch must be nonempty")
    else:
        branch = git_value("branch", "--show-current")
    if explicit_commit is not None:
        commit = explicit_commit.strip()
        if not commit:
            raise ValueError("repository commit must be a 7 to 40 hexadecimal Git hash")
    else:
        commit = git_value("rev-parse", "HEAD")

    if not branch:
        raise ValueError(
            "repository branch could not be detected; provide --repository-branch"
        )
    if not commit:
        raise ValueError(
            "repository commit could not be detected; provide --repository-commit"
        )
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", commit) is None:
        raise ValueError("repository commit must be a 7 to 40 hexadecimal Git hash")

    return {
        "branch": branch,
        "commit": commit,
    }


def _checkpoint_audit(path: Path, source_method: str, source_domain: str, seed: int) -> dict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(path),
        "resolved_absolute_path": str(resolved),
        "sha256": file_sha256(resolved),
        "file_size_bytes": stat.st_size,
        "modification_time_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "source_method": source_method,
        "source_domain": source_domain,
        "source_seed": seed,
    }


def _format_tsv_value(field: str, value) -> str:
    if value is None:
        return ""
    if field in FLOAT_FIELDS:
        return f"{float(value):.6f}"
    if field in {"position_min", "position_max"} and float(value).is_integer():
        return str(int(value))
    return str(value)


def _write_outputs(document: dict, output_json: Path, output_tsv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        for role in ("source", "target"):
            row = {"domain_role": role, **document[role]}
            writer.writerow(
                {field: _format_tsv_value(field, row.get(field)) for field in TSV_FIELDS}
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source", required=True, help="country/tile/year source dataset")
    parser.add_argument("--target", required=True, help="country/tile/year target dataset")
    parser.add_argument("--source-split", default="all")
    parser.add_argument("--target-split", default="all")
    parser.add_argument("--source-indices", default=None)
    parser.add_argument("--target-indices", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-method", required=True)
    parser.add_argument("--source-seed", required=True, type=int)
    parser.add_argument("--max-position", default=365, type=int)
    parser.add_argument("--max-temporal-shift", default=100, type=int)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--repository-branch", default=None)
    parser.add_argument("--repository-commit", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tsv", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    embedding = derive_embedding_range(args.max_position, args.max_temporal_shift)
    source_data = load_domain_temporal_data(
        args.data_root,
        args.source,
        split=args.source_split,
        indices_path=args.source_indices,
    )
    target_data = load_domain_temporal_data(
        args.data_root,
        args.target,
        split=args.target_split,
        indices_path=args.target_indices,
    )
    source_summary = summarize_temporal_positions(
        [source_data.position_sequence] * source_data.sample_count,
        domain=source_data.domain,
        split=source_data.split,
        start_date=source_data.start_date_raw,
        embedding_legal_raw_min=embedding.legal_raw_min,
        embedding_legal_raw_max=embedding.legal_raw_max,
    )
    target_summary = summarize_temporal_positions(
        [target_data.position_sequence] * target_data.sample_count,
        domain=target_data.domain,
        split=target_data.split,
        start_date=target_data.start_date_raw,
        embedding_legal_raw_min=embedding.legal_raw_min,
        embedding_legal_raw_max=embedding.legal_raw_max,
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "repository": _repository_identity(
            Path(args.repository_root).resolve(),
            explicit_branch=args.repository_branch,
            explicit_commit=args.repository_commit,
        ),
        "dataset": {
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "position_unit": "days",
            "position_definition": POSITION_DEFINITION,
            "source_metadata_path": source_data.metadata_path,
            "target_metadata_path": target_data.metadata_path,
        },
        "embedding": asdict(embedding),
        "source": asdict(source_summary),
        "target": asdict(target_summary),
        "comparison": asdict(compare_temporal_summaries(source_summary, target_summary)),
        "checkpoint": _checkpoint_audit(
            Path(args.checkpoint), args.source_method, args.source, args.source_seed
        ),
    }
    _write_outputs(document, Path(args.output_json), Path(args.output_tsv))
    print(
        "V322_TEMPORAL_RANGE_AUDIT|"
        f"source={args.source}|target={args.target}|"
        f"json={Path(args.output_json).resolve()}|tsv={Path(args.output_tsv).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
