"""Compact logging helpers for v3.2.1 local shift."""

from __future__ import annotations

import csv
import os
from typing import Mapping


def append_local_shift_tsv(path: str, row: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(dict(row))
