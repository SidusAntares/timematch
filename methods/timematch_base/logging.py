"""Small logging helpers shared by TimeMatch variants."""

import csv
import os

import numpy as np


def format_diag_value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return ""
        return f"{float(value):.6f}"
    return str(value)


def append_diag_tsv(path, row, fields):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: format_diag_value(row.get(field)) for field in fields})


def format_elapsed_seconds(seconds):
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

