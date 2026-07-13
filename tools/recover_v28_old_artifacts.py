import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


NAME_TERMS = ("v275", "v276", "closedset", "smoothed", "smooth", "lambda12")
VALUE_TERMS = ("0.6278", "0.6525", "0.1644", "0.1790")
TEXT_SUFFIXES = {".md", ".txt", ".tsv", ".csv", ".log", ".out", ".sh", ".json"}
HISTORY_NAMES = {".bash_history", ".zsh_history", "fish_history"}
SKIP_DIRS = {".git", "runs", "anaconda3", "miniconda3", "conda", "DBL", "timematch_data"}
TASK_RE = re.compile(r"(AT1|DK1|FR1|FR2)_to_(AT1|DK1|FR1|FR2)")
SEED_RE = re.compile(r"seed[_-]?(\d+)", re.I)
F1_PATTERNS = (
    re.compile(r"Test result[^\n]*?macro[_ ]?f1[^0-9]*([0-9.]+)", re.I),
    re.compile(r"macro_f1[^0-9]*([0-9.]+)", re.I),
    re.compile(r"Test result[^\n]*?f1=([0-9.]+)", re.I),
)


def bounded_files(root, max_depth):
    root = Path(root).resolve()
    if not root.exists():
        return
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS and depth < max_depth]
        for name in files:
            yield current_path / name


def classify_match(path):
    lowered = str(path).lower()
    basename = path.name.lower()
    if path.name == "model.pt" and any(term in lowered for term in NAME_TERMS):
        return "checkpoint_name"
    if path.name in {"summary.tsv", "train.log"} and any(term in lowered for term in NAME_TERMS):
        return "summary_or_log_name"
    if path.name in HISTORY_NAMES:
        return "shell_history"
    if any(term in basename or term in lowered for term in NAME_TERMS):
        return "name"
    return ""


def read_text_match(path, max_bytes):
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in HISTORY_NAMES:
        return "", ""
    try:
        if path.stat().st_size > max_bytes:
            return "", ""
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", ""
    hits = [term for term in (*NAME_TERMS, *VALUE_TERMS) if term.lower() in text.lower()]
    return ("content", ",".join(hits)) if hits else ("", "")


def infer_config(text):
    lowered = str(text).lower()
    if "smooth" in lowered:
        return "smooth_k3"
    if any(term in lowered for term in ("plain", "baseline", "base")):
        return "base"
    return "unknown"


def parse_text_result(path):
    if path.suffix.lower() not in {".log", ".out", ".txt", ".tsv", ".csv"}:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    values = []
    for pattern in F1_PATTERNS:
        values.extend(float(value) for value in pattern.findall(text))
    task_match = TASK_RE.search(str(path))
    seed_match = SEED_RE.search(str(path))
    if not values or not task_match:
        return None
    return {
        "path": str(path),
        "task": f"{task_match.group(1)}_to_{task_match.group(2)}",
        "seed": seed_match.group(1) if seed_match else "",
        "config": infer_config(path),
        "last_macro_f1": values[-1],
        "all_macro_f1": ",".join(str(value) for value in values),
        "result_kind": "text_log",
    }


def parse_tabular_results(path):
    if path.suffix.lower() not in {".tsv", ".csv"}:
        return []
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=delimiter))
    except (OSError, csv.Error):
        return []
    output = []
    for row in rows:
        task = row.get("task", "")
        if not TASK_RE.search(task):
            continue
        f1 = next(
            (
                row.get(name)
                for name in (
                    "da_test_macro_f1",
                    "test_macro_f1",
                    "macro_f1",
                    "mean_test_f1",
                    "da_f1",
                )
                if row.get(name) not in (None, "")
            ),
            "",
        )
        try:
            f1 = float(f1)
        except (TypeError, ValueError):
            continue
        output.append(
            {
                "path": str(path),
                "task": task,
                "seed": row.get("seed", ""),
                "config": infer_config(row.get("config", row.get("source_config", ""))),
                "last_macro_f1": f1,
                "all_macro_f1": f1,
                "result_kind": "tabular_row",
            }
        )
    return output


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearby_train_config(path):
    candidates = []
    if path.name == "train_config.json":
        candidates.append(path)
    for parent in [path.parent, *list(path.parents)[:3]]:
        candidates.append(parent / "train_config.json")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            return candidate, json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return candidate, {}
    return None, {}


def artifact_metadata(path):
    task_match = TASK_RE.search(str(path))
    seed_match = SEED_RE.search(str(path))
    config_path, config = nearby_train_config(path)
    source = str(config.get("source", ""))
    target = str(config.get("target", ""))
    task = f"{task_match.group(1)}_to_{task_match.group(2)}" if task_match else ""
    return {
        "task": task,
        "seed": config.get("seed", seed_match.group(1) if seed_match else ""),
        "config": infer_config(f"{path} {config.get('source_structure_loss_version', '')}"),
        "source": source,
        "target": target,
        "closed_set": config.get("closed_set", ""),
        "with_shift_aug": config.get("with_shift_aug", ""),
        "epochs": config.get("epochs", ""),
        "steps_per_epoch": config.get("steps_per_epoch", ""),
        "output_student": config.get("output_student", ""),
        "train_config_path": str(config_path) if config_path else "",
        "can_recompute": bool(path.name == "model.pt" and config_path),
    }


def git_value_hits(repo_root):
    repo_root = Path(repo_root)
    if not (repo_root / ".git").exists():
        return []
    rows = []
    for value in VALUE_TERMS:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "log", "--all", "--oneline", "-S", value],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            rows.append({"value": value, "commit": line})
    return rows


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Bounded recovery of historical v2.8 artifacts.")
    parser.add_argument("--search_root", action="append", required=True)
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--max_text_mb", type=int, default=20)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repo_root", default=".")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    inventory = []
    recovered = []
    for root in args.search_root:
        for path in bounded_files(root, args.max_depth):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            kind = classify_match(path)
            content_kind, hits = read_text_match(path, args.max_text_mb * 1024 * 1024)
            if not kind and not content_kind:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            inventory.append(
                {
                    "path": resolved,
                    "kind": kind or content_kind,
                    "content_hits": hits,
                    "size_bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    "sha256": file_sha256(path) if path.name == "model.pt" else "",
                    **artifact_metadata(path),
                }
            )
            parsed = parse_text_result(path)
            if parsed:
                recovered.append(parsed)
            recovered.extend(parse_tabular_results(path))

    git_hits = git_value_hits(args.repo_root)
    write_tsv(
        output / "old_artifact_inventory.tsv",
        inventory,
        [
            "path",
            "kind",
            "content_hits",
            "size_bytes",
            "mtime",
            "sha256",
            "task",
            "seed",
            "config",
            "source",
            "target",
            "closed_set",
            "with_shift_aug",
            "epochs",
            "steps_per_epoch",
            "output_student",
            "train_config_path",
            "can_recompute",
        ],
    )
    write_tsv(
        output / "recovered_results.tsv",
        recovered,
        ["path", "task", "seed", "config", "last_macro_f1", "all_macro_f1", "result_kind"],
    )
    write_tsv(output / "git_value_hits.tsv", git_hits, ["value", "commit"])

    exact = [row for row in inventory if any(value in row["content_hits"] for value in ("0.6278", "0.6525"))]
    exact_lines = "\n".join(
        f"- `{row['path']}`，修改时间 `{row['mtime']}`，命中 `{row['content_hits']}`"
        for row in exact
    ) or "- 未找到直接包含历史均值的文件。"
    report = Path(args.repo_root) / "analysis/v28_old_artifact_recovery_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# v2.8 旧产物恢复报告\n\n"
        "## 搜索范围\n\n"
        f"- 根目录：`{', '.join(args.search_root)}`\n"
        f"- 最大深度：`{args.max_depth}`\n"
        f"- 命中产物：`{len(inventory)}`\n"
        f"- 可重新解析结果：`{len(recovered)}`\n"
        f"- 直接命中 `0.6278/0.6525` 的文件：`{len(exact)}`\n"
        f"- Git 历史数值命中：`{len(git_hits)}`\n\n"
        "## 历史均值来源候选\n\n"
        f"{exact_lines}\n\n"
        "## 明细\n\n"
        "- 文件与 checkpoint：`old_artifact_inventory.tsv`\n"
        "- 从日志和表格重算的任务结果：`recovered_results.tsv`\n"
        "- Git 历史命中：`git_value_hits.tsv`\n",
        encoding="utf-8",
    )
    print(
        f"RECOVERY_DONE|artifacts={len(inventory)}|results={len(recovered)}|"
        f"exact_sources={len(exact)}|output={output}"
    )


if __name__ == "__main__":
    main()
