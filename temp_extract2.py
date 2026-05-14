
import os, re, json, glob
log_dir = "C:/Code/dev/PythonProject/timematch/logs/v255b_source_weight_ablation_20260514_134715"
files = sorted(glob.glob(os.path.join(log_dir, "*structure_light.log")))
file_names = [os.path.basename(f) for f in files]

all_data = []
for fpath in files:
    fname = os.path.basename(fpath)
    with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    variant = ""
    weights = ""
    for line in lines[:30]:
        if line.startswith("V255B_VARIANT"):
            variant = line.strip()
        if line.startswith("V255B_WEIGHTS"):
            weights = line.strip()

    weight_params = {}
    if weights:
        parts = weights.replace("V255B_WEIGHTS|structure_light|", "").split("|")
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                weight_params[k.strip()] = v.strip()

    tail = lines[-300:]
    extracted = {
        "file": fname,
        "variant": variant,
        "weights_raw": weights,
        "weight_params": weight_params,
        "epochs": [],
        "final_metrics": {},
        "errors": [],
        "timestamps": [],
        "lines_total": len(lines),
    }

    for line in tail:
        ts_match = re.search(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})", line)
        if ts_match:
            extracted["timestamps"].append(ts_match.group(1))
        if re.search(r"error|exception|traceback|failed|fail", line, re.IGNORECASE):
            extracted["errors"].append(line.strip())

        m = re.search(r"Epoch\s+(\d+)/(\d+).*?Train Acc[^:]*:\s*([\d.]+).*?Val Acc[^:]*:\s*([\d.]+)", line, re.IGNORECASE)
        if m:
            extracted["epochs"].append({"epoch": int(m.group(1)), "train_acc": float(m.group(3)), "val_acc": float(m.group(4))})
            continue

        m = re.search(r"(?:Test |Final).*?(?:accuracy|acc|OA)[^:]*:\s*([\d.]+)", line, re.IGNORECASE)
        if m:
            extracted["final_metrics"]["test_accuracy"] = float(m.group(1))
        m = re.search(r"(?:Avg |Average |mean).*?(?:F1|f1)[^:]*:\s*([\d.]+)", line, re.IGNORECASE)
        if m:
            extracted["final_metrics"]["avg_f1"] = float(m.group(1))
        m = re.search(r"(?:Kappa|kappa)[^:]*:\s*([\d.]+)", line)
        if m:
            extracted["final_metrics"]["kappa"] = float(m.group(1))
        m = re.search(r"AA[^:]*:\s*([\d.]+)", line)
        if m:
            extracted["final_metrics"]["aa"] = float(m.group(1))
        m = re.search(r"Overall\s+Accuracy.*?:\s*([\d.]+)", line, re.IGNORECASE)
        if m:
            extracted["final_metrics"]["overall_accuracy"] = float(m.group(1))

    for line in lines[:200]:
        if re.search(r"error|exception|traceback|failed|CUDA|out of memory|OOM", line, re.IGNORECASE):
            if line.strip() not in extracted["errors"]:
                extracted["errors"].append(line.strip())

    all_data.append(extracted)
    print(f"=== {fname} ({len(lines)} lines) ===")
    print(f"  Variant: {variant}")
    print(f"  Weights: {weights}")
    print(f"  Final metrics: {json.dumps(extracted["final_metrics"])}")
    if extracted["epochs"]:
        print(f"  Last epoch: epoch {extracted["epochs"][-1]["epoch"]} train_acc={extracted["epochs"][-1]["train_acc"]} val_acc={extracted["epochs"][-1]["val_acc"]}")
    if extracted["errors"]:
        print(f"  Errors: {len(extracted["errors"])}")
        for e in extracted["errors"][:5]:
            print(f"    {e}")
    print()

rows = []
for d in all_data:
    rows.append({
        "timestamp": d["timestamps"][0] if d["timestamps"] else "N/A",
        "file": d["file"],
        "variant": d["variant"],
        "test_accuracy": d["final_metrics"].get("test_accuracy", "N/A"),
        "avg_f1": d["final_metrics"].get("avg_f1", "N/A"),
        "kappa": d["final_metrics"].get("kappa", "N/A"),
        "aa": d["final_metrics"].get("aa", "N/A"),
        "overall_accuracy": d["final_metrics"].get("overall_accuracy", "N/A"),
        "last_val_acc": d["epochs"][-1]["val_acc"] if d["epochs"] else "N/A",
        "last_train_acc": d["epochs"][-1]["train_acc"] if d["epochs"] else "N/A",
        "errors_count": len(d["errors"]),
        "total_lines": d["lines_total"],
        "weight_params": d["weight_params"]
    })

output = {
    "files_found": file_names,
    "extracted_rows": rows,
    "oracle_alignment": {"matched": 0, "mismatched": 0},
    "discrepancies": [],
    "summary": f"Extracted data from {len(files)} structure_light.log files in v255b_source_weight_ablation experiment. " +                 f"Each file corresponds to a different GPU/source-target pair. " +                 f"Oracle alignment was skipped as requested."
}
print("=== JSON OUTPUT ===")
print(json.dumps(output, indent=2))
