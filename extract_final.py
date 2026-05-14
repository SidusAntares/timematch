import os, re, json, glob
log_dir = "C:/Code/dev/PythonProject/timematch/logs/v255b_source_weight_ablation_20260514_134715"
files = sorted(glob.glob(os.path.join(log_dir, "*structure_light.log")))
file_names = [os.path.basename(f) for f in files]

def extract_metrics(fpath):
    with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    result = {"file": os.path.basename(fpath), "total_lines": len(lines), "variant": "", "weights_raw": "", "weight_params": {}, "source_only_val": [], "tm_val": [], "timematch_epochs": [], "timestamps": [], "source_checkpoints_completed": [], "da_test_metrics": []}
    for line in lines[:30]:
        if line.startswith("V255B_VARIANT"): result["variant"] = line.strip()
        if line.startswith("V255B_WEIGHTS"): result["weights_raw"] = line.strip()
    if result["weights_raw"]:
        parts = result["weights_raw"].replace("V255B_WEIGHTS|structure_light|", "").split("|")
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                result["weight_params"][k.strip()] = v.strip()
    tm_start = None
    for i, line in enumerate(lines):
        if "TimeMatch Epoch 1/20" in line:
            tm_start = i
            break
    for line in lines:
        m = re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", line)
        if m and m.group(0) not in result["timestamps"]:
            result["timestamps"].append(m.group(0))
    for i, line in enumerate(lines):
        m = re.search(r"Validation result: loss=([\d.]+), acc=([\d.]+), f1=([\d.]+)", line)
        if m:
            entry = {"loss": float(m.group(1)), "acc": float(m.group(2)), "f1": float(m.group(3))}
            if tm_start is None or i < tm_start:
                result["source_only_val"].append(entry)
            else:
                result["tm_val"].append(entry)
    for line in lines:
        m = re.search(r"TimeMatch Epoch (\d+)/20:.*?loss=([\d.]+)", line)
        if m:
            result["timematch_epochs"].append({"epoch": int(m.group(1)), "loss": float(m.group(2))})
    for line in lines:
        if "V255B_DA_DONE" in line:
            parts = line.strip().split("|")
            da_info = {"source": parts[3] if len(parts) > 3 else "?", "target": parts[4] if len(parts) > 4 else "?", "checkpoint": parts[5] if len(parts) > 5 else "?"}
            result["source_checkpoints_completed"].append(da_info)
        lo = line.lower()
        if "accuracy:" in lo and "±" in line:
            m = re.search(r"accuracy:\s*([\d.]+)", lo)
            if m: result["da_test_metrics"].append(("accuracy", float(m.group(1))))
        if "macro_f1:" in lo and "±" in line:
            m = re.search(r"macro_f1:\s*([\d.]+)", lo)
            if m: result["da_test_metrics"].append(("macro_f1", float(m.group(1))))
        if "weighted_f1:" in lo and "±" in line:
            m = re.search(r"weighted_f1:\s*([\d.]+)", lo)
            if m: result["da_test_metrics"].append(("weighted_f1", float(m.group(1))))
        if lo.startswith("kappa:"):
            m = re.search(r"kappa:\s*([\d.]+)", lo)
            if m: result["da_test_metrics"].append(("kappa", float(m.group(1))))
    result["source_completed"] = any("Epoch 100/100" in line for line in lines)
    result["timematch_started"] = tm_start is not None
    result["timematch_max_epoch"] = max([e["epoch"] for e in result["timematch_epochs"]], default=0)
    result["timematch_last_loss"] = result["timematch_epochs"][-1]["loss"] if result["timematch_epochs"] else "N/A"
    result["source_best_val_f1"] = max(v["f1"] for v in result["source_only_val"]) if result["source_only_val"] else "N/A"
    result["source_last_val_f1"] = result["source_only_val"][-1]["f1"] if result["source_only_val"] else "N/A"
    result["source_last_val_acc"] = result["source_only_val"][-1]["acc"] if result["source_only_val"] else "N/A"
    result["tm_last_val_f1"] = result["tm_val"][-1]["f1"] if result["tm_val"] else "N/A"
    result["tm_last_val_acc"] = result["tm_val"][-1]["acc"] if result["tm_val"] else "N/A"
    return result

all_results = [extract_metrics(f) for f in files]
rows = []
for r in all_results:
    da_test = {}
    for metric_type, value in r["da_test_metrics"]:
        da_test[metric_type] = value
    rows.append({"timestamp": r["timestamps"][0] if r["timestamps"] else "N/A", "file": r["file"], "variant": r["variant"], "weight_params": r["weight_params"], "total_lines": r["total_lines"], "source_completed": r["source_completed"], "source_val_count": len(r["source_only_val"]), "source_best_val_f1": r["source_best_val_f1"], "source_last_val_f1": r["source_last_val_f1"], "source_last_val_acc": r["source_last_val_acc"], "timematch_started": r["timematch_started"], "timematch_max_epoch": r["timematch_max_epoch"], "timematch_last_loss": r["timematch_last_loss"], "da_checkpoints_done": len(r["source_checkpoints_completed"]), "da_test_results": da_test if da_test else "N/A", "tm_last_val_f1": r["tm_last_val_f1"], "tm_last_val_acc": r["tm_last_val_acc"]})
summary = "Extracted data from {} files. ".format(len(files))
for r in rows:
    summary += "{}: epochs={} best_f1={} tm_epoch={} da_done={}. ".format(r["file"], r["source_val_count"], r["source_best_val_f1"], r["timematch_max_epoch"], r["da_checkpoints_done"])
    if r["da_test_results"] != "N/A":
        summary += "DA: {}. ".format(r["da_test_results"])
discrepancies = []
if not rows[0]["source_completed"]:
    discrepancies.append("gpu0 incomplete: stopped at 75/100 source epochs, never reached TimeMatch.")
output = {"files_found": file_names, "extracted_rows": rows, "oracle_alignment": {"matched": 0, "mismatched": 0, "note": "Skipped"}, "discrepancies": discrepancies, "summary": summary}
print(json.dumps(output, indent=2))
