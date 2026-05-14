import os
log_dir = "C:/Code/dev/PythonProject/timematch/logs/v255b_source_weight_ablation_20260514_134715"
for fname in ["gpu1_31TCJ_to_32VNH_structure_light.log", "gpu2_32VNH_to_33UVP_structure_light.log", "gpu3_33UVP_to_32VNH_structure_light.log"]:
    fpath = os.path.join(log_dir, fname)
    with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    tm_start = None
    for i, line in enumerate(lines):
        if "TimeMatch Epoch 1/20" in line:
            tm_start = i
            break
    source_val = 0
    tm_val = 0
    for i, line in enumerate(lines):
        if "Validation result:" in line:
            if tm_start is None or i < tm_start:
                source_val += 1
            else:
                tm_val += 1
    print(fname, "source_val=", source_val, "tm_val=", tm_val, "tm_start=", tm_start)
    sv = [i for i, line in enumerate(lines) if "Validation result:" in line and (tm_start is None or i < tm_start)]
    if sv:
        print("  First:", lines[sv[0]].rstrip())
        print("  Last:", lines[sv[-1]].rstrip())
    tv = [i for i, line in enumerate(lines) if "Validation result:" in line and tm_start is not None and i >= tm_start]
    if tv:
        print("  First TM:", lines[tv[0]].rstrip())
        print("  Last TM:", lines[tv[-1]].rstrip())
    for i, line in enumerate(lines):
        if "kappa" in line.lower():
            print("  Kappa at", i, ":", line.rstrip()[:150])
    print()
