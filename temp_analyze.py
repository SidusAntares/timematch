import os
log_dir = "C:/Code/dev/PythonProject/timematch/logs/v255b_source_weight_ablation_20260514_134715"
fpath = os.path.join(log_dir, "gpu1_31TCJ_to_32VNH_structure_light.log")
with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
    lines = fh.readlines()
print("=== GPU1 - Lines 30-130 ===")
for line in lines[30:130]:
    print(line.rstrip())
print()
print("=== GPU1 - Searching for TimeMatch start ===")
for i, line in enumerate(lines):
    if "TimeMatch" in line and "Epoch 1" in line:
        print(f"Line {i}: {line.rstrip()}")
        for j in range(max(0,i-5), min(len(lines), i+5)):
            print(f"  {j}: {lines[j].rstrip()}")
        break
