import os
log_dir = "C:/Code/dev/PythonProject/timematch/logs/v255b_source_weight_ablation_20260514_134715"
# Check gpu2 for source training completion and test results
fpath = os.path.join(log_dir, "gpu2_32VNH_to_33UVP_structure_light.log")
with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
    lines = fh.readlines()
print(f"Total lines: {len(lines)}")
# Find source training end (search for epoch 90-100)
for i, line in enumerate(lines):
    if "Epoch 100/100" in line or "Epoch 99/100" in line or "Epoch 98/100" in line:
        print(f"Line {i}: {line.rstrip()}")
        for j in range(i, min(i+200, len(lines))):
            if "TimeMatch" in lines[j] or "Test" in lines[j] or "test" in lines[j] or "Final" in lines[j] or "result" in lines[j].lower():
                print(f"  {j}: {lines[j].rstrip()}")
        break
# Find TimeMatch start
for i, line in enumerate(lines):
    if "TimeMatch Epoch 1/20:" in line:
        print(f"
TimeMatch start at line {i}")
        break
# Look for any test or eval results
for i, line in enumerate(lines):
    if "test" in line.lower() and ("accuracy" in line.lower() or "f1" in line.lower() or "oa" in line.lower()):
        print(f"Test result at line {i}: {line.rstrip()}")
    if "Final" in line and ("test" in line.lower() or "acc" in line.lower()):
        print(f"Final at line {i}: {line.rstrip()}")
