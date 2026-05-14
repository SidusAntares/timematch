import os
fpath = "C:/Code/dev/PythonProject/timematch/logs/v255b_source_weight_ablation_20260514_134715/gpu3_33UVP_to_32VNH_structure_light.log"
with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
    lines = fh.readlines()
print("Lines around kappa 83.4:")
for i in range(22820, min(22900, len(lines))):
    print(i, ":", lines[i].rstrip())
print()
print("Last 50 lines of gpu3:")
for line in lines[-50:]:
    print(line.rstrip())
