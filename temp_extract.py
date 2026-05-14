import os, re, json, glob
log_dir = "C:/Code/dev/PythonProject/timematch/logs/v255b_source_weight_ablation_20260514_134715"
files = sorted(glob.glob(os.path.join(log_dir, "*structure_light.log")))
print(f"Files found: {len(files)}")
for f in files:
    print(f"  {os.path.basename(f)}")