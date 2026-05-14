import os, re
dir_path = os.path.normpath(r"C:/Code/dev/PythonProject/timematch/logs/v254_metricbank_quickcheck_20260514_113252")
files = sorted([f for f in os.listdir(dir_path) if f.endswith(".log")])
for fname in files:
    fp = os.path.join(dir_path, fname)
    content = open(fp, "r", encoding="utf-8", errors="replace").read()
    lines = content.split("
")
    print(f"=== {fname} ===")
    for l in lines:
        if "Val" in l and ("acc" in l.lower() or "loss" in l.lower()):
            print(l[:300])
    print()
