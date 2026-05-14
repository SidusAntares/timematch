import os
import re
import json
d=r'C:/Code/dev/PythonProject/timematch/logs/v254_metricbank_quickcheck_20260514_113252'
fs=sorted([f for f in os.listdir(d) if f.endswith(".log")])
for fname in fs:
    c=open(os.path.join(d,fname),"r",encoding="utf-8",errors="replace").read()
    ls=c.split("
")
    row={'file':fname}
    for l in ls:
        if len(l)>300 and 'source=' in l and 'target=' in l:
            m=re.search(r"source='([^']+)'",l)
            n=re.search(r"target='([^']+)'",l)
            if m: row["source"]=m.group(1)
            if n: row["target"]=n.group(1)
        if 'Selected checkpoint' in l and 'epoch_' in l:
            row['selected_epoch']=l.split('epoch_')[1].split('.pt')[0]
    idx=c.find(""best_weights_checkpoint"")
    if idx>0:
        start=c.rfind("{",0,idx)
        end_idx=c.find("Selected checkpoint",idx)
        if end_idx==-1: end_idx=len(c)
        while end_idx>start and c[end_idx-1] in " 	
": end_idx-=1
        try:
            data=json.loads(c[start:end_idx])
            row.update(data)
        except Exception as e:
            row["json_error"]=str(e)
    print(json.dumps(row,default=str,indent=2)[:800])
    print("---")
