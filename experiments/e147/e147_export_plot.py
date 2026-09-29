import json, statistics
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0,"/root/storage/e147_code")
EVAL = Path("/root/local1/changwoo/e142fix/eval_results")
SEEDS=("0","1","2")
BINS=("overall","<0.5","0.5-1.5","1.5-4",">4")
def load(c):
    f=EVAL/(c+".json")
    return json.loads(f.read_text()) if f.exists() else None
base=load("r2_N0_s0")
T={s:m["_ref"]["n_cam_all"] for s,m in base["per_seq"].items()}
coh={"L24":sorted(s for s in T if T[s]>=24),"L32":sorted(s for s in T if T[s]>=32)}
def sv(m,b,stat):
    x=m[b][stat]
    if x is None: return None
    if not np.isnan(x): return float(x)
    if m[b]["n_ref"]>0 and m[b]["n_pred"]==0 and stat in ("f1","recall"): return 0.0
    return None
out={"cohort_n":{k:len(v) for k,v in coh.items()},"data":{}}
for mode in ("r2","r8"):
  for lname,ns in (("L24",[1,2,4,8,16,24]),("L32",[1,2,4,8,16,32])):
    C=coh[lname]
    d={"N":ns}
    for b in BINS:
      for stat in ("f1","precision","recall"):
        mus=[];sds=[]
        for N in ns:
          per=[]
          for sd in SEEDS:
            cd=load(f"{mode}_N{N}_s{sd}")
            vals=[sv(cd["per_seq"][s],b,stat) for s in C if s in cd["per_seq"]]
            vals=[v for v in vals if v is not None]
            if vals: per.append(float(np.mean(vals)))
          mus.append(float(np.mean(per)) if per else None)
          sds.append(float(statistics.pstdev(per)) if len(per)>1 else 0.0)
        d[f"{b}|{stat}"]=mus; d[f"{b}|{stat}|sd"]=sds
    out["data"][f"{mode}_{lname}"]=d
Path("/root/local1/changwoo/e147/results/plotdata_e147.json").write_text(json.dumps(out,indent=1))
print("ok", out["cohort_n"])
