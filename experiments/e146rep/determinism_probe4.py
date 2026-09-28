# E146REP: make torch NAME the nondeterministic ops it hits in OAAv2Depth fwd+bwd.
import os, sys, hashlib, warnings, torch
sys.path.insert(0, "/root/storage/e143_code")
sys.path.insert(0, "/root/storage/implementation/shared_audio/hear360")
from model.oaa import OAAv2Depth
dev = "cuda:0"
ck = torch.load("/root/local1/changwoo/e146rep/out/ctrl_s2_rep1/best.pth", map_location="cpu", weights_only=False)
a = ck["args"]; a = a if isinstance(a, dict) else vars(a)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(a["seed"])
m = OAAv2Depth(C=a["dim"], nviews=a["nviews"], in_ch=a["in_ch"], rounds=a["rounds"],
               lh=a["lift_h"], lw=a["lift_w"], stem_stride1=a["stem_stride1"],
               max_depth=a["max_depth"], no_pose_emb=a["no_pose_emb"], no_ray_emb=a["no_ray_emb"],
               no_geo_bias=a["no_geo_bias"], no_tf_pe=a["no_tf_pe"], no_cross=a["no_cross"]).to(dev)
torch.manual_seed(999)
spec = torch.randn(2, a["nviews"] * a["in_ch"], 256, 256, device=dev)
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter("always")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(spec)
    D = out[0] if isinstance(out, (tuple, list)) else out
    D.float().abs().mean().backward()
seen = []
for w in W:
    s = str(w.message).replace(chr(10), " ")
    if s not in seen:
        seen.append(s)
print("distinct warnings: %d" % len(seen))
for s in seen:
    print("  -", s[:300])
print("PROBE4_DONE")
