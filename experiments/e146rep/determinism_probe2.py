# E146REP follow-up: does CUBLAS_WORKSPACE_CONFIG alone make OAAv2Depth bit-reproducible?
# Probe 1 proved the full model bwd is NON-DETERMINISTIC while interpolate/ConvTranspose2d are
# deterministic, and torch under use_deterministic_algorithms(True) named cuBLAS. This pins it.
import os, sys, hashlib, torch
sys.path.insert(0, "/root/storage/e143_code")
sys.path.insert(0, "/root/storage/implementation/shared_audio/hear360")
print("CUBLAS_WORKSPACE_CONFIG =", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
from model.oaa import OAAv2Depth
dev = "cuda:0"
ck = torch.load("/root/local1/changwoo/e146rep/out/ctrl_s2_rep1/best.pth", map_location="cpu", weights_only=False)
a = ck["args"]; a = a if isinstance(a, dict) else vars(a)

def h(t): return hashlib.md5(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]

def t_model():
    torch.manual_seed(a["seed"])
    m = OAAv2Depth(C=a["dim"], nviews=a["nviews"], in_ch=a["in_ch"], rounds=a["rounds"],
                   lh=a["lift_h"], lw=a["lift_w"], stem_stride1=a["stem_stride1"],
                   max_depth=a["max_depth"], no_pose_emb=a["no_pose_emb"], no_ray_emb=a["no_ray_emb"],
                   no_geo_bias=a["no_geo_bias"], no_tf_pe=a["no_tf_pe"], no_cross=a["no_cross"]).to(dev)
    torch.manual_seed(999)
    spec = torch.randn(2, a["nviews"] * a["in_ch"], 256, 256, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(spec)
    D = out[0] if isinstance(out, (tuple, list)) else out
    D.float().abs().mean().backward()
    g = [p.grad for p in m.parameters() if p.grad is not None]
    return h(torch.cat([x.flatten() for x in g]))

hs = [t_model() for _ in range(3)]
print("  env-var only          ", "DETERMINISTIC" if len(set(hs)) == 1 else "NON-DETERMINISTIC", hs)

torch.use_deterministic_algorithms(True, warn_only=False)
try:
    hs2 = [t_model() for _ in range(3)]
    print("  env var + det_algos   ", "DETERMINISTIC" if len(set(hs2)) == 1 else "NON-DETERMINISTIC", hs2)
except Exception as e:
    print("  env var + det_algos    RAISES %s: %s" % (type(e).__name__, str(e)[:240]))
print("PROBE2_DONE")
