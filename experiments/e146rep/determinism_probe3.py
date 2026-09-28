# E146REP: isolate WHICH knob fixes OAAv2Depth. Probe2 showed env-var alone fails,
# env-var + use_deterministic_algorithms(True) succeeds. Bisect the knobs.
import os, sys, hashlib, torch
sys.path.insert(0, "/root/storage/e143_code")
sys.path.insert(0, "/root/storage/implementation/shared_audio/hear360")
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
def run(label):
    hs = [t_model() for _ in range(3)]
    print("  %-46s %s %s" % (label, "DETERMINISTIC" if len(set(hs)) == 1 else "NON-DETERMINISTIC", hs))
print("CUBLAS_WORKSPACE_CONFIG =", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
run("baseline (all knobs off)")
torch.backends.cudnn.deterministic = True
run("cudnn.deterministic=True only")
