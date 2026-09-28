# E146REP: why do same-seed same-node reruns diverge from epoch 0?
# Isolates candidate non-deterministic CUDA kernels in the OAA trainer path.
# No training, no checkpoint writes. ~30 s on one idle card.
import os, sys, hashlib, torch, torch.nn.functional as F
sys.path.insert(0, "/root/storage/e143_code")
sys.path.insert(0, "/root/storage/implementation/shared_audio/hear360")

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cudnn.benchmark    =", torch.backends.cudnn.benchmark)
print("cudnn.deterministic=", torch.backends.cudnn.deterministic)
print("det_algorithms     =", torch.are_deterministic_algorithms_enabled())
print("matmul.allow_tf32  =", torch.backends.cuda.matmul.allow_tf32)
print("cudnn.allow_tf32   =", torch.backends.cudnn.allow_tf32)
dev = "cuda:0"

def h(t):
    return hashlib.md5(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]

def rep(name, fn, n=3):
    hs = []
    for _ in range(n):
        torch.manual_seed(1234)
        hs.append(fn())
    same = len(set(hs)) == 1
    lab = "DETERMINISTIC" if same else "NON-DETERMINISTIC"
    print("  %-34s %s  %s" % (name, lab, hs))
    return same

print("\n== A. isolated ops, grad hash over 3 identical replays ==")

def t_interp():
    x = torch.randn(8, 64, 64, 128, device=dev, requires_grad=True)
    y = F.interpolate(x, size=(128, 256), mode="bilinear", align_corners=False)
    y.sum().backward()
    return h(x.grad)
rep("F.interpolate bilinear bwd", t_interp)

def t_convt():
    torch.manual_seed(7)
    m = torch.nn.ConvTranspose2d(64, 64, 4, 2, 1).to(dev)
    x = torch.randn(8, 64, 64, 128, device=dev, requires_grad=True)
    m(x).sum().backward()
    return h(x.grad) + "/" + h(m.weight.grad)
rep("ConvTranspose2d bwd", t_convt)

def t_conv():
    torch.manual_seed(7)
    m = torch.nn.Conv2d(64, 64, 3, 1, 1).to(dev)
    x = torch.randn(8, 64, 64, 128, device=dev, requires_grad=True)
    m(x).sum().backward()
    return h(x.grad) + "/" + h(m.weight.grad)
rep("Conv2d bwd (reference)", t_conv)

print("\n== B. full OAAv2Depth fwd+bwd under bf16 autocast, as the trainer runs it ==")
from model.oaa import OAAv2Depth
ck = torch.load("/root/local1/changwoo/e146rep/out/ctrl_s2_rep1/best.pth", map_location="cpu", weights_only=False)
a = ck["args"]
a = a if isinstance(a, dict) else vars(a)
print("  nviews=%s dim=%s rounds=%s lh=%s lw=%s in_ch=%s" % (a["nviews"], a["dim"], a["rounds"], a["lift_h"], a["lift_w"], a["in_ch"]))

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
try:
    rep("OAAv2Depth full bwd", t_model)
except Exception as e:
    print("  model probe failed:", type(e).__name__, e)

print("\n== C. torch.use_deterministic_algorithms(True): which op does torch itself name? ==")
torch.use_deterministic_algorithms(True, warn_only=False)
for name, fn in (("F.interpolate bilinear bwd", t_interp), ("ConvTranspose2d bwd", t_convt), ("OAAv2Depth full bwd", t_model)):
    try:
        fn()
        print(f"  {name:34s} OK under deterministic mode")
    except Exception as e:
        print(f"  {name:34s} RAISES {type(e).__name__}: {str(e)[:240]}")
print("\nPROBE_DONE")
