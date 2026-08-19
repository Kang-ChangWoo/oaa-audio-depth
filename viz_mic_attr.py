"""Per-ERP-region mic attribution for a trained OAA run (default: kany).

Three views, all saved under comparison/mic_attribution/:
  A. occlusion:  per-mic drop -> mean per-pixel |D_full - D_drop_i|   (behavioural, causal-ish)
  C. keep-only:  per-mic solo -> mean per-pixel |D_keep_i - gt|       (what one mic can support)
  B. attention:  RayMicAttn softmax weights per ERP grid ray per view (mechanistic)

Also prints a sanity table: each mic's yaw vs the azimuth peak of its occlusion/attention
map (+ flatness ratios) so we can judge whether the maps are meaningful at all.

  DATA_MODULE=data_0422 R0422_SPLIT=off3 EVAL_BS=6 CUDA_VISIBLE_DEVICES=4 \
    python3 viz_mic_attr.py --run-name oaa_r8_kany
"""
import os, json, math, argparse
import numpy as np
import torch

import eval as ev
from model.oaa import RayMicAttn, _yaw_rot_inv

OUT = "comparison/mic_attribution"


def patch_raymic(store):
    orig = RayMicAttn.forward

    def fwd(self, q_in, tokens, ray_dir3, poses, M):
        B, R, C = q_in.shape; N = len(poses)
        Q = self.q(self.nq(q_in)).view(B, R, self.h, self.dk).transpose(1, 2)
        tk = self.nk(tokens)
        K = self.k(tk).view(B, -1, self.h, self.dk).transpose(1, 2)
        V = self.v(tk).view(B, -1, self.h, self.dk).transpose(1, 2)
        logits = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)
        bias = []
        for yaw, ear in poses:
            local = _yaw_rot_inv(ray_dir3, yaw)
            a = torch.tensor([math.cos(yaw), 0.0, -math.sin(yaw)], device=ray_dir3.device)
            c = (ray_dir3 @ a).unsqueeze(-1) * ear
            e = torch.full_like(c, float(ear))
            bias.append(self.bias_mlp(torch.cat([local, c, e], -1)))
        bmic = torch.stack(bias, 1)
        bfull = bmic.unsqueeze(2).expand(R, N, M, self.h).reshape(R, N * M, self.h)
        logits = logits + bfull.permute(2, 0, 1).unsqueeze(0)
        att = logits.softmax(-1)
        if store.get("on"):
            pv = att.view(B, self.h, R, N, M).sum(-1).mean(1)          # (B,R,N) per-view mass
            store["sum"] = store.get("sum", 0) + pv.float().sum(0).cpu()
            store["n"] = store.get("n", 0) + B
        out = (att @ V).transpose(1, 2).reshape(B, R, C)
        h = q_in + self.o(out)
        return h + self.ffn(h)

    RayMicAttn.forward = fwd
    return orig


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="oaa_r8_kany")
    ap.add_argument("--ckpt", default="best")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda")
    rd = ev.resolve_run(a.run_name, ["out", "comparison"])
    ck = torch.load(os.path.join(rd, f"{a.ckpt}.pth"), map_location="cpu", weights_only=False)
    model, dmode, nch, kind, poses = ev.build(ck["args"])
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    md = ck["args"].get("max_depth", 10.0)
    N = nch
    store = {}
    patch_raymic(store)
    ld = ev.loader("test", int(os.environ.get("EVAL_BS", "6")), False, 5, dmode)
    H, W = 256, 512
    occ = torch.zeros(N, H, W); keep_err = torch.zeros(N, H, W); keep_cnt = torch.zeros(N, H, W)
    nimg = 0
    for b in ld:
        x0 = b["spec"][:, :N].to(device)
        gt = b["depth"].to(device) * md; mask = b["mask"].to(device)
        B = x0.shape[0]
        store["on"] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Df = model(x0, view_poses=poses).float() * md
        store["on"] = False
        for i in range(N):
            xd = x0.clone(); xd[:, i] = 0
            xk = torch.zeros_like(x0); xk[:, i] = x0[:, i]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Dd = model(xd, view_poses=poses).float() * md
                Dk = model(xk, view_poses=poses).float() * md
            occ[i] += (Df - Dd).abs().squeeze(1).sum(0).cpu()
            keep_err[i] += ((Dk - gt).abs() * mask).squeeze(1).sum(0).cpu()
            keep_cnt[i] += mask.squeeze(1).sum(0).cpu()
        nimg += B
    occ /= nimg
    keep = keep_err / keep_cnt.clamp(min=1)
    attn = (store["sum"] / store["n"]).T.numpy()                      # (N,R)
    lh = ck["args"].get("lift_h", 16); lw = ck["args"].get("lift_w", 32)
    attn = attn.reshape(N, lh, lw)
    np.save(f"{OUT}/occlusion_{a.run_name}.npy", occ.numpy())
    np.save(f"{OUT}/keeponly_{a.run_name}.npy", keep.numpy())
    np.save(f"{OUT}/attention_{a.run_name}.npy", attn)

    # ---- verification: per-mic azimuth localisation ----
    import importlib
    _DM = importlib.import_module(os.environ.get("DATA_MODULE", "data_0422"))
    P = poses or _DM.POSES[dmode]
    def circ_peak(m):                                   # azimuth (deg) of column-mass peak, ERP col->deg
        col = m.mean(0)                                 # (W,)
        th = np.linspace(0, 2 * np.pi, len(col), endpoint=False)
        z = (col * np.exp(1j * th)).sum() / max(col.sum(), 1e-9)
        return math.degrees(np.angle(z)) % 360, abs(z)
    rows = []
    for i, (yaw, ear) in enumerate(P):
        po, ro = circ_peak(occ[i].numpy())
        pa, ra = circ_peak(attn[i])
        rows.append((i, math.degrees(yaw) % 360, "R" if ear > 0 else "L", po, ro, pa, ra,
                     float(occ[i].max() / occ[i].mean()), float(attn[i].max() / attn[i].mean())))
    json.dump({"rows": rows, "note": "cols: idx, yaw_deg, ear, occl_peak_az, occl_conc, attn_peak_az, attn_conc, occl_flatness, attn_flatness"},
              open(f"{OUT}/verify_{a.run_name}.json", "w"), indent=2)
    print("idx yaw ear | occl_az conc | attn_az conc | flat(occ,attn)")
    for r in rows:
        print(f"{r[0]}  {r[1]:5.0f} {r[2]} | {r[3]:6.1f} {r[4]:.2f} | {r[5]:6.1f} {r[6]:.2f} | {r[7]:.1f},{r[8]:.1f}")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lbl = [f"{int(math.degrees(y))%360}°{'R' if e>0 else 'L'}" for y, e in P]
    for name, data, cm in [("occlusion", occ.numpy(), "magma"), ("keeponly", keep.numpy(), "viridis"),
                           ("attention", attn, "magma")]:
        fig, axes = plt.subplots(2, 4, figsize=(18, 5.5))
        for i, ax in enumerate(axes.flat):
            im = ax.imshow(data[i], cmap=cm, aspect="auto")
            ax.set_title(f"mic {i} ({lbl[i]})", fontsize=10); ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.03)
        fig.suptitle(f"{a.run_name} — {name}")
        fig.tight_layout(); fig.savefig(f"{OUT}/{name}_{a.run_name}.png", dpi=110); plt.close(fig)
    am = occ.numpy().argmax(0)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(am, cmap="tab10", vmin=0, vmax=9, aspect="auto")
    ax.set_title(f"{a.run_name} — per-pixel argmax mic (occlusion)")
    cb = fig.colorbar(im, ax=ax, ticks=range(N)); cb.ax.set_yticklabels(lbl)
    fig.tight_layout(); fig.savefig(f"{OUT}/occl_argmax_{a.run_name}.png", dpi=110); plt.close(fig)
    print(f"[saved] {OUT}/  (npy x3, png x4, verify json)", flush=True)


if __name__ == "__main__":
    main()
