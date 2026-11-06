"""Add --unshared-enc: per-observation encoders with NO weight sharing.

Exact-match patch over model/oaa.py, train_oaa.py, core/ckpt.py.
Backs up each file to <file>.bak-pre-unsharedenc-20261106 before writing.
"""
import os
import shutil
import sys

ROOT = "/root/storage/implementation/shared_audio/hear360"
SUF = ".bak-pre-unsharedenc-20261106"

EDITS = {
    "model/oaa.py": [
        # 1) docstring: record the switch where the other construction-time switches are listed
        ("""Ablation switches (construction-time, default off => reference model):
  no_tf_pe, no_pose_emb, no_ray_emb, no_geo_bias, no_cross.""",
         """Ablation switches (construction-time, default off => reference model):
  no_tf_pe, no_pose_emb, no_ray_emb, no_geo_bias, no_cross, unshared_enc.
  unshared_enc drops the weight sharing stated above: every observation gets its own encoder of
  the same architecture, everything else (token grid, pose codes, fusion, input tensor) unchanged."""),
        # 2) constructor signature
        ("""                 geo_key_coupling=False, enc_time_stride1=False):""",
         """                 geo_key_coupling=False, enc_time_stride1=False, unshared_enc=False):"""),
        # 3) build the extra encoders
        ("""        self.enc = ViewEncoder(C, in_ch=in_ch, norm=norm, lh=lh, lw=lw, enc_res=enc_res, stem_stride1=stem_stride1,
                               time_stride1_stages=(2 if enc_time_stride1 else 0))""",
         """        self.enc = ViewEncoder(C, in_ch=in_ch, norm=norm, lh=lh, lw=lw, enc_res=enc_res, stem_stride1=stem_stride1,
                               time_stride1_stages=(2 if enc_time_stride1 else 0))
        # unshared_enc (2026-11-06): drop the weight sharing ACROSS observations. Each of the nviews
        # observations gets its own encoder of the SAME architecture (so this is a weight-sharing
        # ablation, not an architecture change). Observation 0 keeps the `enc.*` keys, and the extra
        # encoders are only created when the switch is on, so a reference checkpoint still loads with
        # strict=True while the switch is off. Token grid, pose codes, inter-observation attention and
        # the input tensor are untouched; the only difference is whose weights encode which observation.
        self.unshared_enc = bool(unshared_enc)
        self.enc_extra = nn.ModuleList(
            [ViewEncoder(C, in_ch=in_ch, norm=norm, lh=lh, lw=lw, enc_res=enc_res, stem_stride1=stem_stride1,
                         time_stride1_stages=(2 if enc_time_stride1 else 0))
             for _ in range(nviews - 1)]) if unshared_enc else None"""),
        # 4) per-observation encode
        ("""        v = spec.view(B, self.nv, self.in_ch, H, W).reshape(B * self.nv, self.in_ch, H, W)
        enc_t, fine_t = self.enc(v)
        t = enc_t.reshape(B, self.nv, self.M, self.C)
        if not self.no_tf_pe:
            t = t + self.tf_pe.unsqueeze(1)
        fine = fine_t.reshape(B, self.nv, 4 * self.M, self.enc.fine_ch)""",
         """        v = spec.view(B, self.nv, self.in_ch, H, W)
        if self.enc_extra is None:                                   # weight-shared: one batched call
            enc_t, fine_t = self.enc(v.reshape(B * self.nv, self.in_ch, H, W))
            t = enc_t.reshape(B, self.nv, self.M, self.C)
            fine = fine_t.reshape(B, self.nv, 4 * self.M, self.enc.fine_ch)
        else:                                                        # --unshared-enc: observation i -> encoder i
            outs = [self.enc_for(i)(v[:, i]) for i in range(self.nv)]
            t = torch.stack([o[0] for o in outs], 1)
            fine = torch.stack([o[1] for o in outs], 1)
        if not self.no_tf_pe:
            t = t + self.tf_pe.unsqueeze(1)"""),
        # 5) helper
        ("""    def _encode(self, spec, view_pose=None):""",
         """    def enc_for(self, i):
        \"\"\"The encoder that owns observation i: `enc` for i=0, `enc_extra[i-1]` with --unshared-enc.\"\"\"
        return self.enc if (i == 0 or self.enc_extra is None) else self.enc_extra[i - 1]

    def _encode(self, spec, view_pose=None):"""),
    ],
    "train_oaa.py": [
        ("""    p.add_argument("--stem-stride1", action="store_true")   # stride-1 stem on the native input (no 2x upsample)""",
         """    p.add_argument("--stem-stride1", action="store_true")   # stride-1 stem on the native input (no 2x upsample)
    p.add_argument("--unshared-enc", action="store_true")  # per-observation encoders: same architecture, NO weight sharing"""),
        ("""                       enc_time_stride1=a.enc_time_stride1,
                       in_ch=pairfeat.in_ch(a.pair_feat, a.nviews)).to(device)""",
         """                       enc_time_stride1=a.enc_time_stride1,
                       unshared_enc=a.unshared_enc,
                       in_ch=pairfeat.in_ch(a.pair_feat, a.nviews)).to(device)"""),
    ],
    "core/ckpt.py": [
        ("""                   no_cross=args.get("no_cross", False) or False,""",
         """                   no_cross=args.get("no_cross", False) or False,
                   # weight-sharing ablation (2026-11-06); absent in every earlier checkpoint and the
                   # default is the weight-shared reference encoder, so old args rebuild unchanged
                   unshared_enc=args.get("unshared_enc", False) or False,"""),
    ],
}


def main():
    for rel, edits in EDITS.items():
        path = os.path.join(ROOT, rel)
        src = open(path).read()
        for old, new in edits:
            n = src.count(old)
            if n != 1:
                sys.exit(f"ABORT {rel}: pattern count {n} != 1 for:\n{old[:120]}")
            src = src.replace(old, new)
        bak = path + SUF
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        open(path, "w").write(src)
        print(f"patched {rel} ({len(edits)} edits), backup {os.path.basename(bak)}")


if __name__ == "__main__":
    main()
