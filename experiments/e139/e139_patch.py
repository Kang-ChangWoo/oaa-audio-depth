"""E139: build oaa_e139.py from the repo's model/oaa.py by exact-string replacement.

Every edit is a literal find/replace that must match exactly once, so everything NOT listed here is
byte-identical to the repo file (md5 8addeb9d193d6eef7294925fd9e67c6d). Scope: give the ray queries a
distance coordinate. Decoder, ERP blocks, head, loss, data and recipe are untouched -- the K axis is
pooled away before the queries leave RayMicAttn, so every downstream shape is unchanged.

See PREREG_E139.md sec.0 for why "just add one input to bias_mlp" was impossible as written.
"""
import hashlib
import os
import sys

SRC = "/root/storage/implementation/shared_audio/hear360/model/oaa.py"
DST = "/root/storage/e139_code/oaa_e139.py"
SRC_MD5 = "8addeb9d193d6eef7294925fd9e67c6d"

HEADER = '''"""E139: model/oaa.py + a distance coordinate on the ray queries. Built by e139_patch.py.

RayMicAttn's bias came from direction only ([R^T r (3), ray.ear_axis (1), ear_sign (1)]), and the
query was self.q + dir_mlp(dir6) -- no distance anywhere, and ray_mic is called exactly once, so
there is no round-2 hook for a depth estimate either. E139 expands each ray query into K range bins
(a learned dist_emb added to the query), feeds the normalised bin centre as bias_mlp's 6th input,
and mean-pools over K on the way out so the decoder sees the same (B, R, C) it always did.

K = E139_K (default 4), bin centres = max_depth * (k + 0.5) / K.

Original docstring follows.
"""
'''

# ---- edit 1: bias_mlp takes one more input ----
OLD1 = "        self.bias_mlp = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, heads))"
NEW1 = "        self.bias_mlp = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, heads))"

# ---- edit 2: RayMicAttn.forward gets the K axis ----
OLD2 = '''    def forward(self, q_in, tokens, ray_dir3, poses, M):
        B, R, C = q_in.shape; N = len(poses)
        Q = self.q(self.nq(q_in)).view(B, R, self.h, self.dk).transpose(1, 2)
        tk = self.nk(tokens)
        K = self.k(tk).view(B, -1, self.h, self.dk).transpose(1, 2)
        V = self.v(tk).view(B, -1, self.h, self.dk).transpose(1, 2)
        logits = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)
        if self.use_bias:
            bias = []
            for yaw, ear in poses:
                local = _yaw_rot_inv(ray_dir3, yaw)
                a = torch.tensor([math.cos(yaw), 0.0, -math.sin(yaw)], device=ray_dir3.device)  # ear axis R(yaw)@x_hat
                c = (ray_dir3 @ a).unsqueeze(-1) * ear
                e = torch.full_like(c, float(ear))
                bias.append(self.bias_mlp(torch.cat([local, c, e], -1)))
            bmic = torch.stack(bias, 1)                                    # (R,N,h)
            bfull = bmic.unsqueeze(2).expand(R, N, M, self.h).reshape(R, N * M, self.h)
            logits = logits + bfull.permute(2, 0, 1).unsqueeze(0)
        out = (logits.softmax(-1) @ V).transpose(1, 2).reshape(B, R, C)
        h = q_in + self.o(out)
        return h + self.ffn(h)'''
NEW2 = '''    def forward(self, q_in, tokens, ray_dir3, poses, M, dist_n=None):
        """E139: q_in may be (B, R*Kb, C) with dist_n (Kb,) the normalised bin centres. The K axis is
        mean-pooled before the residual, so the return shape is (B, R, C) exactly as before."""
        B, RK, C = q_in.shape; N = len(poses)
        R = ray_dir3.size(0)
        Kb = RK // R
        assert R * Kb == RK, f"query count {RK} is not a multiple of {R} rays"
        if dist_n is None:
            assert Kb == 1, "dist_n is required when the query carries range bins"
            dist_n = q_in.new_zeros(1)
        Q = self.q(self.nq(q_in)).view(B, RK, self.h, self.dk).transpose(1, 2)
        tk = self.nk(tokens)
        K = self.k(tk).view(B, -1, self.h, self.dk).transpose(1, 2)
        V = self.v(tk).view(B, -1, self.h, self.dk).transpose(1, 2)
        logits = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)
        if self.use_bias:
            bias = []
            for yaw, ear in poses:
                local = _yaw_rot_inv(ray_dir3, yaw)
                a = torch.tensor([math.cos(yaw), 0.0, -math.sin(yaw)], device=ray_dir3.device)  # ear axis R(yaw)@x_hat
                c = (ray_dir3 @ a).unsqueeze(-1) * ear
                e = torch.full_like(c, float(ear))
                f5 = torch.cat([local, c, e], -1)                           # (R,5)
                f5 = f5.unsqueeze(1).expand(R, Kb, 5)                       # (R,Kb,5)
                d = dist_n.to(f5.dtype).view(1, Kb, 1).expand(R, Kb, 1)     # (R,Kb,1)
                bias.append(self.bias_mlp(torch.cat([f5, d], -1).reshape(RK, 6)))
            bmic = torch.stack(bias, 1)                                    # (R*Kb,N,h)
            bfull = bmic.unsqueeze(2).expand(RK, N, M, self.h).reshape(RK, N * M, self.h)
            logits = logits + bfull.permute(2, 0, 1).unsqueeze(0)
        out = (logits.softmax(-1) @ V).transpose(1, 2).reshape(B, RK, C)
        h = q_in + self.o(out)
        h = h + self.ffn(h)
        return h.view(B, R, Kb, C).mean(2)                                 # pool the range axis away'''

# ---- edit 3: OAAv2Depth builds the range embedding ----
OLD3 = "        self.ray_mic = RayMicAttn(C, use_bias=not no_geo_bias)"
NEW3 = '''        self.ray_mic = RayMicAttn(C, use_bias=not no_geo_bias)
        # E139: K range bins on the query side. dist_emb is the only new parameter block (K x C).
        self.e139_k = int(os.environ.get("E139_K", "4"))
        self.dist_emb = nn.Embedding(self.e139_k, C)
        nn.init.normal_(self.dist_emb.weight, std=0.02)
        self.register_buffer("e139_dist_n",
                            (torch.arange(self.e139_k, dtype=torch.float32) + 0.5) / self.e139_k,
                            persistent=False)'''

# ---- edit 4: forward feeds the expanded query ----
OLD4 = '''            q = (self.q if self.no_ray_emb else self.q + self.dir_mlp(dir6).unsqueeze(0)).expand(B, -1, -1)
            h = self.ray_mic(q, F4.reshape(B, self.nv * self.M, self.C), dir3, poses, self.M)'''
NEW4 = '''            q = (self.q if self.no_ray_emb else self.q + self.dir_mlp(dir6).unsqueeze(0)).expand(B, -1, -1)
            # E139: (B,R,C) -> (B,R*K,C) by adding the range embedding; RayMicAttn pools K away.
            Kb = self.e139_k
            q = (q.unsqueeze(2) + self.dist_emb.weight.view(1, 1, Kb, self.C)).reshape(B, self.M * Kb, self.C)
            h = self.ray_mic(q, F4.reshape(B, self.nv * self.M, self.C), dir3, poses, self.M,
                             dist_n=self.e139_dist_n)'''

EDITS = [(OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3), (OLD4, NEW4)]


def main():
    src = open(SRC).read()
    got = hashlib.md5(src.encode()).hexdigest()
    if got != SRC_MD5:
        sys.exit(f"ABORT: {SRC} md5 {got} != expected {SRC_MD5} -- the repo file changed, re-verify the edits")
    out = src
    for i, (old, new) in enumerate(EDITS, 1):
        n = out.count(old)
        if n != 1:
            sys.exit(f"ABORT: edit {i} matched {n} times, expected exactly 1")
        out = out.replace(old, new, 1)
    if "import os" not in out.split("\n\n")[0] and "\nimport os" not in out:
        out = out.replace("import math", "import math\nimport os", 1)
    assert "\nimport os" in out, "os import not injected"
    out = HEADER + out
    if os.path.exists(DST):
        os.replace(DST, DST + ".bak-pre-rebuild")
    open(DST, "w").write(out)
    print(f"wrote {DST}  ({len(out)} bytes, {len(EDITS)} edits applied)")


if __name__ == "__main__":
    main()
