#!/usr/bin/env python3
"""Content hash of an OAA checkpoint, for E145's PREREG s4 integrity check.

Why not sha256 of the file: `torch.save` names the zip container after the OUTPUT FILENAME, so
best.pth holds entries `best.pth/data.pkl ...` while best_own.pth holds `best_own.pth/data.pkl ...`.
Two checkpoints with byte-identical *contents* therefore always differ as files. Verified on the
E145 smoke: 304/304 tensors equal, `args` equal, file sha256 different.

So the integrity check hashes the CONTENT: every state_dict key in sorted order, its dtype, shape and
raw values, plus the recipe args. Deterministic and filename-independent.

usage: ckpt_content_hash.py <ckpt.pth> [more.pth ...]
"""
import hashlib
import json
import sys

import torch


def tensor_bytes(t):
    t = t.detach().cpu().contiguous().flatten()
    try:
        return t.numpy().tobytes()
    except TypeError:                      # bfloat16 / float16 edge cases have no numpy dtype
        return t.float().numpy().tobytes()


def content_hash(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k]
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(tensor_bytes(t))
    h.update(json.dumps(ck.get("args", {}), sort_keys=True, default=str).encode())
    return h.hexdigest(), len(sd)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d, n = content_hash(p)
        print(f"{d} {n} {p}")
