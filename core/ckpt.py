"""Rebuild a model from the args saved in its checkpoint (`{"state_dict", "args"}`) and locate run dirs.

build(args, DM) -> (model, dmode, nch, kind, poses)
    kind  "spec" | "wave" (EchoScan consumes the raw waveform)
    dmode the data-module channel mode (r2 / cB / fb / r6 / r8 / ...), nch = channels the model expects
    poses OAA view poses [(yaw, ear)] for that mode (None for other models)
EchoDiffusion checkpoints are not handled here (isolated env; see eval_echodiffusion.py).
"""
import os
import torch

from model.oaa import OAAv2Depth
from model.batvision import RotDepth
from model import PretrainedResNet, PretrainedViT, BeyondI2DDepth, EchoScanDepth

_BASE_SPEC = {"resnet": PretrainedResNet, "vit": PretrainedViT, "beyond": BeyondI2DDepth}   # train_baseline.py ckpts
_NV2MODE = {2: "r2", 4: "cB", 6: "r6", 8: "r8"}   # fallback when no mode is stored


def build(args, DM):
    """Reconstruct the model from a checkpoint's saved args; DM = data module (core.data.get_data_module())."""
    IN_CH = DM.IN_CH
    poses_for = lambda md: getattr(DM, "POSES", {}).get(md)
    name = args.get("model")
    if name in _BASE_SPEC:
        mode = args.get("mode") or _NV2MODE[args.get("in_ch", 4)]
        in_ch = IN_CH[mode]
        m = _BASE_SPEC[name](in_ch=in_ch, pretrained=False) if name in ("resnet", "vit") \
            else BeyondI2DDepth(in_ch=in_ch, pretrained_material=False)
        return m, mode, in_ch, "spec", None
    if name == "echoscan":
        mode = args.get("mode", "r2")
        return EchoScanDepth(in_ch=IN_CH[mode], fs=args.get("fs", 48000)), mode, IN_CH[mode], "wave", None
    if "feat_c" in args or ("mode" in args and "model" not in args):         # batvision
        mode = args.get("mode", "cB")
        m = RotDepth(in_ch=IN_CH[mode], feat_c=args.get("feat_c", 32), ngf=args.get("ngf", 64))
        return m, mode, IN_CH[mode], "spec", None
    # ---- OAA
    nv = args.get("nviews", 4)
    dmode = args.get("data_mode") or _NV2MODE[nv]
    if args.get("data_module") and args["data_module"] != DM.__name__:        # checkpoints that recorded their dataset
        raise RuntimeError(f"checkpoint was trained with DATA_MODULE={args['data_module']} but {DM.__name__} is loaded")
    # The released model is the full-resolution multi-scale OAA with AdaLN conditioning; refuse checkpoints
    # trained with research-only options that this code base no longer implements.
    for k, want in (("cond_mode", "adaln"), ("full_res_enc", True), ("multi_scale_lift", True)):
        assert args.get(k, want) == want, f"unsupported checkpoint: {k}={args.get(k)}"
    for k in ("rope", "full_res_gated"):
        assert not args.get(k), f"unsupported checkpoint option: {k}"
    assert (args.get("ctx_mode") or "none") == "none", "unsupported checkpoint option: ctx_mode"
    # NOTE: checkpoints without `rounds_wired` were trained before rounds/lift reached the model; they are
    # rounds=2 / 16x32 regardless of the stored args (load_state_dict(strict) guards the mismatch).
    wired = args.get("rounds_wired", False)
    if args.get("audio_backbone") and args["audio_backbone"] != "cnn":     # 0820 AFM-encoder runs
        from model.audio_backbones_0820 import build_afm_model
        m = build_afm_model(args, pretrained=False)        # weights come from the checkpoint state_dict
        poses = poses_for(dmode)
        if args.get("yaw_flip") and poses:
            poses = [(-y, e) for (y, e) in poses]
        return m, dmode, nv, "spec", poses
    m = OAAv2Depth(C=args.get("dim", 256), nviews=nv, dec_deep=args.get("dec_deep", True),
                   stem_stride1=args.get("stem_stride1", False) or False, max_depth=args.get("max_depth", 10.0),
                   rounds=args.get("rounds", 2) if wired else 2,
                   lh=args.get("lift_h", 16) if wired else 16, lw=args.get("lift_w", 32) if wired else 32,
                   no_pose_emb=args.get("no_pose_emb", False) or False, no_ray_emb=args.get("no_ray_emb", False) or False,
                   no_geo_bias=args.get("no_geo_bias", False) or False, no_tf_pe=args.get("no_tf_pe", False) or False,
                   no_cross=args.get("no_cross", False) or False)
    poses = poses_for(dmode)
    if args.get("yaw_flip") and poses:
        poses = [(-y, e) for (y, e) in poses]
    if args.get("pose_blind"):
        poses = [(0.0, 1.0)] * nv
    elif args.get("ear_blind") and poses:
        poses = [(y, 1.0) for (y, _) in poses]
    return m, dmode, nv, "spec", poses


def resolve_run(run, search_dirs):
    """Find <dir>/<run>/ across search_dirs (e.g. ["out", "comparison"])."""
    for d in search_dirs:
        if os.path.isdir(os.path.join(d, run)):
            return os.path.join(d, run)
    raise FileNotFoundError(f"run '{run}' not found under {search_dirs}")


def load_run(run_dir, DM, ckpt="best", device="cpu"):
    """Load <run_dir>/<ckpt>.pth -> (model (eval mode, on device), dmode, nch, kind, poses, ck_dict)."""
    ck = torch.load(os.path.join(run_dir, f"{ckpt}.pth"), map_location="cpu", weights_only=False)
    model, dmode, nch, kind, poses = build(ck["args"], DM)
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    return model, dmode, nch, kind, poses, ck
