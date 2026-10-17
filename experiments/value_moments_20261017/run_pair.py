"""Replay a real control manifest as paired control/moment runs (dry-run by default).

This uses the trainer's own parser. It does not guess the current hop-16 campaign
recipe from the older configs/*_fin.json files. Results must stay outside the repo.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from train_oaa import make_parser

STFT = {"STFT_HOP": "stft_hop", "STFT_WIN": "stft_win",
        "STFT_NFFT": "stft_nfft", "STFT_WINDOW": "stft_window"}
GENERATED = {"full_res", "full_res_enc", "multi_scale_lift", "dec_deep", "rounds_wired",
             "cond_mode", "data_module", *STFT.values()}


def read_recipe(run_dir, baseline_world_size=None):
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    args = manifest["args"]
    dm = manifest.get("data_module") or args.get("data_module")
    if dm not in ("data_0422", "data_mp3d"):
        raise ValueError("A Replica or MP3D train_oaa manifest is required")
    if args.get("data_module", dm) != dm:
        raise ValueError("Conflicting data modules in args and manifest")
    world = manifest.get("ddp_world", baseline_world_size)
    if world is None:
        raise ValueError("Old manifest omits DDP world size; supply --baseline-world-size after checking the run")
    if baseline_world_size is not None and int(world) != baseline_world_size:
        raise ValueError("--baseline-world-size conflicts with the manifest")
    if int(world) != 1:
        raise ValueError("This launcher is single-GPU; do not change a DDP control's effective batch")
    if args.get("depth_loss", "plain") != "plain" or args.get("aux_l2", 0.) != 0.:
        raise ValueError("This experiment keeps the reference plain masked L1 loss")
    if (args.get("value_moments") or args.get("no_cross") or args.get("geo_value_film")
            or args.get("film_lse") or args.get("registers") or args.get("ray_gate")
            or args.get("view_fusion", "global") != "global"):
        raise ValueError("Use an unmodified global-softmax control, not an existing readout variant")
    if args.get("resume"):
        raise ValueError("Use a fresh-run control manifest; resumed histories need a separate audit")
    if args.get("rounds_wired", True) is not True:
        raise ValueError("Legacy unwired rounds/lift settings need an explicit recipe audit")
    for name in ("full_res_enc", "multi_scale_lift", "dec_deep"):
        if args.get(name, True) is not True:
            raise ValueError(f"Unsupported control setting {name}")
    if args.get("cond_mode", "adaln") != "adaln":
        raise ValueError("Only the released AdaLN architecture is supported")
    env = {"DATA_MODULE": dm}
    for name, key in STFT.items():
        recorded = (manifest.get("stft") or {}).get(name)
        value = args.get(key, recorded)
        if value is None:
            raise ValueError(f"Missing {name}; refusing to substitute a different input recipe")
        if recorded is not None and int(value) != int(recorded):
            raise ValueError(f"Conflicting {name} in args and manifest")
        env[name] = str(int(value))
    actions = {a.dest: a for a in make_parser()._actions if a.option_strings}
    unknown = set(args) - set(actions) - GENERATED
    if unknown:
        raise ValueError(f"Unrecognised control args require review: {sorted(unknown)}")
    return args, env, actions


def make_commands(run_dir, out_root, seeds, baseline_world_size=None):
    args, env, actions = read_recipe(run_dir, baseline_world_size)
    out_root = Path(out_root).resolve()
    if out_root.is_relative_to(REPO):
        raise ValueError("Write experiment outputs outside the repository")
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        raise ValueError("Launch this driver directly, not from torchrun")
    commands = []
    for seed in seeds:
        for arm in ("control", "moments"):
            name = f"{env['DATA_MODULE']}_n{args['nviews']}_{arm}_s{seed}"
            if (out_root / name).exists():
                raise FileExistsError(f"Refusing to overwrite {out_root / name}")
            values = dict(args, run_name=name, out_dir=str(out_root), seed=seed,
                          value_moments=arm == "moments", dump_attn_stats=True)
            cmd = [sys.executable, str(REPO / "train_oaa.py")]
            for key, action in actions.items():
                if key not in values or key == "help":
                    continue
                value = values[key]
                if isinstance(action, argparse._StoreTrueAction):
                    if value:
                        cmd.append(action.option_strings[0])
                elif value is not None:
                    cmd.extend([action.option_strings[0], str(value)])
            # Validate flags before spending GPU time. The horizon, batch/accumulation,
            # dropout, stem, and every other supported field come from this same manifest.
            make_parser().parse_args(cmd[2:])
            commands.append((name, cmd))
    return env, commands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--baseline-world-size", type=int,
                   help="required for old manifests which did not record DDP world size")
    p.add_argument("--execute", action="store_true", help="run sequentially; otherwise only print commands")
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds):
        p.error("seeds must be distinct")
    env, commands = make_commands(a.baseline_run, a.out_root, a.seeds, a.baseline_world_size)
    for name, cmd in commands:
        print(shlex.join(["env", *(f"{k}={v}" for k, v in env.items()), *cmd]), flush=True)
        if a.execute:
            (a.out_root / name).mkdir(parents=True, exist_ok=False)
            with (a.out_root / name / "launcher.log").open("w") as log:
                subprocess.run(cmd, env=dict(os.environ, **env), cwd=REPO,
                               stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
