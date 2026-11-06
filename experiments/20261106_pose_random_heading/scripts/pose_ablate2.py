"""인퍼런스에서만 포즈를 바꿔 채점한다. 학습 0.
pose_ablate.py 의 확장판: MetricAccumulator 가 내는 지표를 전부 저장하고,
yaw 섞기를 순열 시드 3개로 돌리고, 전역 회전(yawroll) 대조군을 넣는다.

모드
  true      원본 포즈
  swap      귀 부호 뒤집기
  blind     모든 포즈 동일 (방향·귀 정보 제거)
  earblind  귀 부호만 제거, yaw 유지
  randyaw   yaw 값을 채널 간 무작위 순열 (seed 1234)
  randyaw2  같음 (seed 7)
  randyaw3  같음 (seed 99)
  yawroll   모든 yaw 를 +pi/2 로 일관되게 회전 (상대 기하 보존, 절대 정합만 깨뜨림)
  earleft   모든 귀 부호 = -1 (왼귀). earblind 의 전부 +1 과 짝이 되는 상수 배정
  randear   귀 부호를 채널 간 무작위 순열 (yaw 유지). 귀 방향은 헤딩 +-90 도이므로
  randear2  이것이 "귀 방향이 90/270 로 제대로 붙지 않고 섞인" 경우다 (seed 3 개)
  randear3

사용: python pose_ablate2.py <gibson|replica|mp3d> <run_dir,run_dir,...> <out.json>
"""
import os, sys, json, math, random
sys.path.insert(0, "/root/storage/implementation/shared_audio/hear360")
os.chdir("/root/storage/implementation/shared_audio/hear360")
DS = sys.argv[1]; RUNS = sys.argv[2].split(","); OUT = sys.argv[3]
os.environ["DATA_MODULE"] = {"gibson": "data_gibson", "replica": "data_0422", "mp3d": "data_mp3d"}[DS]
os.environ.setdefault("GIBSON_ROOT", "/root/local1/gibson_hear360")
os.environ.setdefault("REPLICA_ROOT", "/root/local1/changwoo/replica_0422_lite")
os.environ.setdefault("MP3D_ROOT", "/root/local1/changwoo/matterport3d_0303renew")
os.environ["STFT_NFFT"], os.environ["STFT_WIN"], os.environ["STFT_HOP"] = "512", "400", "160"
os.environ.setdefault("EVAL_BS", "4")
import torch
from core.data import get_data_module
from core.ckpt import load_run
from core.metrics import MetricAccumulator
from core import pairfeat
DM = get_data_module()

# 데이터셋 루트가 없으면 evaluate 가 에러 없이 전 지표 0.0 을 돌려준다 -> 먼저 막는다
_root = {"gibson": "GIBSON_ROOT", "replica": "REPLICA_ROOT", "mp3d": "MP3D_ROOT"}[DS]
assert os.path.isdir(os.environ[_root]), f"{_root} 없음: {os.environ[_root]}"

MODES = tuple(os.environ.get(
    "ABL_MODES", "true,swap,blind,earblind,randyaw,randyaw2,randyaw3,yawroll").split(","))
# 시드는 fb(서로 다른 yaw 2개)·r8(4개) 둘 다에서 서로 다른 비항등 배열이 나오도록 골랐다.
# 1234/7/99 는 fb 에서 7과 99가 같은 배열을 줘서 버렸다.
SEEDS = {"randyaw": 1, "randyaw2": 2, "randyaw3": 6}
# 귀 부호 순열용 시드. 2관측에서는 2원소 순열이라 항등 또는 swap 뿐이고 (_identity_noop 로 표시),
# 4·8관측에서 비로소 swap 과 다른 "섞인" 배정이 나온다.
EAR_SEEDS = {"randear": 11, "randear2": 23, "randear3": 41}
# 8 개 방향을 무작위 순서로 배정: 짝의 두 귀가 서로 무관한 yaw 를 받는다
PERM_SEEDS = {"spreadperm": 5, "spreadperm2": 17, "spreadpermear": 5, "spreadpermear2": 17}


def perturb(P, mode):
    """원본 포즈 리스트 P=[(yaw, ear_sign), ...] 를 모드에 따라 바꾼다."""
    if mode == "true":
        return list(P)
    if mode == "swap":
        return [(y, -e) for (y, e) in P]
    if mode == "blind":
        return [(0.0, 1.0)] * len(P)
    if mode == "earblind":
        return [(y, 1.0) for (y, _) in P]
    if mode == "yawblind":
        # 헤딩만 제거한다. 모든 yaw 를 0 으로 눕히고 귀 부호는 원본 그대로 둔다.
        # blind 는 yaw 와 귀를 같이 지우므로 "헤딩만" 의 대조군이 못 된다.
        return [(0.0, e) for (_, e) in P]
    if mode.startswith("yawrot"):
        deg = float(mode[len("yawrot"):])          # global rotation by `deg`, pair structure intact
        return [((y + math.radians(deg)) % (2 * math.pi), e) for (y, e) in P]
    if mode == "yawroll":
        return [((y + math.pi / 2) % (2 * math.pi), e) for (y, e) in P]
    if mode == "earleft":
        return [(y, -1.0) for (y, _) in P]
    if mode in SEEDS:
        rnd = random.Random(SEEDS[mode])
        ys = [y for y, _ in P]
        rnd.shuffle(ys)
        return [(ys[i], P[i][1]) for i in range(len(P))]
    if mode in EAR_SEEDS:
        rnd = random.Random(EAR_SEEDS[mode])
        es0 = [e for _, e in P]
        es = list(es0)
        for _ in range(100):                      # 항등 배열은 버린다 (무효 칸이 되므로)
            rnd.shuffle(es)
            if es != es0:
                break
        return [(P[i][0], es[i]) for i in range(len(P))]
    if mode in ("spread", "spreadear", "spreadoff", "spreadoffear"):
        n = len(P)
        half = 0.5 if mode.startswith("spreadoff") else 0.0      # half-step grid agrees with no true heading
        ys = [2 * math.pi * (i + half) / n for i in range(n)]
        keep_ear = mode in ("spreadear", "spreadoffear")
        return [(ys[i], (P[i][1] if keep_ear else 1.0)) for i in range(n)]
    if mode in PERM_SEEDS:
        n = len(P)
        ys = [2 * math.pi * (i + 0.5) / n for i in range(n)]   # half-step grid: agrees with no true heading
        rnd = random.Random(PERM_SEEDS[mode])
        for _ in range(100):                                   # reject the increasing order (that is `spreadoff`)
            rnd.shuffle(ys)
            if ys != sorted(ys):
                break
        keep_ear = "ear" in mode
        return [(ys[i], (P[i][1] if keep_ear else 1.0)) for i in range(n)]
    if mode == "pairbreak":
        n = len(P)
        ys = [P[(i + 1) % n][0] for i in range(n)]
        return [(ys[i], P[i][1]) for i in range(n)]
    if mode.startswith("earflip"):
        k = int(mode[len("earflip"):])            # 뒤쪽 헤딩 k 개의 귀 부호만 뒤집는다
        n = len(P)
        assert n % 2 == 0, f"earflip needs (heading, L/R) pairs, got {n} channels"
        first = max(0, n - 2 * k)
        return [(y, (e if i < first else -e)) for i, (y, e) in enumerate(P)]
    raise ValueError(mode)


@torch.no_grad()
def run(run_dir, mode, device="cuda"):
    model, dmode, nch, kind, poses, ck = load_run(run_dir, DM, "best", device)
    md = ck["args"].get("max_depth", 10.0)
    pf = (ck["args"].get("pair_feat") or "none") if kind == "spec" else "none"
    pf_chans = list(DM._CH[dmode][:nch]) if pf != "none" else None
    if hasattr(DM, "EMIT_IPD"):
        DM.EMIT_IPD = pairfeat.needs_ipd(pf)
    P0 = list(poses)
    P = perturb(P0, mode)
    noop = (P == P0) and mode != "true"
    bs = int(os.environ["EVAL_BS"])
    ld = DM.loader("test", bs, False, 5, dmode)
    acc = MetricAccumulator(device)
    for b in ld:
        x = b["spec"][:, :nch].to(device)
        if pf != "none":
            x = pairfeat.build(x, pf_chans, pf, ipd=(b["ipd"][:, :nch].to(device) if "ipd" in b else None))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            D = (model(x, view_poses=P) if P is not None else model(x)).float() * md
        acc.update(D, b["depth"].to(device) * md, b["mask"].to(device))
    m = acc.result()
    del model
    torch.cuda.empty_cache()
    return {k: float(v) for k, v in m.items()}, P, P0, noop, dmode, nch


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for rd in RUNS:
    nm = os.path.basename(rd.rstrip("/"))
    res.setdefault(nm, {})
    for mode in MODES:
        if mode in res[nm]:
            continue
        try:
            m, P, P0, noop, dmode, nch = run(rd, mode)
            rec = {k: round(v, 6) for k, v in m.items()}
            rec["_poses"] = [[round(y, 4), e] for y, e in P]
            rec["_poses_true"] = [[round(y, 4), e] for y, e in P0]
            rec["_identity_noop"] = bool(noop)
            rec["_mode_key"] = dmode
            rec["_nviews"] = nch
            rec["_run_dir"] = rd
            res[nm][mode] = rec
            flag = "  [항등 순열 - 무효]" if noop else ""
            print(f"{nm:28s} {mode:9s} RMSE {m['RMSE']:.4f}  AbsRel {m['AbsRel']:.5f}  "
                  f"d1 {m['delta1']:.4f}{flag}", flush=True)
        except Exception as e:
            print(f"FAIL {nm} {mode} {type(e).__name__} {str(e)[:110]}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
print("[done]", flush=True)
