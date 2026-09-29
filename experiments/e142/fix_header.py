#!/usr/bin/env python3
"""Rewrite REPORT_E142.md's headline verdict after the fixed-reference re-scoring."""
from pathlib import Path

R = Path("/root/local1/changwoo/e142/REPORT_E142.md")
txt = R.read_text()

OLD_H = ("# REPORT E142 — 회전이냐 이동이냐: **판정 A + P.** "
         "회전이 전체를 사고, **이동만 원거리를 산다.**")
NEW_H = """# REPORT E142 — 회전이냐 이동이냐: **판정 P 확정, A 철회.**
# **heading은 fidelity를 사고, 위치는 coverage를 산다. 원거리는 coverage로만 산다.**

> **2026-09-26 오후 갱신 — 머리글이 바뀌었다.** 아래 §1~§9는 **원본 프로토콜**(`e114_eval.py`,
> 참조 클라우드가 선택된 스텝에서 만들어짐) 결과이고 **그대로 보존한다**. §9가 남긴 참조 크기 교란을
> 같은 날 **고정 참조 재채점**(`e142_eval_fixedref.py`, 42칸, 2분 21초)으로 닫았고, 그 결과가
> 이 문서 끝의 **追記 §A~§N**이다. 판정이 두 갈래로 갈렸다:
>
> - **`>4 m`의 이동 우위 = SURVIVE.** `.0045` vs `.0524` (Δ `−.0479`, 시드 0/3, 짝 t = −5.60).
>   원본 `−.0463`과 사실상 동일하고, `FRAC .25/.50/1.00` 전부에서 부호가 유지된다.
>   → **"위치가 원거리를 산다"는 논문 주장으로 확정.**
> - **§4-1의 판정 A(전체 F1 회전 우위) = 철회.** 참조를 고정하면 `+.0228` → **`−.1051`**로 뒤집히고,
>   top-k 비율에 따라 부호가 또 바뀐다(`−.1051` / `−.0438` / `+.0085`).
>   → **"예산이 같으면 회전이 전체 F1에서 낫다"는 쓸 수 없다.** 두 배치는 순위가 아니라
>   **precision–recall 거래**다.
> - **판정 P는 살았고 더 날카로워졌다.** 같은 8개 위치 대조에서 회전은 세 FRAC 전부에서 이기고,
>   이득은 **전부 precision**이다 (ΔP `+.0407`, ΔR `−.0033`).
> - **예산 4배를 쓰는 곳**: heading에 쓰면 precision(`+.0407`), 위치에 쓰면 recall(`+.1695`).
>   overall에서 위치가 **9.7배**, `1.5-4 m`에서 **20배** 효율적이고, **`<0.5 m`만 heading이 이긴다**(0.76배).
>
> §4-1·§8·§7-8을 읽을 때는 반드시 追記 §F·§K·§L을 함께 볼 것. 세부 수치는 追記에 있다."""

assert txt.count(OLD_H) == 1, f"header match count = {txt.count(OLD_H)}"
txt = txt.replace(OLD_H, NEW_H)

# mark the two superseded claim blocks in place so nobody quotes them standalone
OLD_1 = """**관측 예산 8개를 쓸 수 있다면 한 자리에서 사방을 보는 게 낫다 — 전체 F1 `.6081` vs `.5853` (`+.0228`,
3/3 시드, 39 시퀀스 중 30개, 짝지은 t = 3.44). 단 `>4 m`은 정반대다: `.0052` vs `.0515` (`−.0463`,
0/3 시드, 11개 공통 시퀀스 중 회전이 이긴 건 1개, t = −3.77).**"""
NEW_1 = OLD_1 + """

> **[2026-09-26 갱신] 앞 문장(전체 F1)은 철회됐다** — 고정 참조에서 `−.1051`로 뒤집힌다(追記 §F-2).
> **뒷 문장(`>4 m`)은 살았다** — 고정 참조에서 `−.0479`(追記 §F-1). 갱신된 주장은 追記 §K."""
assert txt.count(OLD_1) == 1, "claim-1 match"
txt = txt.replace(OLD_1, NEW_1)

OLD_2 = "→ **판정 A.** (사후 변경 없음.)"
NEW_2 = ("→ **판정 A.** (사후 변경 없음.) "
         "**[2026-09-26 갱신] 이 판정은 고정 참조에서 성립하지 않는다 — 追記 §F-2에서 철회됐다.**")
assert txt.count(OLD_2) == 1, "claim-2 match"
txt = txt.replace(OLD_2, NEW_2)

R.write_text(txt)
print("header + 2 inline markers updated")
print(f"lines now: {len(txt.splitlines())}")
