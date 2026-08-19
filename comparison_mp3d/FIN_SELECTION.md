# MP3D fin 선택 (2026-07-27 확정)

논문 표(depth_comparison.tex) MP3D 행의 소스 런. eco fin = **wstd**(원전 기본 wave 설정;
wall/wnone은 nonselected/에 보존, ablation 각주용). Ours fin = **fullres 풀패키지 통일**(15.4M).

| 표 행 | fin 디렉토리 | 원래 run 이름 (compare.json에 구키 병존) | MAE |
|---|---|---|---|
| 2ch Ours | `oaa_r2_fin` | oaa_r2 (fullres-pkg, bs16, 40ep) | 0.9084 |
| 4ch Ours | `oaa_fb_fin` | oaa_fbm_fr_s0 (풀패키지 fb, eff32, 40ep) | 0.7849 |
| 6ch Ours | `oaa_r6_fin` | oaa_r6m_s1_s0 (풀패키지+stride1, eff16, 30ep) | 0.7502 |
| 8ch Ours | `oaa_r8_fin` **잠정** | oaa_r8m_s1kany_s0 (+kany vdrop) | 0.7467 |
| 2ch eco | `eco_r2_wstd` | — | 0.9007 |
| 4ch eco | `eco_fb_wstd` | — | 0.7928 |
| 6ch eco | `eco_r6` (wstd) | — | 0.7786 |
| 8ch eco | `eco_r8` (wstd) | — | 0.7572 |
| baselines | `{bat,vit,rn,es}_{r2,fb,r6,r8}_fin` | rn_fb/rn_r2/vit_r2는 무접미 런을 개명 | — |

- 8ch Ours는 학습 중인 no-drop(`oaa_r8m_s1_s0`/`_wd5e4_s0`)·짝수드롭(vd22c/vd42c)·yaw-flip
  결과에 따라 교체 예정. 교체 시: 새 런을 `oaa_r8_fin`으로 개명, kany는 nonselected/로.
- 구 대표(plain legacy: OAA_r6_adaln_s1 0.7294 등)는 연구 리포 `../out/`에 보존 — 표에서 제외됐을 뿐.
- 비선택 런 전부 `nonselected/` (35개), 옛 폐기물은 `deprecated/`.
