#!/usr/bin/env python3
"""Appendix A: per-cell full-metric comparison table, generated from compare.json files.
Inserts (or replaces) the <!--APPENDIX_A--> ... <!--/APPENDIX_A--> block in both report HTMLs."""
import json, re

rep=json.load(open('../comparison_0820/compare.json')); rep.update(json.load(open('../comparison/compare.json')))
rep.update({'eco_'+k.split('eco_')[1]: v for k,v in json.load(open('../comparison/compare_eco.json')).items()} if True else {})
rep.update(json.load(open('../comparison/compare_eco.json')))
mp=json.load(open('../comparison_0820/mp3d_eval/compare.json')); mp.update(json.load(open('../comparison_mp3d/compare.json')))
mp.update(json.load(open('../comparison_mp3d/compare_eco.json')))
M=[('MAE','MAE'),('RMSE','RMSE'),('AbsRel','AbsRel'),('delta1','δ1'),('near<3','근<3m'),('mid3-6','중3–6m'),('far>6','원>6m')]
HIGHER={'delta1'}
CELLS=[
 ('Replica 2ch',rep,[('OAA-CNN','oaa_r2_fin'),('EchoScan','es_r2_fin'),('EchoDiffusion','eco_r2_fin'),('Beyond-I2D','0820_beyond_r2_rep'),('eat+LLRD','0820_eatllrd_r2_rep'),('eat+LLRD+cs','0820_eatllrd_cs_r2_rep'),('sslam','0820_sslam_r2_rep'),('sslam+LLRD','0820_sslam_llrd_r2_rep')]),
 ('Replica 4ch',rep,[('OAA-CNN','oaa_fb_fin'),('EchoScan','es_fb_fin'),('EchoDiffusion','eco_fb_fin'),('Beyond-I2D','0820_beyond_fb_rep'),('eat+LLRD','0820_eatllrd_fb_rep'),('sslam s0','0820_sslam_fb_rep'),('sslam s1','0820_sslam_s1_fb_rep'),('sslam s2','0820_sslam_s2_fb_rep'),('sslam+LLRD','0820_sslam_llrd_fb_rep')]),
 ('Replica 6ch',rep,[('OAA-CNN','oaa_r6_fin'),('EchoScan','es_r6_fin'),('EchoDiffusion','eco_r6_fin'),('eat+LLRD','0820_eatllrd_r6_rep'),('sslam s0','0820_sslam_r6_rep'),('sslam s1','0820_sslam_r6_rep_s1'),('sslam s2','0820_sslam_r6_rep_s2'),('sslam+LLRD','0820_sslam_llrd_r6_rep')]),
 ('Replica 8ch',rep,[('OAA-CNN','oaa_r8_fin'),('EchoScan','es_r8_fin'),('EchoDiffusion','eco_r8_fin'),('eat+LLRD vd','0820_eatllrd_r8_rep'),('sslam vd','0820_sslam_r8vd_rep'),('sslam+LLRD vd','0820_sslam_llrd_r8vd_rep')]),
 ('MP3D 2ch',mp,[('OAA-CNN','oaa_r2_fin'),('EchoScan','es_r2_fin'),('EchoDiffusion','eco_r2_fin'),('Beyond-I2D','0820_beyond_r2_rep'),('eat+LLRD','0820_eatllrd_r2_mp3d'),('eat+LLRD+cs','0820_eatllrd_cs_r2_mp3d'),('sslam','0820_sslam_r2_mp3d'),('sslam+LLRD','0820_sslam_llrd_r2_mp3d')]),
 ('MP3D 4ch',mp,[('OAA-CNN','oaa_fb_fin'),('EchoScan','es_fb_fin'),('EchoDiffusion','eco_fb_wstd'),('eat+LLRD s0','0820_eat_llrd_fb_mp3d'),('sslam','0820_sslam_fb_mp3d'),('sslam+LLRD65','0820_sslam_llrd65_fb_mp3d')]),
 ('MP3D 6ch',mp,[('OAA-CNN','oaa_r6_fin'),('EchoScan','es_r6_fin'),('EchoDiffusion','eco_r6_fin'),('eat+LLRD','0820_eatllrd_r6_mp3d'),('eat+LLRD+vd','0820_eatllrd_r6vd_mp3d'),('sslam+LLRD','0820_sslam_llrd_r6_mp3d')]),
 ('MP3D 8ch',mp,[('OAA-CNN','oaa_r8_fin'),('EchoScan','es_r8_fin'),('EchoDiffusion','eco_r8_fin'),('eat+LLRD novd s0','0820_eatllrd_r8novd_mp3d'),('eat+LLRD vd','0820_eatllrd_r8_mp3d')]),
]
rows=[]
rows.append('<!--APPENDIX_A-->')
rows.append('<h2>부록 A. 칸별 전 지표 비교표</h2>')
rows.append('<p>본문 표는 판정 지표(MAE)만 싣지만, MAE가 동률인 칸에서도 보조 지표는 체계적으로 갈린다. 아래 표는 여덟 평가 칸 각각에서 판정 대상 모델들의 전 지표를 같은 eval 파이프라인으로 정렬한 것이다(굵게 = 해당 칸·해당 지표 1위; δ1은 높을수록, 나머지는 낮을수록 좋음). 다시드 설정은 시드별 행을 그대로 실어 분산을 감추지 않았다. 표가 보여주는 반복 구조는 세 가지다. 첫째, 근거리(&lt;3m) 열은 거의 모든 칸에서 CNN이 1위이고, 원거리(&gt;6m) 열은 AFM이 일관되게 1위다 — MAE 동률의 내부는 "CNN=근거리 정밀, AFM=원거리·큰 오차 억제"의 교환이며, 이는 §6의 초기 국소 단서 대 분산 잔향 활용 그림과 같은 방향이다(Replica 8ch: far 1.395 vs 1.589, −12%). 둘째, 유일한 예외가 Replica 6ch 시드 0으로, 근거리까지 포함해 일곱 지표 전부에서 1위인 캠페인 유일 사례다. 셋째, EchoDiffusion은 전 칸에서 열세이고(초기 게시판에서 EchoScan 행이 EchoDiffusion으로 잘못 표기되어 있었음을 바로잡는다 — 두 모델을 이제 별도 행으로 싣는다), MP3D 6ch은 CNN이 MAE만이 아니라 지표 전반에서 방어하는 유일한 칸이다. MP3D 8ch의 vdrop 판(0.9858)은 전 지표 붕괴로, "vdrop은 MP3D에서 전면 기각" 법칙의 극단 사례다. eat+LLRD의 다시드 칸(MP3D 4/8ch)은 지면상 시드 0 행만 실었고 판정은 본문대로 시드 평균을 따른다.</p>')
for cell,src,models in CELLS:
    avail=[(l,k) for l,k in models if k in src]
    best={}
    for mk,_ in M:
        vals=[(src[k][mk],l) for l,k in avail]
        best[mk]=max(vals)[1] if mk in HIGHER else min(vals)[1]
    rows.append(f'<div class="tw"><table><caption style="text-align:left;font-weight:600;padding:.3rem 0">{cell}</caption>')
    rows.append('<tr><th>model</th>'+''.join(f'<th>{h}</th>' for _,h in M)+'</tr>')
    for l,k in avail:
        v=src[k]
        tds=''.join(f'<td class="w">{v[mk]:.4f}</td>' if best[mk]==l else f'<td>{v[mk]:.4f}</td>' for mk,_ in M)
        rows.append(f'<tr><td>{l}</td>{tds}</tr>')
    rows.append('</table></div>')
rows.append('<!--/APPENDIX_A-->')
block='\n'.join(rows)

for p in ['report.html','report_artifact.html']:
    s=open(p).read()
    if '<!--APPENDIX_A-->' in s:
        s=re.sub(r'<!--APPENDIX_A-->.*?<!--/APPENDIX_A-->', lambda m: block, s, flags=re.S)
    else:
        anchor='<p class="sub">재현: '
        assert anchor in s
        s=s.replace(anchor, block+'\n'+anchor)
    open(p,'w').write(s)
    print('inserted into',p)
