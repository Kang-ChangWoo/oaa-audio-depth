#!/usr/bin/env python3
"""Combined view of all real-data predictions in results/: per scene, models x datasets.
OAA = campaign AFM models (replica: sslam_llrd_r8vd / mp3d: eatllrd_r8novd);
bat = BatVision (bat_r8_fin); eco = EchoDiffusion (eco_r8_fin / eco_r8)."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(HERE,"results")
MODELS=[("oaa","OAA-AFM"),("bat","BatVision"),("eco","EchoDiffusion")]
DS=[("replica","Replica-trained"),("mp3d","MP3D-trained")]
for scene in ("room","corner"):
    fig,axes=plt.subplots(len(MODELS),2,figsize=(13,3.1*len(MODELS)))
    prof={}
    for i,(mk,mn) in enumerate(MODELS):
        for j,(dk,dn) in enumerate(DS):
            p=np.load(os.path.join(R,f"pred_depth_{scene}_{dk}_{mk}.npy"))
            prof[(mk,dk)]=p[p.shape[0]//2-8:p.shape[0]//2+8].mean(0)
            ax=axes[i,j]; im=ax.imshow(p,cmap="turbo",vmin=0,vmax=10,aspect="auto")
            ax.set_title(f"{mn} · {dn}  (med {np.median(p):.2f} m)",fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle(f"scene: {scene} — 8ch models, canonical mapping (frame defined up to 45° rot + mirror)")
    fig.tight_layout(); fig.savefig(os.path.join(R,f"summary_{scene}.png"),dpi=110)
    # horizon overlay
    fig2,ax=plt.subplots(figsize=(10,4)); az=np.linspace(0,360,512,endpoint=False)
    for (mk,dk),pr in prof.items(): ax.plot(az,pr,label=f"{mk}/{dk}",lw=1.4)
    ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("horizon depth (m)"); ax.grid(alpha=.3); ax.legend(fontsize=8,ncol=3)
    ax.set_title(f"{scene}: horizon profiles"); fig2.tight_layout(); fig2.savefig(os.path.join(R,f"summary_{scene}_horizon.png"),dpi=110)
    print(f"== {scene}")
    for (mk,dk),pr in prof.items():
        print(f"  {mk:4s}/{dk:7s} min {pr.min():.2f}@{az[pr.argmin()]:.0f}° max {pr.max():.2f}@{az[pr.argmax()]:.0f}° med {np.median(pr):.2f}")
    base=prof[("oaa","replica")]
    for k,pr in prof.items():
        if k!=("oaa","replica"): print(f"  corr(oaa/replica, {k[0]}/{k[1]}): r={np.corrcoef(base,pr)[0,1]:.3f}")
