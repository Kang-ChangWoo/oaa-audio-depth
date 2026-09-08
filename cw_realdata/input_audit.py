#!/usr/bin/env python3
"""Sim-to-real input audit: compare the real recordings (after our mapping) against the
training IR distribution on every axis the model could care about.

Checks:
 1. direct-pulse width (training = renderer band-limited spike ~1.5 ms)
 2. tail/direct energy ratio (G-experiment: echo-vs-direct amplitude is a key cue)
 3. pre-direct noise floor (training = digital zero before sample 49)
 4. spectral band occupancy (speaker/mic band-limit vs full-band sim impulse)
 5. intra-pair timing: training has direct at exactly 49 in BOTH ears; real stereo pairs
    (the batvision_XX_YY dirs show mics 180 deg apart were recorded as synced stereo) —
    our per-channel alignment forces both to 49, matching the training contract, but we
    report the raw offsets it discards.
Writes results/input_audit.png + prints a table.
"""
import os, sys, wave, glob
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE))
from scipy.signal import resample_poly

def loadw(fp):
    w=wave.open(fp); return np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768.0

def stats(x48, name):
    pk=int(np.argmax(np.abs(x48))); a=np.abs(x48)
    # width: samples around peak above 20% of peak
    th=0.2*a[pk]; l=pk; r=pk
    while l>0 and a[l]>th: l-=1
    while r<len(a)-1 and a[r]>th: r+=1
    width_ms=(r-l)/48.0
    d0,d1=max(0,pk-24),pk+48                      # ~1.5ms direct window
    e_direct=float((x48[d0:d1]**2).sum())
    e_tail=float((x48[d1:d1+2700]**2).sum())
    noise=float(np.sqrt((x48[:pk-24]**2).mean())) if pk>48 else float('nan')
    return dict(name=name,peak=float(a[pk]),width_ms=width_ms,tail_over_direct=e_tail/max(e_direct,1e-12),
                noise_floor=noise, spec=np.abs(np.fft.rfft(x48[d0:d0+2048],2048)))

rows_real=[]
for scene in ("room","corner","hall"):
    for fp in sorted(glob.glob(os.path.join(HERE,"data",scene,"clipped_audio","*.wav"))):
        x=resample_poly(loadw(fp),160,147).astype(np.float32)
        rows_real.append(stats(x, f"{scene}/{os.path.basename(fp).split('_')[1]}"))

import soundfile as sf
rows_sim=[]
for fp in sorted(glob.glob("/root/local2/replica_0422_lite/apartment_0/audio_wav/audio_0[0-3]*.wav"))[:20]:
    y,_=sf.read(fp,frames=2799,dtype="float32",always_2d=True)
    for c in range(2):
        if np.abs(y[:,c]).max()>0.5:                # only channels where direct survived pair-norm
            rows_sim.append(stats(y[:,c], os.path.basename(fp)))

def agg(rows,k):
    v=[r[k] for r in rows if np.isfinite(r[k])]
    return (np.mean(v),np.min(v),np.max(v)) if v else (float('nan'),)*3
print(f"{'':22s}{'REAL (24ch)':>28s}{'TRAIN sim (sample)':>28s}")
for k in ("peak","width_ms","tail_over_direct","noise_floor"):
    a=agg(rows_real,k); b=agg(rows_sim,k)
    print(f"{k:22s}{a[0]:9.3f} [{a[1]:.3f}..{a[2]:.3f}] {b[0]:12.3f} [{b[1]:.3f}..{b[2]:.3f}]")

# intra-pair raw offsets (before our per-channel alignment)
print("\nintra-pair direct-arrival offsets (samples @44.1k, before alignment):")
for scene in ("room","corner","hall"):
    for d in sorted(glob.glob(os.path.join(HERE,"data",scene,"batvision_*"))):
        pair=os.path.basename(d).replace("batvision_","").split("_")
        offs=[]
        for deg in pair:
            fp=os.path.join(HERE,"data",scene,"clipped_audio",f"mono_{deg}deg_clip_1s.wav")
            if os.path.exists(fp): offs.append(int(np.argmax(np.abs(loadw(fp)))))
        if len(offs)==2: print(f"  {scene} {pair[0]}/{pair[1]}: {offs[0]} vs {offs[1]} (d={offs[0]-offs[1]:+d} smp = {(offs[0]-offs[1])/44.1:+.2f} ms)")

# spectra figure
os.makedirs(os.path.join(HERE,"results","audit"),exist_ok=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
f=np.fft.rfftfreq(2048,1/48000)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for r in rows_real[:8]: axes[0].semilogy(f/1000, r["spec"]/r["spec"].max(), lw=.6, color="tab:blue", alpha=.5)
for r in rows_sim[:8]:  axes[0].semilogy(f/1000, r["spec"]/r["spec"].max(), lw=.6, color="tab:orange", alpha=.5)
axes[0].set_xlabel("kHz"); axes[0].set_title("direct-pulse spectrum (blue=real, orange=sim)"); axes[0].set_ylim(1e-4,1.2)
tor=[r["tail_over_direct"] for r in rows_real]; tos=[r["tail_over_direct"] for r in rows_sim]
axes[1].hist([tos,tor],bins=20,label=["sim","real"],density=True); axes[1].legend(); axes[1].set_xlabel("tail/direct energy ratio"); axes[1].set_title("reverb-to-direct energy")
fig.tight_layout(); fig.savefig(os.path.join(HERE,"results","audit","input_audit.png"),dpi=110)
print("\nsaved results/audit/input_audit.png")
