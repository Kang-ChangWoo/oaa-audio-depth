# Prior-work STFT hop survey (2026-09-20)

Why this exists: the campaign's hop 44 must be defensible without reference to our own score.
This collects what every relevant prior work actually used, with the exact source. Collected via
paper PDFs + official repos (line numbers quoted); nothing here is from this tree's baselines
(they all inherit data_0422's hop 160).

## The table

| Work | sr | n_fft | win | hop (samples) | hop (ms) | source quality |
|---|---:|---:|---:|---:|---:|---|
| VisualEchoes (ECCV'20) | 44.1k | 512 | 64 | **16 (explicit)** | 0.363 | paper Sec.4 states it verbatim — the ONLY echo-depth paper that prints its hop |
| Beyond-Image-to-Depth (CVPR'21) — Replica | 44.1k | 512 | 64 | 16 (librosa default win//4) | 0.363 | code `data_loader/audio_visual_dataset.py` L21-23 + declared shape [2,257,166] confirms |
| Beyond-Image-to-Depth — MP3D | 16k | 512 | 32 | 8 (librosa default) | 0.5 | code + declared shape [2,257,121] confirms (0.06s×16k=960 → 960/8+1=121) |
| **EchoDiffusion (Zhang et al., AAAI'25)** — Replica | 44.1k | 512 | 64 | 16 (librosa default) | 0.363 | repo zzu-yinjun/EchoDiffusion `dataloader/Beyound_Dataset.py` L53-55, L138-140; paper prints NO audio numbers |
| **EchoDiffusion** — MP3D | 16k | 512 | 32 | 8 (librosa default) | 0.5 | same file L96-98; pipeline byte-identical to Parida's |
| BatVision (ICRA'20) | 44.1k | 512 | 64 | 16 (INFERRED — paper omits hop, no public training code) | 0.363 | paper Sec.III-C gives n_fft 512 + win 64 only; cite as inferred |
| BatVision+GCC (2020) | 44.1k | — | — | — | — | GCC-PHAT time features, no own STFT; spectrogram baseline = ICRA'20 config |
| EchoScan (TASLP'24) | 8k | — | — | — | — | raw RIR waveform input, no STFT exists to cite |
| SoundSpaces nav recipe (ECCV'20, Sec.4.2) | 44.1k/16k | — | 512 | **160** | 3.63 / 10 | their own navigation spectrogram — the likely origin of our released hop 160 |
| VGGish/AudioSet | 16k | — | 25ms | 10ms | **10** | vggish_params.py |
| AST (Interspeech'21) | 16k | — | 25ms | 10ms | **10** | kaldi fbank frame_shift=10 |
| EAT (IJCAI'24) | 16k | — | 25ms | 10ms | **10** | feature_extract.py, same call |
| SSLAM (ICLR'25) | 16k | — | 25ms | 10ms | **10** | SSLAM_Inference/inference.py L93-94 — our AFM's native frontend |
| librosa / torch.stft default | — | — | — | win//4, floor(n_fft/4) | — | library docs |
| **ours (campaign)** | 48k | 512 | 400 | **44** | **0.92** | this tree |
| **ours (released recipe)** | 48k | 512 | 400 | 160 | 3.33 | data_0422 default — matches SoundSpaces' nav hop-160 convention |

## What it means for the hop-44 defense

1. **The echo-depth lineage standard is hop 0.36–0.5 ms** (VisualEchoes → Parida → EchoDiffusion all
   share one pipeline; librosa win//4 default). Our 0.92 ms is ~2x COARSER than the lineage, not
   suspiciously fine. The unconventional element of our recipe vs that lineage is the long window
   (400 vs 32/64 samples) — which is exactly the AFM-frontend convention (25 ms win, SSLAM/EAT/AST).
2. So the recipe is honestly described as: **AFM-lineage window x echo-depth-lineage dense hop**,
   with hop 0.92 ms sitting between the two lineage standards (0.36 ms << 0.92 ms << 10 ms).
3. The released hop 160 has its own provenance: it is SoundSpaces' *navigation* spectrogram hop —
   a recipe designed for continuous ambient audio, not for a 58 ms echo snippet.
4. EchoDiffusion's paper prints no audio parameters at all; its numbers are nailed by the official
   repo code (two identical author copies). BatVision's hop is inferred (flag in any citation).
5. EchoScan must not appear in an STFT-hop table (raw-waveform model).

Full source list (arXiv ids, repo URLs, line numbers) in the session research log; key ones:
arXiv:1912.07011 (BatVision), arXiv:2005.01616/ECCV (VisualEchoes, Sec.4), krantiparida/beyond-image-to-depth,
AAAI 10.1609/aaai.v39i21.34416 + zzu-yinjun/EchoDiffusion (Zhang et al.), arXiv:2310.11728 (EchoScan),
arXiv:1912.11474 (SoundSpaces), ta012/SSLAM.
