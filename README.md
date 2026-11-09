# Contactless-to-Contact Fingerprint Recognition with CMRA

**Cross-Modal Relational Alignment (CMRA) with risk-controlled abstention**

Course project, Biometric Security — National Institute of Technology Warangal
Aryan Desai · Naman Sinha · Saurabh Bharrdwaj · Sayan Das · November 2026

This repository contains the full study: literature review, protocol, code, saved scores,
analysis, report and slides. The task is **contactless-to-contact** matching on the
HKPolyU Contactless 2D to Contact-based 2D Fingerprint Database, under an
identity-disjoint train/validation/test split.

The proposed method is **CMRA**, a reproducible, CPU-trainable pipeline:

1. **Fixed scattering front-end** — a 2-D scattering transform (Kymatio, J = 2) produces
   81 × 48 × 48 features without learning early convolutional filters.
2. **Shared embedding** — a small CNN trained with ArcFace identity supervision produces a
   128-D L2-normalised embedding (about 0.13 M parameters).
3. **Identity-balanced PK batches** — each batch contains P = 16 identities with K = 2
   contactless and K = 2 contact images per identity (64 images per step).
4. **Prototype consistency** — aligns the mean embedding of each identity across the two
   modalities.
5. **Relational consistency** — a temperature-scaled symmetric KL divergence between the
   P × P cross-modal similarity matrices, so both modalities agree on *who is close to whom*.
6. **Risk-controlled abstention** — at inference, comparisons are ranked by |score − τ| and
   low-confidence pairs can be rejected, trading coverage for lower error.

The VerifNet-style Siamese branch and score fusion are **baselines only** in this version.
They are not part of the proposed system.

Headline result (seed 7, 10,728 test pairs): **EER 16.80%**, **TAR@FAR=1% 29.55%**,
**TAR@FAR=0.1% 7.20%**. The best seed (8) reaches **16.06% / 30.93% / 11.09%**. The v1
three-branch quality-aware fusion remains slightly stronger in the low-FAR tail
(18.16% / 32.94% / 12.19%), so the honest claim is **comparable-or-better accuracy with a
simpler, genuinely different pipeline**, not a blowout.

---

## 1. Dataset and protocol

| Item | Value |
|---|---|
| Database | HKPolyU Contactless 2D to Contact-based 2D (Lin & Kumar, TIP 2018) |
| Fingers | 336 |
| Images | 2,976 contactless + 2,976 contact = 5,952 |
| Sessions | First session: 336 fingers; second session: 160 fingers |
| Impressions | 6 per finger per modality per session |
| Split | 201 train / 34 validation / 101 test fingers |
| Identity overlap | None (disjoint by finger, asserted in `src/make_pairs.py`) |

**Primary test set.** 10,728 pairs = 5,364 genuine + 5,364 impostor. All genuine pairs are
same-session. Impostor pairs are sampled by preferring the same session, falling back to the
other session when the sampled identity has no image there; this gives 4,446 same-session and
918 cross-session impostor pairs. This is the number reported everywhere unless stated otherwise.

**Supplementary cross-session set.** 48 test fingers appear in both sessions. For those
identities only: 3,456 genuine and 3,456 impostor cross-session pairs. The same-session
reference is recomputed on the same 48 identities so the comparison is like-for-like.

**Metrics.** EER (the point where false accept and false reject rates cross), TAR at
FAR = 1% and 0.1%, and 95% confidence intervals from 500 identity-cluster bootstrap
resamples (all pairs of a resampled identity stay together). Risk-controlled abstention is
summarised by error-versus-coverage curves and the area under the risk-coverage curve (AURC).

**Model selection.** The CMRA checkpoint is chosen by validation EER. The test set is scored
once after the checkpoint is frozen.

### Important caveat on comparisons

Published HKPolyU numbers (Ridgeformer, IMPOSE, Multi-Siamese, MANet, etc.) use different
splits, training sets and protocols. They are included in the report as context only. The
**primary comparison** is against the VerifNet-style baseline and the v1 fusion pipeline
reproduced here under the same identity-disjoint split.

---

## 2. Results

All values are percentages on the 10,728-pair test set.

### 2.1 CMRA ablation (seed 7)

| Method | EER | TAR@1% | TAR@0.1% |
|---|---:|---:|---:|
| ISE (v1 single embedding) | 19.56 | 28.24 | 4.05 |
| CMRA ArcFace + PK only | 19.41 | 24.33 | 4.34 |
| CMRA + prototype | 18.23 | 27.44 | 5.87 |
| CMRA + relational | 17.74 | 28.93 | 6.99 |
| **CMRA full (prototype + relational)** | **16.80** | **29.55** | **7.20** |

### 2.2 Comparison with the v1 pipeline

| System | EER | TAR@1% | TAR@0.1% |
|---|---:|---:|---:|
| Minutiae only | 39.29 | 6.41 | 1.96 |
| VerifNet-style DL (3-seed avg) | 36.11 | 3.56 | 0.15 |
| v1 fixed fusion DL+MM | 32.83 | 8.35 | 2.16 |
| ISE (v1 single embedding) | 19.56 | 28.24 | 4.05 |
| v1 best fusion (quality-aware LR) | 18.16 | 32.94 | 12.19 |
| **CMRA full (seed 7)** | **16.80** | **29.55** | **7.20** |
| **CMRA full (seed 8)** | **16.06** | **30.93** | **11.09** |

**Seed sensitivity.** Full CMRA across seeds 7, 8, 9 and 10 gives EERs of 16.80%, 16.06%,
20.15% and 18.96%. Seeds 7, 8 and 10 average 17.3% (seed 9 is an outlier), so the result is
reported as a range rather than a single cherry-picked value.

### 2.3 Cross-session study (48 identities)

| Method | Cross EER | Cross TAR@1% | Same EER | Same TAR@1% |
|---|---:|---:|---:|---:|
| **CMRA seed 8 (proposed)** | **14.89** | **32.81** | **15.79** | 25.64 |
| v1 quality-aware fusion | 18.50 | 27.66 | 16.26 | 30.35 |
| ISE (v1 baseline) | 18.89 | 31.57 | 17.43 | 28.36 |
| Minutiae | 37.66 | 9.90 | 37.72 | 6.86 |

### 2.4 Risk-controlled abstention (seed 8)

| Coverage | Error rate |
|---:|---:|
| 100% | 16.04 |
| 90% | 13.00 |
| 80% | 9.94 |
| 70% | 7.68 |
| 50% | 3.88 |

AURC is 0.056 for CMRA seed 8, versus 0.084 for ISE. The seed-7 curve is similar
(16.79% at 100% coverage, 5.69% at 50%, AURC 0.068).

**Attribution.** The consistency losses, not extra fusion, produce the gain. ArcFace on
identity-balanced PK batches is already at the ISE level (19.41% vs 19.56%); adding the two
consistency terms lowers EER to 16.80%. The relational term is the stronger single addition
at seed 7 (17.74% vs 18.23% for prototype-only).

---

## 3. Repository layout

```
fingerprint_project/
├─ src/
│  ├─ make_pairs.py          # identity-disjoint split + genuine/impostor pairs
│  ├─ preprocess.py          # flip, rotation, illumination correction, ridge-scale normalisation
│  ├─ cache_images.py        # preprocess all used images -> images.npy
│  ├─ cache_scat.py          # fixed scattering transform -> scat.npy
│  ├─ minutiae.py            # minutiae extraction and matching
│  ├─ run_minutiae.py        # minutiae scores for val/test
│  ├─ siamese.py             # Siamese embedding network + contrastive loss
│  ├─ train.py               # VerifNet-style baseline training
│  ├─ evaluate.py            # EER and TAR@FAR
│  ├─ ensemble.py, fuse.py   # baseline score averaging / fixed fusion
│  ├─ proposed/
│  │  ├─ cmra.py             # CMRA: arc / proto / rel / full modes
│  │  ├─ cmra_eval.py        # ablation, bootstrap CIs, paired test, DET + risk-coverage
│  │  ├─ cmra_cross_session.py  # cross-session CMRA scoring
│  │  ├─ ise.py              # v1 identity-aware scattering embedding (baseline)
│  │  ├─ quality.py          # per-image quality features (v1 baseline)
│  │  ├─ fusion.py           # v1 score fusion / calibration (baseline)
│  │  ├─ analysis.py         # v1 stratified and failure analysis
│  │  ├─ cross_session.py    # v1 cross-session evaluation
│  │  ├─ make_figures.py     # CMRA pipeline, training-curve and comparison figures
│  │  ├─ build_report.py     # DOCX report builder
│  │  └─ build_slides.py     # PPTX slide builder
│  └─ data/                  # dataset folders + cached arrays + pair lists
├─ results/                  # metrics, scores, figures, per-run checkpoints
├─ report/                   # DOCX/PDF report and Overleaf LaTeX source
├─ slides/                   # final PPTX
├─ requirements.txt
└─ README.md
```

`src/data` is large (about 13.4 GB including the database and cached arrays) and
machine-specific. It is **not committed to this repository**: the HKPolyU database is
provided for research use only and must be obtained separately. The cached arrays and
pair lists are rebuilt by steps 1–2 of the reproduction commands below.

The published VerifNet paper PDF and the `results/literature/` scrapes are also excluded
for copyright reasons; the report cites those works instead.

---

## 4. Environment

Verified on Windows 11 with:

- Python 3.13.12
- PyTorch 2.14.1 (CPU build, no CUDA required)
- NumPy 2.2.6, pandas 3.0.6, scikit-learn 1.9.1
- OpenCV 5.0.0, kymatio 0.3.0, scikit-image 0.26.0
- 16 logical CPU threads, 32 GB RAM

```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

The report and slide builders use Microsoft Word / PowerPoint COM automation on Windows
(`pywin32`, installed conditionally by `requirements.txt`). They can be skipped if only the
metrics are needed; the DOCX/PDF, Overleaf source and PPTX are already included.

---

## 5. Reproduction

Run everything from the project root. Steps 1–3 rebuild caches and are only needed if
`src/data/*.npy` are missing.

```powershell
# 1. Split and pair lists (deterministic, seed 42)
.\venv\Scripts\python src\make_pairs.py

# 2. Preprocess images and cache fixed scattering features
.\venv\Scripts\python src\cache_images.py
.\venv\Scripts\python src\cache_scat.py

# 3. Classical minutiae scores (validation + test)
.\venv\Scripts\python src\run_minutiae.py

# 4. VerifNet-style baseline, three seeds (no rotation augmentation)
.\venv\Scripts\python src\train.py base 1 0
.\venv\Scripts\python src\train.py base 2 0
.\venv\Scripts\python src\train.py base 3 0

# 5. v1 ISE baseline (used for context in the report)
.\venv\Scripts\python src\proposed\ise.py --seed 7 --epochs 30 --out ise_seed7

# 6. CMRA ablation (seed 7)
.\venv\Scripts\python src\proposed\cmra.py --mode arc  --seed 7 --out cmra_arc_s7
.\venv\Scripts\python src\proposed\cmra.py --mode proto --seed 7 --out cmra_proto_s7
.\venv\Scripts\python src\proposed\cmra.py --mode rel  --seed 7 --out cmra_rel_s7
.\venv\Scripts\python src\proposed\cmra.py --mode full --seed 7 --out cmra_full_s7

# 7. Seed-sensitivity runs of the full model
.\venv\Scripts\python src\proposed\cmra.py --mode full --seed 8  --out cmra_full_s8
.\venv\Scripts\python src\proposed\cmra.py --mode full --seed 9  --out cmra_full_s9
.\venv\Scripts\python src\proposed\cmra.py --mode full --seed 10 --out cmra_full_s10

# 8. Ablation table, bootstrap CIs, paired test, DET + risk-coverage figures
.\venv\Scripts\python src\proposed\cmra_eval.py

# 9. Supplementary cross-session evaluation
.\venv\Scripts\python src\proposed\cmra_cross_session.py --run cmra_full_s8

# 10. Report/slide figures, DOCX report, PPTX slides
.\venv\Scripts\python src\proposed\make_figures.py
.\venv\Scripts\python src\proposed\build_report.py
.\venv\Scripts\python src\proposed\build_slides.py
```

`build_report.py` writes the DOCX. To produce the PDF, open the DOCX in Word and export, or
use the same COM call used in this project:

```powershell
$docx = (Resolve-Path '.\report\Contactless_Fingerprint_Recognition_Report.docx').Path
$pdf  = "$PWD\report\Contactless_Fingerprint_Recognition_Report.pdf"
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$d = $w.Documents.Open($docx)
$d.SaveAs2($pdf, 17)   # 17 = wdFormatPDF
$d.Close($false); $w.Quit()
```

For Overleaf, upload `report/Contactless_Fingerprint_Recognition_Report_CMRA.tex` together
with the PNG files in `report/` and compile with pdfLaTeX.

---

## 6. Result provenance

Every number in the report and slides is read from, or computed directly from, these files.

| Claim | Source |
|---|---|
| CMRA ablation table, CIs | `results/cmra_metrics.csv` |
| Risk-coverage table and figure | `results/cmra_risk_coverage.csv`, `results/cmra_risk_coverage.png` |
| DET figure | `results/cmra_det.png` |
| Cross-session comparison | `results/cmra_cross_session.csv` |
| v1 context rows | `results/cmra_v1_context.csv` |
| Paired bootstrap / summary | `results/cmra_summary.json` |
| CMRA training history and test scores | `results/cmra_full_s{7,8,9,10}/summary.json` |
| Per-run checkpoints and scores | `results/cmra_*/model.pt`, `scores_val.csv`, `scores_test.csv` |
| Pipeline / training / comparison figures | `results/pipeline.png`, `training_curve.png`, `cmra_comparison.png` |
| v1 fusion metrics (baseline context) | `results/fusion_metrics.csv` |
| Full report | `report/Contactless_Fingerprint_Recognition_Report.docx` / `.pdf` |
| Overleaf source | `report/Contactless_Fingerprint_Recognition_Report_CMRA.tex` |
| Slides | `slides/Contactless_Fingerprint_Recognition_Slides.pptx` |

The report text quotes percentages to two decimals; values in the CSV are fractions
(e.g. `0.1680` = 16.80%).

---

## 7. Scope and limitations

- **One dataset.** All results are on HKPolyU CL2CB; no external validation set was used.
- **Same-session primary test.** The main genuine pairs are same-session because the second
  session covers only 160 of 336 fingers. The cross-session study is supplementary and
  restricted to 48 test identities.
- **Seed variance.** Full CMRA spans 16.06–20.15% EER across four seeds. The report quotes a
  range and a stable-seed mean rather than a single best run.
- **Low-FAR tail.** The v1 quality-aware fusion still has higher TAR@1% and TAR@0.1%.
  CMRA's advantage is a lower EER with a simpler single-embedding pipeline.
- **No state-of-the-art claim.** Published methods reaching 2–4% EER use transformers,
  external training data or generative augmentation. This project deliberately stays
  CPU-trainable and reproducible.
- **Five-paper related work.** The literature review is a focused set of the most relevant
  recent work, not an exhaustive survey.
- **Dataset terms.** HKPolyU data are for research and non-commercial use only; see
  `src/data/CrossFingerprint_Database_Readme.txt`. The dataset is not redistributed here.

---

## 8. Key references

1. C. Lin and A. Kumar, “Matching Contactless and Contact-based Conventional Fingerprint
   Images for Biometrics Identification,” *IEEE TIP*, vol. 27, no. 4, pp. 2008–2021, 2018.
2. H. Tan and A. Kumar, “Minutiae Attention Network with Reciprocal Distance Loss for
   Contactless to Contact-Based Fingerprint Identification,” *IEEE TIFS*, vol. 16, 2021.
3. S. Parasnis et al., “VerifNet — A Novel Score Fusion-Based Method Leveraging Wavelets
   With Deep Learning and Minutiae Matching for Contactless Fingerprint Recognition,”
   *IEEE TBIOM*, vol. 7, no. 3, pp. 344–353, 2025.
4. S. Pandey, B. Jawade and S. Setlur, “Ridgeformer: Multi-Stage Contrastive Training for
   Fine-grained Cross-Domain Fingerprint Recognition,” *IEEE ICIP*, 2025.
5. P. Kaplesh, A. Gupta, D. Bansal, S. Sofat and A. Mittal, “A systematic review of
   end-to-end framework for contactless fingerprint recognition,” *Engineering Applications
   of Artificial Intelligence*, 2025.
6. J. Deng, J. Guo, N. Xue and S. Zafeiriou, “ArcFace: Additive Angular Margin Loss for Deep
   Face Recognition,” *CVPR*, 2019.
7. M. Andreux et al., “Kymatio: Scattering Transforms in Python,” *JMLR*, vol. 21, no. 60,
   pp. 1–6, 2020.
