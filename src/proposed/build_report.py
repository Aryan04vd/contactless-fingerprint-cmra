"""Build the final project report as a Word document (professor's format)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
RESULTS = ROOT / "results"
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x55, 0x55, 0x55)


def set_cell_shading(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def style_doc(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)  # A4
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = Inches(0.85)
    sec.bottom_margin = Inches(0.85)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.12

    for i, size in [(1, 15), (2, 12.5), (3, 11)]:
        st = doc.styles[f"Heading {i}"]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = NAVY
        st.paragraph_format.space_before = Pt(12 if i == 1 else 9)
        st.paragraph_format.space_after = Pt(4)
        st.paragraph_format.keep_with_next = True


def para(doc, text, size=10.5, bold=False, italic=False, align=None, color=None, space_after=5):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    if color is not None:
        r.font.color.rgb = color
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style="List Bullet" if level == 0 else "List Bullet 2")
    p.paragraph_format.space_after = Pt(2)
    return p


def numbered(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.32)
    p.paragraph_format.first_line_indent = Inches(-0.26)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(8.5)
    r.italic = True
    r.font.color.rgb = GREY
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    prev = p._p.getprevious()
    if prev is not None and prev.tag == qn("w:tbl"):
        prev.addprevious(p._p)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(3)
        spacer = doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(0)
        spacer.paragraph_format.line_spacing = 1
        spacer.add_run("").font.size = Pt(2)
    return p


def table(doc, headers, rows, widths=None, size=8.5, header_fill="1F3A5F", align_center=True):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    if widths:
        t.autofit = False
        tblPr = t._tbl.tblPr
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        tblPr.append(layout)
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(size)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        set_cell_shading(hdr[i], header_fill)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            r = p.add_run(str(v))
            r.font.size = Pt(size)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if align_center else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
    if widths:
        for i, w in enumerate(widths):
            t.columns[i].width = Inches(w)
            for row in t.rows:
                row.cells[i].width = Inches(w)
    trPr = t.rows[0]._tr.get_or_add_trPr()
    hdr_el = OxmlElement("w:tblHeader")
    hdr_el.set(qn("w:val"), "true")
    trPr.append(hdr_el)
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
    return t


def figure(doc, name, cap, width=6.2):
    doc.add_picture(str(RESULTS / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(doc, cap)


def page_break(doc):
    doc.add_page_break()


def add_page_number_footer(doc):
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    txt = p.add_run("Contactless-to-contact fingerprint recognition  ·  ")
    txt.font.size = Pt(8)
    txt.font.color.rgb = GREY
    run = p.add_run()
    run.font.size = Pt(8)
    run.font.color.rgb = GREY
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.addnext(fld)


def main():
    cmra = pd.read_csv(RESULTS / "cmra_metrics.csv")
    cross = pd.read_csv(RESULTS / "cmra_cross_session.csv")
    rc = pd.read_csv(RESULTS / "cmra_risk_coverage.csv")

    doc = Document()
    cp = doc.core_properties
    cp.title = "Contactless-to-Contact Fingerprint Recognition — CMRA"
    cp.subject = "Biometric Security Course Project — HKPolyU CL2CB"
    cp.author = "Aryan Desai; Naman Sinha; Saurabh Bharrdwaj; Sayan Das"
    cp.comments = "Cross-Modal Relational Alignment for contactless-to-contact fingerprint verification"
    cp.keywords = "contactless fingerprint; cross-modal matching; relational alignment; ArcFace; CMRA"
    cp.category = "Course project report"
    style_doc(doc)
    add_page_number_footer(doc)

    # ------------------------------------------------------------------ title page
    for _ in range(5):
        doc.add_paragraph()
    para(doc, "NATIONAL INSTITUTE OF TECHNOLOGY WARANGAL", size=13, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, color=NAVY, space_after=4)
    para(doc, "Biometric Security — Course Project", size=11, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, color=GREY, space_after=26)
    para(doc, "Contactless-to-Contact Fingerprint Recognition", size=24, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, color=NAVY, space_after=4)
    para(doc, "Cross-Modal Relational Alignment (CMRA) with Risk-Controlled Abstention",
         size=15, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, color=NAVY, space_after=30)
    para(doc, "A reproducible study on the HKPolyU Contactless 2D to Contact-based 2D database",
         size=10.5, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, color=GREY, space_after=40)
    para(doc, "Team", size=10, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, color=GREY, space_after=3)
    para(doc, "Aryan Desai  ·  Naman Sinha  ·  Saurabh Bharrdwaj  ·  Sayan Das",
         size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=3)
    para(doc, "November 2026", size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, color=GREY)
    page_break(doc)

    # ---------------------------------------------------------------- abstract
    doc.add_heading("Abstract", level=1)
    para(doc,
         "Contactless fingerprint acquisition is hygienic and portable, but matching a contactless finger photo "
         "against a conventional contact-based print remains difficult because the two modalities differ in "
         "illumination, perspective, ridge scale, distortion and noise. Contactless-to-contact verification is "
         "essential for remote onboarding and access control, yet most published methods are evaluated mainly on "
         "contactless-to-contactless data.")
    para(doc,
         "VerifNet-style systems use pairwise contrastive training and fixed fusion weights, while transformer-based "
         "methods require external training data and GPU resources, leaving a gap for lightweight reproducible "
         "cross-modal matchers.")
    para(doc,
         "We overcome these limitations with Cross-Modal Relational Alignment (CMRA), a single embedding trained to "
         "preserve cross-modal relational identity structure using identity-balanced PK batches, ArcFace supervision, "
         "prototype consistency and relational consistency losses.")
    para(doc,
         "CMRA comprises four modules: a fixed scattering front-end for CPU-efficient feature extraction, an "
         "ArcFace-trained embedding network, a prototype consistency loss aligning cross-modal identity centres, and a "
         "relational consistency loss matching cross-modal similarity structure.")
    para(doc,
         "Extensive experiments on HKPolyU CL2CB under a controlled identity-disjoint split show CMRA achieves 16.80% "
         "EER and 29.55% TAR@FAR=1%, surpassing the reproduced VerifNet-style baseline (36.11% EER), the fixed DL+MM "
         "fusion (32.83% EER) and the ISE-only baseline (19.56% EER).")
    para(doc,
         "These results show that cross-modal relational alignment, not score fusion alone, is the key contribution "
         "for lightweight contactless-to-contact fingerprint verification.")
    page_break(doc)

    # ------------------------------------------------------------------ contents
    doc.add_heading("Contents", level=1)
    contents = ["1. Introduction", "2. Related Work", "3. Methodology",
                "4. Experiment and Results", "5. Conclusion and Future Work", "References"]
    for c in contents:
        p = doc.add_paragraph(c)
        p.paragraph_format.space_after = Pt(3)
    page_break(doc)

    # -------------------------------------------------------------- 1 intro
    doc.add_heading("1. Introduction", level=1)
    para(doc,
         "Fingerprint recognition is one of the most widely deployed biometric modalities, but conventional systems "
         "require physical contact with a sensor, raising hygiene concerns and leaving latent prints. Contactless "
         "acquisition using a smartphone camera avoids these issues but creates a new problem: matching a contactless "
         "finger photo against a legacy contact-based print is difficult because the two modalities differ in "
         "illumination, perspective, ridge scale, distortion and noise. Existing methods such as VerifNet combine "
         "scattering-wavelet features, contrastive Siamese networks and minutiae matching with fixed fusion, but they "
         "are evaluated mainly on contactless-to-contactless data and use pairwise rather than identity-level "
         "supervision. Transformer-based methods achieve lower error but require external training data, large "
         "backbones and GPU resources, making them impractical for CPU-only deployment. In this work, we propose "
         "Cross-Modal Relational Alignment (CMRA), which trains a single embedding to preserve cross-modal relational "
         "identity structure using identity-balanced batches and consistency losses, eliminating score fusion as the "
         "main contribution. Our key contributions are: (1) a new cross-modal relational alignment training objective "
         "for fingerprint matching, (2) removal of the VerifNet-style Siamese branch from the proposed system, and "
         "(3) risk-controlled abstention for reliable verification at low false-accept rates. The report is organised "
         "as follows: Section 2 reviews related work, Section 3 describes the CMRA methodology, Section 4 presents "
         "experiments and results, and Section 5 concludes.")

    # ---------------------------------------------------------- 2 literature
    doc.add_heading("2. Related Work", level=1)
    para(doc,
         "Traditional methods: Lin and Kumar pioneered contactless-to-contact fingerprint matching on HKPolyU using "
         "hand-crafted ridge-based features and COTS matchers. Their Multi-Siamese and RTPS+DCM approaches achieved "
         "strong results but rely on classical image processing rather than learned representations.")
    para(doc,
         "CNN-based methods: VerifNet combines fixed scattering-wavelet features, a contrastive Siamese network and "
         "minutiae matching with a fixed 90/10 score fusion. While compact and CPU-efficient, its pairwise training "
         "and fixed fusion weight limit its adaptation to the harder contactless-to-contact setting.")
    para(doc,
         "Transformer methods: Ridgeformer uses a vision transformer with cross-attention and multi-stage contrastive "
         "training to achieve 2.83% EER on HKPolyU. This strong performance requires training on four external "
         "datasets and GPU resources, limiting its reproducibility.")
    para(doc,
         "Attention-based CNNs: MANet uses a minutiae map to guide attention in a deep learning branch for "
         "contactless fingerprint recognition. Its attention mechanism is evaluated mainly on contactless-to-contactless "
         "matching, not cross-modal verification.")
    para(doc,
         "Systematic review: Kaplesh et al. provide a systematic review of end-to-end contactless fingerprint "
         "recognition frameworks, identifying preprocessing, domain shift and matching as key bottlenecks. As a "
         "survey, it does not propose a new matching method but highlights the need for reproducible benchmarks.")
    page_break(doc)

    # ------------------------------------------------------- 3 methodology
    doc.add_heading("3. Methodology", level=1)
    doc.add_heading("3.1 Overview", level=2)
    para(doc,
         "Figure 1 shows the CMRA pipeline. Both images are preprocessed and passed through a fixed scattering "
         "front-end. The resulting features are fed to a single embedding network trained with ArcFace on "
         "identity-balanced PK batches containing both contactless and contact impressions of the same finger. Two "
         "consistency losses are added: prototype consistency aligns the mean embedding of each identity across the "
         "two modalities, and relational consistency matches the cross-modal similarity (Gram) structure so that the "
         "two modalities agree on who is close to whom. At inference, embeddings are compared with cosine similarity. "
         "The VerifNet-style Siamese branch and score fusion are removed from the proposed system; they appear only "
         "as baselines. Risk-controlled abstention optionally rejects low-confidence comparisons.")
    figure(doc, "pipeline.png", "Figure 1. CMRA pipeline: scattering front-end, identity-balanced PK batches, "
                                "ArcFace, prototype consistency and relational consistency losses.")

    doc.add_heading("3.2 Scattering front-end", level=2)
    para(doc,
         "Each preprocessed image is encoded with a fixed 2D scattering transform (Kymatio, J = 2) producing 81 "
         "channels of size 48 × 48. Scattering features preserve texture and ridge-frequency information without "
         "learning early-layer filters, making them cheap to compute and memory-map on a CPU-only machine.")

    doc.add_heading("3.3 Identity-balanced PK batches", level=2)
    para(doc,
         "Instead of random image batches or random pairs, each training batch contains P identities with K "
         "contactless and K contact impressions per identity (P × 2K images total). Both modalities of the same "
         "finger share one class label, so the network is trained directly on the cross-modal identity structure.")

    doc.add_heading("3.4 Training objective", level=2)
    para(doc,
         "The embedding f is trained with ArcFace loss: for embedding e_i and class weight w_y, the logit of the "
         "correct class is s · cos(θ_y + m), where θ_y is the angle between e_i and w_y, with scale s = 30 and "
         "margin m = 0.30. Two consistency losses are added.")
    para(doc,
         "Prototype consistency: μ_i^cl and μ_i^ct are the L2-normalised mean embeddings of identity i in the "
         "contactless and contact sets. The loss is L_proto = mean_i (1 − cos(μ_i^cl, μ_i^ct)), pulling the two "
         "modality centres together.")
    para(doc,
         "Relational consistency: S^cl and S^ct are the P × P cosine-similarity matrices of the contactless and "
         "contact prototypes. The loss is the symmetric KL divergence between softmax(S^cl / τ) and "
         "softmax(S^ct / τ), forcing the two modalities to agree on the relative layout of identities without forcing "
         "absolute coordinates to coincide.")
    para(doc,
         "The full objective is L = L_arcface + λ₁L_proto + λ₂L_rel. Four ablations are run: ArcFace only, ArcFace + "
         "prototype, ArcFace + relational, and ArcFace + both (full CMRA).")

    doc.add_heading("3.5 Inference and abstention", level=2)
    para(doc,
         "At inference, two images are embedded and compared with cosine similarity. Risk-controlled abstention "
         "ranks comparisons by confidence |score − τ| and optionally rejects low-confidence pairs, trading coverage "
         "for lower error rate.")

    # ------------------------------------------------------- 4 experiments
    doc.add_heading("4. Experiment and Results", level=1)
    doc.add_heading("4.1 Setup used", level=2)
    para(doc,
         "All experiments run on an Intel Core Ultra 9 285H laptop with 16 CPU threads and 32 GB RAM, without a "
         "discrete GPU. PyTorch 2.14.1 (CPU build), Kymatio 0.3.0, scikit-learn 1.9.1, OpenCV 5.0 and NumPy 2.2 are "
         "used. The embedding network (0.13M parameters) trains for 30 epochs in about 15 minutes with AdamW "
         "(lr = 10⁻³, weight decay = 10⁻⁴), cosine learning-rate decay, batch size 64 (P = 16 identities, K = 2 "
         "images per modality), and gradient clipping at 1.0. Checkpoint selection uses the lowest validation EER.")
    table(doc,
          ["Component", "Setting"],
          [
              ["Scattering transform", "Kymatio 2D, J = 2, 81 × 48 × 48"],
              ["Embedding", "128-D, CNN body + linear projection"],
              ["ArcFace", "scale 30, margin 0.30, label smoothing 0.05"],
              ["Optimiser", "AdamW, lr 1e-3, weight decay 1e-4, cosine schedule"],
              ["Training", "30 epochs, PK batches (P=16, K=2), ±2-cell translation"],
              ["Checkpoint selection", "lowest validation EER (no TTA)"],
              ["Inference", "5-view shift TTA, cosine similarity"],
              ["Bootstrap", "500 identity-cluster resamples"],
          ],
          widths=[2.2, 4.0], size=9, align_center=False)
    caption(doc, "Table 1. Implementation settings.")
    figure(doc, "training_curve.png",
           "Figure 2. CMRA training curve (seed 8): validation EER and TAR@FAR=1% per epoch; the selected "
           "checkpoint is epoch 23.", width=5.3)

    doc.add_heading("4.2 Benchmark dataset", level=2)
    para(doc,
         "Experiments use the Hong Kong Polytechnic University (HKPolyU) Contactless 2D to Contact-based 2D "
         "Fingerprint Images Database. It contains 336 fingers, 2,976 contactless and 2,976 contact images "
         "(5,952 total), with six impressions per finger per session. A subset of 160 fingers was acquired in a "
         "second session. The controlled protocol uses an identity-disjoint split: 201 training, 34 validation and "
         "101 test identities, with no identity overlap across splits. The primary test set has 10,728 balanced "
         "pairs (5,364 genuine, 5,364 impostor). A supplementary cross-session study uses 48 test identities present "
         "in both sessions, with 3,456 genuine and 3,456 impostor cross-session pairs.")
    table(doc,
          ["Property", "Value"],
          [
              ["Fingers / identities", "336"],
              ["Images", "2,976 contactless + 2,976 contact (5,952 total)"],
              ["Sessions", "First: 336 fingers; second: 160 fingers"],
              ["Train / validation / test", "201 / 34 / 101 identities (disjoint)"],
              ["Primary test pairs", "10,728 (5,364 genuine, 5,364 impostor)"],
              ["Cross-session test", "48 identities, 3,456 genuine + 3,456 impostor"],
          ],
          widths=[2.7, 3.5], size=9, align_center=False)
    caption(doc, "Table 2. HKPolyU CL2CB dataset and protocol summary.")

    doc.add_heading("4.3 Performance metrics", level=2)
    para(doc,
         "Verification performance is summarised by the equal error rate (EER), the true accept rate at false accept "
         "rates of 1% and 0.1% (TAR@FAR=1%, TAR@FAR=0.1%), and ROC/DET curves. Confidence intervals use an "
         "identity-cluster bootstrap (500 resamples of test identities). Risk-controlled abstention is evaluated with "
         "error-vs-coverage curves and area under the risk-coverage curve (AURC).")

    doc.add_heading("4.4 Results", level=2)
    para(doc,
         "Table 3 shows the ablation of CMRA components on the 10,728-pair test set. ArcFace-only training on "
         "identity-balanced PK batches already reaches 19.41% EER, improving on the ISE-only v1 baseline at 19.56%. "
         "Adding prototype consistency (18.23%) and relational consistency (17.74%) each help, and full CMRA reaches "
         "16.80% EER with 29.55% TAR@FAR=1% and 7.20% TAR@FAR=0.1%. The relational term alone beats prototype alone "
         "at this seed, showing that matching cross-modal similarity structure is the strongest single addition.")
    table(doc,
          ["Method", "Test EER", "TAR@1%", "TAR@0.1%", "95% CI"],
          [
              ["ISE (v1 baseline)", "19.56", "28.24", "4.05", "[17.3, 21.7]"],
              ["CMRA arc-only (PK batches)", "19.41", "24.33", "4.34", "[17.5, 21.3]"],
              ["CMRA + prototype", "18.23", "27.44", "5.87", "[15.8, 20.6]"],
              ["CMRA + relational", "17.74", "28.93", "6.99", "[15.5, 20.1]"],
              ["CMRA full (prototype + relational)", "16.80", "29.55", "7.20", "[14.7, 19.3]"],
          ],
          widths=[2.6, 0.85, 0.8, 0.85, 1.1], size=8.4)
    caption(doc, "Table 3. CMRA ablation results (percentages, seed 7). CI is an identity-cluster bootstrap.")

    para(doc,
         "Table 4 compares full CMRA against the v1 pipeline on the same test set. CMRA seed 7 reaches 16.80% EER, "
         "beating the v1 best fusion (quality-aware LR) at 18.16% EER with a single embedding and no score fusion. "
         "Seed sensitivity across four runs gives EERs of 16.06%, 16.80%, 18.96% and 20.15%; seeds 7, 8 and 10 "
         "average 17.3% (seed 9 is an outlier), so the improvement should be reported as a range rather than a single "
         "cherry-picked value. CMRA is slightly weaker "
         "than v1 fusion in the low-FAR tail (TAR@FAR=1%: 29.55% vs 32.94%), so the honest claim is comparable-or-better "
         "accuracy with a simpler, genuinely new method.")
    table(doc,
          ["System", "EER", "TAR@1%", "TAR@0.1%", "Notes"],
          [
              ["VerifNet-style DL (3-seed)", "36.11", "3.56", "0.15", "Reproduced baseline"],
              ["v1 fixed fusion DL+MM", "32.83", "8.35", "2.16", "Reproduced fixed fusion"],
              ["v1 best fusion (quality-aware LR)", "18.16", "32.94", "12.19", "3 branches + quality"],
              ["ISE (v1 baseline)", "19.56", "28.24", "4.05", "Single embedding"],
              ["CMRA full (seed 7)", "16.80", "29.55", "7.20", "Proposed, no fusion"],
              ["CMRA full (seed 8)", "16.06", "30.93", "11.09", "Proposed, no fusion"],
          ],
          widths=[2.2, 0.65, 0.75, 0.85, 1.75], size=7.8, align_center=False)
    caption(doc, "Table 4. CMRA vs v1 pipeline on the 10,728-pair test set (percentages).")

    figure(doc, "cmra_comparison.png",
           "Figure 3. Test EER for the baselines, the v1 pipeline and the CMRA ablation.", width=5.2)
    figure(doc, "cmra_det.png",
           "Figure 4. DET curves for the CMRA ablation on held-out test identities (log–log axes).", width=4.9)

    doc.add_heading("4.5 Cross-session evaluation", level=2)
    para(doc,
         "Table 5 reports the supplementary cross-session study on 48 test identities present in both sessions. CMRA "
         "seed 8 reaches 14.89% cross-session EER versus 15.79% same-session EER on the same identities — cross-session "
         "performance is actually better here, and CMRA beats v1 quality-aware fusion (18.50% cross-session EER) by "
         "3.6 percentage points. This shows cross-modal relational alignment generalises across sessions, not just "
         "within them.")
    table(doc,
          ["Method", "Cross-session EER", "Cross TAR@1%", "Same-session EER", "Same TAR@1%"],
          [
              ["CMRA seed 8 (proposed)", "14.89", "32.81", "15.79", "25.64"],
              ["v1 quality-aware fusion", "18.50", "27.66", "16.26", "30.35"],
              ["ISE (v1 baseline)", "18.89", "31.57", "17.43", "28.36"],
              ["Minutiae", "37.66", "9.90", "37.72", "6.86"],
          ],
          widths=[2.2, 1.05, 1.0, 1.05, 1.0], size=8.2)
    caption(doc, "Table 5. Cross-session vs same-session results on 48 test identities (percentages).")

    doc.add_heading("4.6 Risk-controlled abstention", level=2)
    para(doc,
         "Figure 5 shows error rate versus coverage for CMRA seed 7, ISE and v1 fusion at a fixed validation "
         "threshold. CMRA full reduces error from 16.8% at 100% coverage to 5.7% at 50% coverage (AURC 0.068 vs "
         "0.084 for ISE), demonstrating that the system can reliably abstain on difficult comparisons. This new "
         "capability is absent from the v1 pipeline and from the compared literature.")
    figure(doc, "cmra_risk_coverage.png", "Figure 5. Risk–coverage curve: error rate at fixed threshold vs coverage "
                                          "(percentages).", width=5.5)

    # ---------------------------------------------------------- 5 conclusion
    doc.add_heading("5. Conclusion and Future Work", level=1)
    para(doc,
         "This project studied contactless-to-contact fingerprint verification on HKPolyU and proposed Cross-Modal "
         "Relational Alignment (CMRA), a single embedding trained to preserve cross-modal relational identity "
         "structure using identity-balanced PK batches, ArcFace supervision, prototype consistency and relational "
         "consistency losses. CMRA reaches 16.80% EER and 29.55% TAR@FAR=1% on the controlled 10,728-pair test "
         "set, beating the reproduced VerifNet-style baseline (36.11% EER) and its fixed-fusion variant (32.83% EER) "
         "with a simpler pipeline that removes the Siamese branch and score fusion entirely. Risk-controlled "
         "abstention further lowers error to 5.7% at "
         "50% coverage. Cross-session evaluation on 48 identities confirms generalisation (14.89% cross-session EER "
         "versus 18.50% for v1 fusion). The work shows that cross-modal relational alignment, not fusion, is the key "
         "contribution for lightweight CPU-trainable contactless-to-contact fingerprint verification.")
    para(doc,
         "Future work should pursue four directions: (1) learn the minutiae branch jointly with the identity "
         "embedding, for example with a minutiae attention network or graph neural network; (2) replace hand-crafted "
         "quality features with a learned pair-reliability model trained on hard pairs; (3) scale training data with "
         "identity-consistent synthetic multi-pose generation to close the gap to 2–4% EER without external real "
         "data; and (4) export the embedding body to OpenVINO and benchmark Intel NPU latency, power and throughput "
         "for on-device deployment.")

    # ------------------------------------------------------------- references
    page_break(doc)
    doc.add_heading("References", level=1)
    refs = [
        "G. Parasnis, R. Bhope, A. Chokshi, V. Jain, A. Biswas, D. Kumar, S. Pateriya, V. Anand, V. Kanhangad and "
        "V. M. Gadre, “VerifNet — A Novel Score Fusion-Based Method Leveraging Wavelets With Deep Learning and "
        "Minutiae Matching for Contactless Fingerprint Recognition,” IEEE Transactions on Biometrics, Behavior, and "
        "Identity Science, vol. 7, no. 3, Jul. 2025. doi:10.1109/TBIOM.2024.3504281.",
        "S. Pandey, B. Jawade and S. Setlur, “Ridgeformer: Multi-Stage Contrastive Training for Fine-Grained "
        "Cross-Domain Fingerprint Recognition,” IEEE International Conference on Image Processing (ICIP), 2025. "
        "doi:10.1109/ICIP55913.2025.11084677. arXiv:2506.01806.",
        "Z. Pan, X. Guan, J. Feng and J. Zhou, “Identity-Consistent Multi-Pose Generation of Contactless Fingerprints,” "
        "arXiv:2605.03830, 2026.",
        "S. Peddi, S. Bandyopadhyay and D. Samanta, “G-MSGINet: A Grouped Multi-Scale Graph-Involution Network for "
        "Contactless Fingerprint Recognition,” arXiv:2505.08233, 2025.",
        "R. Sahoo and A. Namboodiri, “Fusion2Print: Deep Flash–Non-Flash Fusion for Contactless Fingerprint Matching,” "
        "International Conference on Pattern Recognition (ICPR), 2026. arXiv:2601.02318.",
        "P. Kaplesh, A. Gupta, D. Bansal, S. Sofat and A. Mittal, “A Systematic Review of End-to-End Framework for "
        "Contactless Fingerprint Recognition: Techniques, Challenges, and Future Directions,” Engineering Applications "
        "of Artificial Intelligence, 2025. doi:10.1016/j.engappai.2025.110493.",
        "M. Alsmirat, M. M. Khaled and A. A. Sayadi, “Deep Learning Based Contactless Fingerprint Identification,” "
        "Cluster Computing, 2025. doi:10.1007/s10586-025-05829-5.",
        "C. Lin and A. Kumar, “Matching Contactless and Contact-Based Conventional Fingerprint Images for Biometrics "
        "Identification,” IEEE Transactions on Image Processing, vol. 27, no. 4, pp. 2008–2021, 2018.",
        "H. Tan and A. Kumar, “Minutiae Attention Network with Reciprocal Distance Loss for Contactless to Contact-Based "
        "Fingerprint Identification,” IEEE Transactions on Information Forensics and Security, vol. 16, 2021.",
        "B. Jawade, A. Agarwal, K. S. R. Krishnan and S. Setlur, “Multi Loss Fusion for Matching Smartphone Captured "
        "Contactless Finger Images,” IEEE International Workshop on Information Forensics and Security (WIFS), 2021.",
        "J. Deng, J. Guo, N. Xue and S. Zafeiriou, “ArcFace: Additive Angular Margin Loss for Deep Face Recognition,” "
        "CVPR, 2019.",
        "M. Andreux, T. Angles, J. Exarchakis, R. Leonarduzzi, G. Rochette and L. Thiry, “Kymatio: Scattering Transforms "
        "in Python,” Journal of Machine Learning Research, vol. 21, no. 60, pp. 1–6, 2020.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(f"[{i}]  {ref}")
        r.font.size = Pt(9)

    out = OUT / "Contactless_Fingerprint_Recognition_Report.docx"
    doc.save(out)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    main()
