"""Build the 13-slide CMRA project presentation."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "slides"
OUT.mkdir(exist_ok=True)

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
NAVY_LIGHT = RGBColor(0x2E, 0x52, 0x82)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
ORANGE = RGBColor(0xC2, 0x41, 0x0C)
GREY = RGBColor(0x6B, 0x72, 0x80)
LIGHT = RGBColor(0xF1, 0xF5, 0xFA)
MID = RGBColor(0xD7, 0xE0, 0xEC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x1F, 0x29, 0x37)
PURPLE = RGBColor(0x5B, 0x21, 0xB6)

prs = Presentation()
cp = prs.core_properties
cp.title = "Contactless-to-Contact Fingerprint Recognition — CMRA"
cp.subject = "Biometric Security Course Project — HKPolyU CL2CB"
cp.author = "Aryan Desai; Naman Sinha; Saurabh Bharrdwaj; Sayan Das"
cp.comments = "Cross-Modal Relational Alignment with risk-controlled abstention"
cp.keywords = "contactless fingerprint; cross-modal matching; relational alignment; ArcFace; CMRA"
cp.category = "Course project presentation"
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
TOTAL = 13


def solid(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def textbox(slide, x, y, w, h, text, size=16, bold=False, color=DARK,
            align=PP_ALIGN.LEFT, font="Calibri", anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return tb


def bullets(slide, x, y, w, h, items, size=15, color=DARK, space=8):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = 0
    first = True
    for item in items:
        level = 0
        text = item
        if isinstance(item, tuple):
            level, text = item
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(space)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Inches(0.26 if level == 0 else 0.52))))
        pPr.set("indent", str(int(-Inches(0.24))))
        for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
            for el in pPr.findall(qn(tag)):
                pPr.remove(el)
        buFont = etree.SubElement(pPr, qn("a:buFont"))
        buFont.set("typeface", "Arial")
        buChar = etree.SubElement(pPr, qn("a:buChar"))
        buChar.set("char", "•" if level == 0 else "–")
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size if level == 0 else size - 2)
        r.font.color.rgb = color
        r.font.name = "Calibri"
    return tb


def content_slide(title, kicker=None):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.02))
    solid(bar, NAVY)
    accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.02), prs.slide_width, Inches(0.055))
    solid(accent, GREEN)
    textbox(s, 0.55, 0.20, 11.0, 0.62, title, size=26, bold=True, color=WHITE,
            anchor=MSO_ANCHOR.MIDDLE)
    if kicker:
        textbox(s, 10.0, 0.34, 2.85, 0.35, kicker, size=11, color=RGBColor(0xC7, 0xD6, 0xE8),
                align=PP_ALIGN.RIGHT)
    line = s.shapes.add_connector(1, Inches(0.55), Inches(7.06), Inches(12.78), Inches(7.06))
    line.line.color.rgb = MID
    line.line.width = Pt(0.75)
    textbox(s, 0.55, 7.12, 7.0, 0.26, "Biometric Security · Course Project · CMRA", size=9, color=GREY)
    n = len(prs.slides)
    textbox(s, 12.0, 7.12, 0.8, 0.26, f"{n} / {TOTAL}", size=9, color=GREY, align=PP_ALIGN.RIGHT)
    return s


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def pic(slide, name, x, y, w):
    return slide.shapes.add_picture(str(RESULTS / name), Inches(x), Inches(y), width=Inches(w))


def table(slide, headers, rows, x, y, w, col_widths, font=10.5, row_h=0.34,
          highlight=None, header_h=0.38):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y),
                                   Inches(w), Inches(header_h + row_h * len(rows)))
    t = shape.table
    t.first_row = True
    t.horz_banding = False
    for i, cw in enumerate(col_widths):
        t.columns[i].width = Inches(cw)
    t.rows[0].height = Inches(header_h)
    for j in range(len(rows)):
        t.rows[j + 1].height = Inches(row_h)
    for i, h in enumerate(headers):
        cell = t.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.margin_left = cell.margin_right = Inches(0.05)
        cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if i else PP_ALIGN.LEFT
        r = p.add_run()
        r.text = h
        r.font.size = Pt(font)
        r.font.bold = True
        r.font.color.rgb = WHITE
    for j, row in enumerate(rows):
        for i, v in enumerate(row):
            cell = t.cell(j + 1, i)
            cell.fill.solid()
            cell.fill.fore_color.rgb = (
                RGBColor(0xDC, 0xF2, 0xE3) if highlight is not None and j == highlight
                else (LIGHT if j % 2 == 0 else WHITE))
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if i else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(v)
            r.font.size = Pt(font)
            r.font.color.rgb = DARK
            if highlight is not None and j == highlight:
                r.font.bold = True
    return t


def card(slide, x, y, w, h, value, label, color):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    solid(sh, LIGHT)
    sh.line.color.rgb = MID
    sh.line.width = Pt(1)
    sh.adjustments[0] = 0.08
    textbox(slide, x, y + 0.12, w, 0.55, value, size=28, bold=True, color=color,
            align=PP_ALIGN.CENTER)
    textbox(slide, x + 0.1, y + 0.72, w - 0.2, 0.5, label, size=11, color=GREY,
            align=PP_ALIGN.CENTER)


def callout(slide, x, y, w, h, text, color=GREEN, size=12.5):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    solid(sh, RGBColor(0xE8, 0xF5, 0xEC))
    sh.line.color.rgb = color
    sh.line.width = Pt(1.2)
    sh.adjustments[0] = 0.12
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.12)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = NAVY
    return sh


def method_box(slide, x, y, w, h, title, equation, body, color=PURPLE):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    solid(sh, LIGHT)
    sh.line.color.rgb = color
    sh.line.width = Pt(1.4)
    sh.adjustments[0] = 0.06
    textbox(slide, x + 0.18, y + 0.12, w - 0.36, 0.3, title, size=15, bold=True, color=color)
    textbox(slide, x + 0.18, y + 0.50, w - 0.36, 0.42, equation, size=13.5, bold=True,
            color=NAVY, font="Cambria Math")
    textbox(slide, x + 0.18, y + 0.98, w - 0.36, h - 1.06, body, size=11.5, color=DARK)
    return sh


# ---------------------------------------------------------------- 1 title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
solid(bg, NAVY)
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(5.35), prs.slide_width, Inches(0.08))
solid(band, GREEN)
textbox(s, 0.9, 0.75, 11.5, 0.4, "NATIONAL INSTITUTE OF TECHNOLOGY WARANGAL", size=13,
        bold=True, color=RGBColor(0xC7, 0xD6, 0xE8))
textbox(s, 0.9, 1.12, 11.5, 0.35, "Biometric Security — Course Project", size=12,
        color=RGBColor(0x9F, 0xB4, 0xCD))
textbox(s, 0.9, 2.05, 11.8, 1.0, "Contactless-to-Contact Fingerprint Recognition", size=40,
        bold=True, color=WHITE)
textbox(s, 0.9, 3.10, 11.8, 0.75,
        "Cross-Modal Relational Alignment (CMRA) with Risk-Controlled Abstention",
        size=20, color=RGBColor(0xB9, 0xE6, 0xC9))
textbox(s, 0.9, 3.95, 11.5, 0.4, "A reproducible study on the HKPolyU Contactless 2D to Contact-based 2D database",
        size=13, color=RGBColor(0xC7, 0xD6, 0xE8))
textbox(s, 0.9, 5.75, 6.5, 0.35, "Aryan Desai · Naman Sinha · Saurabh Bharrdwaj · Sayan Das", size=13,
        color=WHITE)
textbox(s, 0.9, 6.15, 6.5, 0.3, "November 2026", size=11, color=RGBColor(0x9F, 0xB4, 0xCD))
notes(s, "Title. The final system is CMRA: one shared embedding trained with identity-balanced batches, "
         "ArcFace and two cross-modal consistency losses. The VerifNet-style Siamese branch and score fusion "
         "are now baselines only.")

# ---------------------------------------------------------------- 2 motivation
s = content_slide("Why contactless-to-contact is still hard", "MOTIVATION")
bullets(s, 0.6, 1.45, 6.5, 4.9, [
    "Contactless capture removes the sensor, but not the legacy contact database.",
    "Cross-modal gap: illumination, pose, perspective, ridge scale and sensor noise change together.",
    "VerifNet-style systems still rely on a Siamese branch plus fixed 90/10 score fusion.",
    "The strongest 2025–26 methods need transformers, external data or synthetic augmentation.",
    "Biometric security is decided at low FAR, where calibration and abstention matter.",
], size=14.5, space=13)
card(s, 7.5, 1.55, 4.9, 1.15, "Hygiene + access", "touchless capture, remote onboarding", NAVY_LIGHT)
card(s, 7.5, 2.90, 4.9, 1.15, "Interoperability", "match phone photos to legacy prints", NAVY_LIGHT)
card(s, 7.5, 4.25, 4.9, 1.15, "Security", "operate at FAR = 1% and 0.1%", GREEN)
callout(s, 7.5, 5.70, 4.9, 0.85,
        "Goal: one embedding that aligns cross-modal identity structure — not another fusion layer.")
notes(s, "Set up the problem: the sensor disappears but interoperability does not. Emphasise the low-FAR regime, "
         "because that is where a biometric system is judged.")

# ---------------------------------------------------------------- 3 data
s = content_slide("Data and evaluation protocol", "HKPolyU CL2CB")
card(s, 0.6, 1.45, 3.0, 1.15, "336", "fingers, 2 sessions", NAVY)
card(s, 4.0, 1.45, 3.0, 1.15, "5,952", "paired images (2 modalities)", NAVY)
card(s, 7.4, 1.45, 3.0, 1.15, "101", "held-out test identities", GREEN)
card(s, 10.55, 1.45, 2.2, 1.15, "10,728", "test pairs", NAVY)
bullets(s, 0.6, 3.05, 12.2, 3.6, [
    "Every finger: 6 contactless + 6 contact impressions per session; 160 fingers return in session 2.",
    "Split by finger: 201 train / 34 validation / 101 test — no identity appears twice.",
    "Primary test: 10,728 pairs (5,364 genuine, 5,364 impostor); all genuine pairs are same-session.",
    "Supplementary cross-session test: 48 test fingers in both sessions, 3,456 genuine + 3,456 impostor pairs.",
    "Metrics: EER, TAR@FAR = 1% and 0.1%; 95% CI from 500 identity-cluster bootstrap resamples.",
], size=14.5, space=9)
callout(s, 0.6, 6.05, 12.2, 0.55,
        "Model selection uses validation only; the test set is scored once after freezing.", NAVY_LIGHT)
notes(s, "Stress the identity-disjoint split and that the primary genuine pairs are same-session. The cross-session "
         "study is supplementary because only 48 test fingers have a second session.")

# ---------------------------------------------------------------- 4 literature
s = content_slide("Related work: five papers, one gap", "LITERATURE")
rows = [
    ["Lin & Kumar (TIP 2018)", "Traditional ridge and minutiae features + COTS matcher",
     "Contactless-to-contact baseline on HKPolyU"],
    ["Tan & Kumar MANet (TIFS 2021)", "Minutiae-guided attention CNN",
     "Contactless-to-contactless identification"],
    ["VerifNet (TBIOM 2025)", "Scattering + Siamese + minutiae, fixed 90/10",
     "2.5% EER, IITI-CFD; CL2CL only"],
    ["Ridgeformer (ICIP 2025)", "ViT + multi-stage contrastive, 4 training sets",
     "2.83% EER, HKPolyU"],
    ["Kaplesh et al. (EAAI 2025)", "Systematic review of end-to-end pipelines",
     "Domain shift and preprocessing remain open"],
]
table(s, ["Paper", "Core idea", "Reported result"], rows, 0.6, 1.40, 12.1,
      [3.35, 5.0, 3.75], font=10.5, row_h=0.62)
bullets(s, 0.6, 5.35, 12.2, 1.3, [
    "Common thread: representation, calibration and quality are entangled; fusion weights are usually fixed.",
    "Gap: no reproducible identity-disjoint CL2CB baseline built around a genuinely different training objective.",
], size=12.5, space=5)
callout(s, 0.6, 6.45, 12.1, 0.5,
        "State of the art is 2–4% EER — but with external data, transformers or synthetic pipelines.", ORANGE)
notes(s, "The point is not the leaderboard. The strong numbers come from settings we cannot reproduce on a CPU. "
         "VerifNet is the closest lightweight idea, but it is evaluated contactless-to-contactless and keeps a fixed fusion.")

# ---------------------------------------------------------------- 5 pipeline
s = content_slide("Proposed pipeline: CMRA", "METHOD")
pic(s, "pipeline.png", 1.15, 1.22, 11.0)
textbox(s, 0.55, 6.62, 12.2, 0.35,
        "One shared embedding; the Siamese branch and score fusion are removed from the proposed system.",
        size=12.5, color=GREY)
notes(s, "Walk left to right. Both modalities pass through the same front-end and embedding. The two consistency "
         "losses are used only for training; at inference only cosine similarity and the optional abstention remain.")

# ---------------------------------------------------------------- 6 embedding
s = content_slide("A CPU-cheap front-end, one shared embedding", "METHOD")
bullets(s, 0.6, 1.45, 6.3, 4.7, [
    "Fixed 2-D scattering transform (J = 2, 81 × 48 × 48) replaces learned early filters.",
    "Three conv blocks → 128-D L2-normalised embedding; about 0.13M parameters.",
    "ArcFace: scale 30, margin 0.30; both modalities share one identity class.",
    "30 epochs, AdamW, ±2-cell translation, label smoothing 0.05.",
    "Checkpoint at epoch 23: validation EER 13.49%, test EER 16.06% (seed 8).",
], size=14, space=11)
pic(s, "training_curve.png", 7.05, 1.50, 5.7)
callout(s, 0.6, 5.85, 6.25, 0.6, "Trained from scratch on 201 identities — no external data.", GREEN)
notes(s, "The representation is deliberately cheap: a fixed scattering transform plus a small CNN. The training curve "
         "shows validation EER and TAR@1% moving in the right direction; the test checkpoint is frozen.")

# ---------------------------------------------------------------- 7 PK batches
s = content_slide("Identity-balanced PK batches + ArcFace", "METHOD")
card(s, 0.6, 1.45, 2.9, 1.15, "P = 16", "identities per batch", NAVY)
card(s, 3.65, 1.45, 2.9, 1.15, "K = 2 + 2", "contactless + contact per identity", NAVY)
card(s, 0.6, 2.80, 2.9, 1.15, "64", "images per training step", GREEN)
card(s, 3.65, 2.80, 2.9, 1.15, "64", "steps per epoch", GREEN)
bullets(s, 0.6, 4.20, 6.0, 2.4, [
    "Sampled by identity, not by pair: every step contains both modalities of the same fingers.",
    "Random crop ±2 cells; the identity split is never violated.",
    "A shared class label lets ArcFace learn the cross-modal identity structure directly.",
], size=12.5, space=8)
textbox(s, 7.05, 1.42, 5.7, 0.35, "ArcFace supervision", size=17, bold=True, color=PURPLE)
eq = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.05), Inches(1.85),
                        Inches(5.7), Inches(0.95))
solid(eq, LIGHT)
eq.line.color.rgb = PURPLE
eq.line.width = Pt(1.2)
tf = eq.text_frame
tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = "L_arc = −log( e^{s·cos(θ_y + m)} / Σ_j e^{s·cos(θ_j)} )"
r.font.size = Pt(14)
r.font.bold = True
r.font.color.rgb = NAVY
r.font.name = "Cambria Math"
bullets(s, 7.05, 3.00, 5.7, 3.6, [
    "s = 30, m = 0.30; label smoothing 0.05.",
    "θ_y is the angle between the embedding and its class weight.",
    "Contactless and contact images of the same finger share the label y.",
    "This replaces pairwise contrastive training with identity-level supervision.",
], size=13, space=10)
callout(s, 7.05, 6.10, 5.7, 0.6, "Identity-level supervision, not pairwise contrastive loss.", GREEN)
notes(s, "The PK sampler is the structural change. Because every batch contains both modalities of the same identity, "
         "the consistency losses below have a meaningful cross-modal signal.")

# ---------------------------------------------------------------- 8 losses
s = content_slide("Two cross-modal consistency losses", "METHOD")
method_box(s, 0.6, 1.40, 5.95, 2.05, "Prototype consistency",
           "L_proto = (1/P) Σ_i [ 1 − cos(μ_i^CL, μ_i^CT) ]",
           "Pulls the mean embedding of each identity across the two modalities together.", PURPLE)
method_box(s, 6.80, 1.40, 5.95, 2.05, "Relational consistency",
           "L_rel = ½ KL(p_CT ‖ p_CL) + ½ KL(p_CL ‖ p_CT)",
           "p = softmax(S / τ); aligns who-is-close-to-whom across modalities.", RGBColor(0x03, 0x69, 0xA1))
band = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(3.65),
                          Inches(12.15), Inches(0.7))
solid(band, RGBColor(0xE8, 0xF5, 0xEC))
band.line.color.rgb = GREEN
band.line.width = Pt(1.3)
tf = band.text_frame
tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = "L = L_arc + λ_proto·L_proto + λ_rel·L_rel     with λ_proto = λ_rel = 0.5,  τ = 0.1"
r.font.size = Pt(15)
r.font.bold = True
r.font.color.rgb = NAVY
r.font.name = "Cambria Math"
bullets(s, 0.6, 4.60, 12.15, 1.7, [
    "Prototype loss aligns modality centres; relational loss aligns the full P × P similarity structure.",
    "Temperature τ = 0.1 keeps the KL term numerically meaningful — a raw Gram MSE was negligible (~0.001).",
    "At seed 7, relational-only (17.74% EER) is the strongest single addition, ahead of prototype-only (18.23%).",
], size=12.5, space=6)
callout(s, 0.6, 6.15, 12.15, 0.55,
        "The network is trained on the cross-modal relation, not only on identity labels.", GREEN)
notes(s, "This is the new part. ArcFace alone gives identity separation; the two consistency terms force the two "
         "modalities to agree on identity centres and on relative similarity structure.")

# ---------------------------------------------------------------- 9 results
s = content_slide("Ablation: where the gain comes from", "RESULTS")
rows = [
    ["ISE (v1 single embedding)", "19.56", "28.24", "4.05"],
    ["CMRA ArcFace + PK only", "19.41", "24.33", "4.34"],
    ["CMRA + prototype", "18.23", "27.44", "5.87"],
    ["CMRA + relational", "17.74", "28.93", "6.99"],
    ["CMRA full (prototype + relational)", "16.80", "29.55", "7.20"],
]
table(s, ["Method", "EER", "TAR@1%", "TAR@0.1%"], rows, 0.55, 1.45, 7.3,
      [3.55, 1.05, 1.25, 1.45], font=10.5, row_h=0.48, highlight=4)
textbox(s, 0.55, 4.55, 7.3, 0.35, "Percentages; 10,728 test pairs, 101 identities, seed 7.",
        size=10.5, color=GREY)
bullets(s, 8.15, 1.50, 4.6, 4.8, [
    "ArcFace + PK alone (19.41%) is already at the ISE level (19.56%).",
    "Each consistency term helps; full CMRA reaches 16.80% EER.",
    "v1 best fusion is 18.16% EER, but 32.94% TAR@1% — still stronger in the low-FAR tail.",
    "Four seeds span 16.06–20.15%; report mean ± spread, not one cherry-picked seed.",
], size=12.5, space=10)
callout(s, 0.55, 5.55, 12.15, 0.75,
        "Single embedding, no score fusion — and a lower EER than the v1 fusion pipeline.", GREEN)
notes(s, "Lead with the ablation. ArcFace + PK is a control; the consistency terms are the actual contribution. "
         "Be explicit that v1 fusion still wins TAR@1% and TAR@0.1%.")

# ---------------------------------------------------------------- 10 DET
s = content_slide("DET curves across the operating range", "RESULTS")
pic(s, "cmra_det.png", 0.55, 1.35, 6.9)
bullets(s, 7.75, 1.55, 5.0, 4.4, [
    "CMRA full has the lowest EER of the ablation.",
    "ArcFace-only and ISE overlap; the consistency losses separate them.",
    "The low-FAR tail is close: v1 fusion keeps a small advantage at TAR@1% and 0.1%.",
    "DET uses log–log axes, so the tail is visible rather than hidden by the EER point.",
], size=13, space=12)
callout(s, 7.75, 5.90, 5.0, 0.8,
        "Honest claim: comparable-or-better accuracy with a simpler, genuinely new pipeline.", ORANGE)
notes(s, "The DET plot is the fair way to show this result. CMRA wins at EER and remains competitive in the tail, "
         "but the v1 fusion tail is still slightly better.")

# ---------------------------------------------------------------- 11 abstention
s = content_slide("Risk-controlled abstention", "RESULTS")
pic(s, "cmra_risk_coverage.png", 0.55, 1.40, 6.5)
rows = [
    ["100%", "16.04"],
    ["90%", "13.00"],
    ["80%", "9.94"],
    ["70%", "7.68"],
    ["50%", "3.88"],
]
table(s, ["Coverage", "Error rate"], rows, 7.35, 1.55, 4.9, [2.4, 2.5],
      font=11, row_h=0.48, highlight=4)
bullets(s, 7.35, 4.35, 5.0, 1.5, [
    "Rank by confidence |s − τ| and reject the least confident pairs.",
    "AURC 0.056 vs 0.084 for ISE.",
], size=12.5, space=8)
callout(s, 0.55, 6.05, 12.15, 0.65,
        "At 50% coverage the error falls to 3.9% — a security-facing capability absent from v1 and the compared literature.",
        GREEN)
notes(s, "This is the deployment story: when the system is unsure, it can ask for another capture instead of "
         "accepting a risky comparison. Coverage 50% means half of the pairs are auto-decided.")

# ---------------------------------------------------------------- 12 cross session
s = content_slide("Does it survive a session gap?", "SUPPLEMENTARY")
rows = [
    ["CMRA seed 8 (proposed)", "14.89", "32.81", "15.79", "25.64"],
    ["v1 quality-aware fusion", "18.50", "27.66", "16.26", "30.35"],
    ["ISE (v1 baseline)", "18.89", "31.57", "17.43", "28.36"],
    ["Minutiae", "37.66", "9.90", "37.72", "6.86"],
]
table(s, ["Method", "Cross EER", "Cross TAR@1%", "Same EER", "Same TAR@1%"], rows,
      0.55, 1.55, 8.0, [2.55, 1.35, 1.45, 1.2, 1.45], font=10.5, row_h=0.62, highlight=0)
bullets(s, 8.9, 1.55, 4.0, 3.4, [
    "CMRA beats v1 fusion by 3.6 points cross-session.",
    "Cross-session EER is actually lower than same-session here.",
    "The gain is not a same-session artefact.",
], size=12.5, space=12)
callout(s, 0.55, 5.60, 12.15, 0.85,
        "48 test identities present in both sessions; 3,456 genuine and 3,456 impostor cross-session pairs.",
        NAVY_LIGHT)
notes(s, "Caveat first: only 48 identities, and the genuine pairs are constructed differently. It is evidence of "
         "robustness, not a full cross-session benchmark.")

# ---------------------------------------------------------------- 13 conclusion
s = content_slide("Conclusion", "WRAP-UP")
textbox(s, 0.6, 1.30, 12.2, 0.45,
        "16.80% EER  ·  29.55% TAR@1%  ·  3.9% error at 50% coverage",
        size=20, bold=True, color=GREEN)
cols = [
    ("What worked", [
        "Cross-modal relational alignment.",
        "Single embedding, no score fusion.",
        "CPU-only: 30 epochs, 0.13M parameters.",
        "Honest identity-disjoint protocol.",
    ], GREEN),
    ("Honest limits", [
        "Seed variance: 16.06–20.15%.",
        "v1 fusion still better at TAR@1%/0.1%.",
        "Cross-session study covers 48 identities.",
        "No state-of-the-art claim.",
    ], ORANGE),
    ("Next", [
        "Joint minutiae + identity learning.",
        "Learned pair-reliability model.",
        "Synthetic multi-pose data.",
        "OpenVINO / NPU deployment.",
    ], NAVY_LIGHT),
]
for i, (title, items, color) in enumerate(cols):
    x = 0.6 + i * 4.15
    hd = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(2.00), Inches(3.85), Inches(0.5))
    solid(hd, color)
    textbox(s, x + 0.15, 2.07, 3.55, 0.36, title, size=14, bold=True, color=WHITE)
    body = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(2.50), Inches(3.85), Inches(2.7))
    solid(body, LIGHT)
    body.line.color.rgb = MID
    bullets(s, x + 0.12, 2.65, 3.6, 2.45, items, size=12, space=8)
callout(s, 0.6, 5.55, 12.15, 0.85,
        "A genuinely different training objective — relational alignment — not another fusion of the VerifNet score.",
        GREEN)
notes(s, "Close on the methodological point: CMRA changes how the embedding is trained, not just how scores are "
         "combined. Leave time for questions about seed variance and the low-FAR tail.")

out = OUT / "Contactless_Fingerprint_Recognition_Slides.pptx"
prs.save(out)
print(f"saved {out} ({len(prs.slides)} slides)")
