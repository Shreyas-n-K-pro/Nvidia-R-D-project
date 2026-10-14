#!/usr/bin/env python3
"""Generate a professional Word document (.docx) for the 7-minute presentation script."""
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=140, bottom=140, left=200, right=200):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_script_document(output_path="assets/ASM_7Min_Presentation_Script.docx"):
    doc = docx.Document()

    # Set page margins (1 inch all sides)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Document Header / Pre-title
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(4)
    run_meta = p_meta.add_run("KLE TECHNOLOGICAL UNIVERSITY  ×  NVIDIA  |  SENIOR DESIGN PROJECT 2026–27")
    run_meta.font.name = "Arial"
    run_meta.font.size = Pt(9.5)
    run_meta.font.bold = True
    run_meta.font.color.rgb = RGBColor(10, 37, 64) # Navy

    # Document Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(2)
    run_title = p_title.add_run("Develop and Validate Optimised Multimodal Models for Automotive Use Cases")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(10, 37, 64)

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(14)
    run_sub = p_sub.add_run("7-Minute Slide-by-Slide Spoken Presentation Script (Bullet-Wise Delivery)")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(12)
    run_sub.font.color.rgb = RGBColor(0, 112, 186) # Accent Blue

    # Metadata Summary Box (Table)
    table = doc.add_table(rows=2, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    col_widths = [Inches(3.1), Inches(3.4)]
    for row in table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = width

    set_cell_background(table.cell(0, 0), "F8FAFC")
    set_cell_background(table.cell(0, 1), "F8FAFC")
    set_cell_background(table.cell(1, 0), "F8FAFC")
    set_cell_background(table.cell(1, 1), "F8FAFC")

    table.cell(0, 0).paragraphs[0].text = "NVIDIA Guides: Gaurav Tendolkar, Pradipt Trivedi"
    table.cell(0, 1).paragraphs[0].text = "Project Guide: Dr. Malathi S Y"
    table.cell(1, 0).paragraphs[0].text = "Team Members: Saanvi Shetty, Raveesh Nayak, Ramya Rao, Shreyas Kulkarni"
    table.cell(1, 1).paragraphs[0].text = "Total Timing: 7 Minutes (~1 min / slide + Q&A buffer)"

    for row in table.rows:
        for cell in row.cells:
            set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            for r in p.runs:
                r.font.name = "Arial"
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(71, 85, 105)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Slides Data
    slides_content = [
        {
            "slide_num": "Slide 1",
            "title": "Title & Team Introduction",
            "timing": "Allocated Time: ~1:00 min",
            "bullets": [
                ("Greeting & Title:", "Good morning / afternoon respected guides and panel members. We are presenting our project: Develop and Validate Optimised Multimodal Models for Automotive Use Cases."),
                ("Mentorship & Affiliation:", "This project is carried out under the valuable mentorship of our NVIDIA guides, Gaurav Tendolkar and Pradipt Trivedi, and our college project guide, Dr. Malathi S Y at KLE Technological University."),
                ("Team Introduction:", "Our project team consists of Saanvi Shetty, Raveesh Nayak, Ramya Rao, and myself, Shreyas Kulkarni."),
                ("The High-Level Motivation:", "Modern vehicles need continuous surround-view monitoring while parked, but running heavy Vision Transformers exhausts battery power. Our goal is to eliminate temporal frame redundancy to achieve low-power, high-accuracy surveillance perception on embedded automotive hardware.")
            ]
        },
        {
            "slide_num": "Slide 2",
            "title": "Problem Statement & Key Quality KPIs",
            "timing": "Allocated Time: ~1:00 min",
            "bullets": [
                ("Automotive Context:", "Our focus application is Automotive Surveillance Mode (ASM), monitoring the vehicle perimeter via four surround fisheye cameras."),
                ("The Computational Bottleneck:", "Vision Transformers like DINOv2 were trained on static images. Processing continuous 30 FPS video across 4 cameras causes battery drain and exceeds the thermal capacity of the embedded processor."),
                ("The Motion & Token Challenge:", "Between consecutive frames, moving pedestrians shift spatial positions. Static same-position token matching fails, which requires a motion-aware spatiotemporal tracking solution."),
                ("Quality KPIs & Benchmarks:", "We benchmark on the open-source VIRAT surveillance dataset, establishing localized action recognition via spatiotemporal tubelet pooling and actor trajectory tracking using Adapter-v2 + YOLO26.")
            ]
        },
        {
            "slide_num": "Slide 3",
            "title": "High-Level Optimised Multimodal Architecture",
            "timing": "Allocated Time: ~1:15 min",
            "bullets": [
                ("Camera Feeds & Temporal Decimation:", "Starting on the left: four 30 FPS fisheye feeds enter the pipeline. We first apply native Temporal Decimation to 5 FPS, cutting 83.3% of redundant frames immediately before model inference."),
                ("Shared DINOv2 Backbone (ViT-S/14):", "A single frozen 22M-parameter backbone processes the 5 FPS stream, outputting spatial patch tokens and multi-layer hidden states for all downstream tasks with zero redundant passes."),
                ("Head A — Actor Detection & Tracking:", "Adapter-v2 with YOLO detects pedestrians and vehicles, and a multi-frame Hungarian tracker builds continuous bounding-box actor trajectories across time."),
                ("Head B — VIRAT Action Head:", "We pool spatiotemporal tubelet features from the DINOv2 patch tokens and pass them to a 2-layer temporal self-attention Transformer to classify actions like walking, standing, or approaching vehicles."),
                ("Embedded PVA Hardware Target:", "Both streams converge into the Low-Power PVA processor and Threat Assessment engine, fulfilling our strict compute budget of < 10 GMACs per frame to preserve battery State of Charge.")
            ]
        },
        {
            "slide_num": "Slide 4",
            "title": "Multimodal Optimization & Cross-Scene Validation Strategy",
            "timing": "Allocated Time: ~1:00 min",
            "bullets": [
                ("Why Leave-One-Scene-Out (LOSO) Validation:", "To verify that our model generalizes to real-world automotive environments, we instituted a strict 3-Fold Leave-One-Scene-Out cross-validation protocol."),
                ("Cross-Scene Split Design:", "Each fold trains on two scenes and evaluates exclusively on a completely unseen third surveillance location with distinct camera angles, elevations, and lighting conditions."),
                ("Embedding Noise Injection & Augmentation:", "We injected Gaussian feature noise (sigma = 0.04) and applied 15% feature dropout to prevent over-reliance on static background features."),
                ("Regularization & Loss Calibration:", "We implemented 0.10 label smoothing and validation early stopping to guarantee calibrated, reliable prediction probabilities for vehicle threat alerts.")
            ]
        },
        {
            "slide_num": "Slide 5",
            "title": "Performance Benchmarking: Baseline vs. Optimized Multimodal Model",
            "timing": "Allocated Time: ~1:15 min",
            "bullets": [
                ("Top-Line Metric Highlights:", "A standard global Kinetics-400 linear probe achieves only 22.4% accuracy due to a complete lack of spatial actor awareness."),
                ("Compute Decimation Impact:", "Our native 5 FPS decimation provides an 83.3% forward-compute saving across the surround-view camera feeds."),
                ("Unseen Cross-Scene Generalization:", "On genuinely unseen surveillance scenes, our optimized multimodal model achieves 70.83% Top-1 Accuracy and a 69.16% Weighted F1-Score."),
                ("Surveillance Action Discrimination:", "On primary surveillance actions, our spatiotemporal tubelet pooling achieves 75.0% F1-score on Walking (81.8% recall) and 69.6% F1-score on Standing (72.7% precision)."),
                ("Key Takeaway:", "Coupling spatial actor bounding-box trajectories with temporal self-attention delivers strong, generalizable accuracy under diverse surveillance conditions.")
            ]
        },
        {
            "slide_num": "Slide 6",
            "title": "Project Milestones & Implementation Roadmap",
            "timing": "Allocated Time: ~1:00 min",
            "bullets": [
                ("Milestone 1 — Achieved:", "Integrated the open-source VIRAT surveillance benchmark, deployed the 5 FPS decimation pipeline, and developed the Tubelet Transformer action head."),
                ("Milestone 2 — Achieved:", "Completed 3-Fold LOSO cross-scene validation, confirmed 70.83% unseen Top-1 accuracy, and exported our calibrated production checkpoint (best.pth)."),
                ("Upcoming Phase — Low-Power PVA Target:", "We will implement motion-aware token pruning and frame differencing to dynamically skip static background tokens across frames, ensuring the active compute stays strictly below our < 10 GMACs per frame KPI."),
                ("Closing for Q&A:", "Thank you. We are now happy to answer any questions from our guides and the panel.")
            ]
        }
    ]

    for s in slides_content:
        # Slide Header Table
        s_tbl = doc.add_table(rows=1, cols=2)
        s_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        s_tbl.autofit = False
        s_tbl.rows[0].cells[0].width = Inches(4.5)
        s_tbl.rows[0].cells[1].width = Inches(2.0)

        set_cell_background(s_tbl.rows[0].cells[0], "0A2540") # Dark Navy
        set_cell_background(s_tbl.rows[0].cells[1], "0070BA") # Tech Blue
        set_cell_margins(s_tbl.rows[0].cells[0], top=100, bottom=100, left=140, right=140)
        set_cell_margins(s_tbl.rows[0].cells[1], top=100, bottom=100, left=140, right=140)

        # Title cell
        p_c1 = s_tbl.rows[0].cells[0].paragraphs[0]
        p_c1.paragraph_format.space_after = Pt(0)
        r_num = p_c1.add_run(f"{s['slide_num']}: ")
        r_num.font.name = "Arial"
        r_num.font.size = Pt(11)
        r_num.font.bold = True
        r_num.font.color.rgb = RGBColor(255, 255, 255)
        
        r_title = p_c1.add_run(s['title'])
        r_title.font.name = "Arial"
        r_title.font.size = Pt(11)
        r_title.font.color.rgb = RGBColor(255, 255, 255)

        # Timing cell
        p_c2 = s_tbl.rows[0].cells[1].paragraphs[0]
        p_c2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_c2.paragraph_format.space_after = Pt(0)
        r_time = p_c2.add_run(s['timing'])
        r_time.font.name = "Arial"
        r_time.font.size = Pt(10)
        r_time.font.bold = True
        r_time.font.color.rgb = RGBColor(255, 255, 255)

        # Content bullets
        for header, text in s['bullets']:
            p_b = doc.add_paragraph(style='List Bullet')
            p_b.paragraph_format.space_before = Pt(3)
            p_b.paragraph_format.space_after = Pt(4)
            p_b.paragraph_format.left_indent = Inches(0.3)

            r_h = p_b.add_run(f"{header} ")
            r_h.font.name = "Arial"
            r_h.font.size = Pt(10.5)
            r_h.font.bold = True
            r_h.font.color.rgb = RGBColor(10, 37, 64)

            r_t = p_b.add_run(f'"{text}"')
            r_t.font.name = "Arial"
            r_t.font.size = Pt(10.5)
            r_t.font.italic = True
            r_t.font.color.rgb = RGBColor(51, 65, 85)

        doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Delivery & Q&A Tips Box
    tip_tbl = doc.add_table(rows=1, cols=1)
    tip_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tip_tbl.autofit = False
    tip_tbl.rows[0].cells[0].width = Inches(6.5)
    set_cell_background(tip_tbl.rows[0].cells[0], "F1F5F9")
    set_cell_margins(tip_tbl.rows[0].cells[0], top=140, bottom=140, left=180, right=180)

    p_tip = tip_tbl.rows[0].cells[0].paragraphs[0]
    p_tip.paragraph_format.space_after = Pt(4)
    r_tip_h = p_tip.add_run("Presentation Timing & Delivery Tips:")
    r_tip_h.font.name = "Arial"
    r_tip_h.font.size = Pt(10)
    r_tip_h.font.bold = True
    r_tip_h.font.color.rgb = RGBColor(10, 37, 64)

    tips = [
        "Pacing: Keep a steady cadence of roughly 130-140 words per minute; each bullet corresponds to a single clear breath.",
        "Transitions: Pause 1-2 seconds when switching slides to let the panel visually digest the clean graphics.",
        "Buffer: The script covers ~6 minutes of active speaking, leaving a comfortable 1-minute buffer for introduction or questions."
    ]
    for tip in tips:
        p_t = tip_tbl.rows[0].cells[0].add_paragraph()
        p_t.paragraph_format.space_after = Pt(2)
        r_dot = p_t.add_run(f"• {tip}")
        r_dot.font.name = "Arial"
        r_dot.font.size = Pt(9.5)
        r_dot.font.color.rgb = RGBColor(71, 85, 105)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    print(f"Presentation script Word document created at: {out}")

if __name__ == "__main__":
    create_script_document()
