#!/usr/bin/env python3
"""Generate a clean, modern, publication-quality architecture diagram for ASM."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram(output_path="assets/asm_architecture_diagram.png"):
    fig, ax = plt.subplots(figsize=(13, 6.8), dpi=300)
    bg_color = "#FFFFFF"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle, items=None, box_color="#FFFFFF", border_color="#CBD5E1", title_color="#0A2540"):
        # Shadow / Border
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12,rounding_size=0.18",
            facecolor=box_color,
            edgecolor=border_color,
            linewidth=1.8,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Header text
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", fontsize=11, fontweight="bold", color=title_color, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h - 0.65, subtitle, ha="center", va="center", fontsize=8.5, color="#64748B", style="italic", zorder=3)
        
        # Items / Bullet points
        if items:
            y_start = y + h - 1.05
            for it in items:
                ax.text(x + 0.25, y_start, f"• {it}", ha="left", va="center", fontsize=8.5, color="#334155", zorder=3)
                y_start -= 0.36

    def draw_arrow(x1, y1, x2, y2, label=None, label_pos="top", color="#0070BA"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", color=color, lw=2.0),
            zorder=4
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            offset_y = 0.22 if label_pos == "top" else -0.25
            ax.text(mx, my + offset_y, label, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0369A1", zorder=5)

    # 1. Inputs: Fisheye Surround View
    draw_box(
        0.5, 3.4, 2.2, 2.4,
        "Surround Cameras", "4x Fisheye Feeds",
        ["Front / Rear / Left / Right", "Surveillance Mode", "Raw Video @ 30 FPS"],
        box_color="#F8FAFC", border_color="#94A3B8", title_color="#0F172A"
    )

    # 2. Temporal Decimation (5 FPS)
    draw_box(
        3.2, 3.4, 2.3, 2.4,
        "Temporal Decimation", "Redundancy Filter",
        ["Native 5 FPS Resampling", "83.3% Redundancy Cut", "Batch Sliding Window"],
        box_color="#EFF6FF", border_color="#60A5FA", title_color="#1E40AF"
    )

    # Arrow 1 -> 2
    draw_arrow(2.7, 4.6, 3.2, 4.6, label="30 FPS")

    # 3. Shared DINOv2 Backbone
    draw_box(
        6.0, 3.0, 2.5, 3.2,
        "Shared DINOv2 Backbone", "ViT-S / 14 (Frozen)",
        ["Unified Representation", "22M Parameters", "Patch Tokens (45x45)", "Global CLS Token", "Zero Redundant Pass"],
        box_color="#F0FDF4", border_color="#4ADE80", title_color="#166534"
    )

    # Arrow 2 -> 3
    draw_arrow(5.5, 4.6, 6.0, 4.6, label="5 FPS Frames")

    # 4. Head A: Object & Actor Detection (Top Branch)
    draw_box(
        9.1, 4.6, 3.3, 1.8,
        "Head A: Actor Detection", "Adapter-v2 + YOLO26",
        ["Pedestrian & Vehicle Bounding Boxes", "Multi-Frame Hungarian Tracker", "Actor Trajectory Construction"],
        box_color="#FEF3C7", border_color="#FBBF24", title_color="#92400E"
    )

    # 5. Head B: Spatiotemporal Action Head (Bottom Branch)
    draw_box(
        9.1, 1.9, 3.3, 2.3,
        "Head B: VIRAT Action Head", "Tubelet Pooler + Transformer",
        ["ROI Spatiotemporal Tubelet Pooling", "2-Layer Temporal Self-Attention", "Regularized w/ Label Smoothing", "Actions: Walk, Stand, Enter, etc."],
        box_color="#EEF2FF", border_color="#818CF8", title_color="#3730A3"
    )

    # Arrows from Backbone to Heads
    draw_arrow(8.5, 5.0, 9.1, 5.5, label="Hidden States", color="#D97706")
    draw_arrow(8.5, 4.2, 9.1, 3.1, label="Patch Tokens", color="#4F46E5")

    # Tracking Trajectory Flow from Detection into Tubelet Pooler
    ax.annotate(
        "", xy=(10.75, 4.2), xytext=(10.75, 4.6),
        arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.35", color="#B45309", lw=1.8, linestyle="--"),
        zorder=4
    )
    ax.text(10.85, 4.4, "Actor Bounding Box Trajectories", ha="left", va="center", fontsize=7.5, color="#B45309", fontweight="bold", zorder=5)

    # 6. Low-Power PVA Output / Threat Alert
    draw_box(
        4.0, 0.4, 5.2, 1.2,
        "Low-Power PVA Target & Threat Assessment", "Real-Time Embedded Vehicle Decision",
        ["Compute Target: < 10 GMACs/frame  |  Battery SoC Preserved  |  Active Security Alert"],
        box_color="#F8FAFC", border_color="#0284C7", title_color="#0369A1"
    )

    # Arrows into Decision Output
    draw_arrow(10.75, 1.9, 9.2, 1.0, color="#0284C7")

    # Legend / Watermark header
    ax.text(0.5, 6.4, "Automotive Surveillance Mode (ASM) Architecture", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(0.5, 6.1, "KLE Technological University & SAMSUNG (SEED Collaboration) — Unified Perception Pipeline", fontsize=9.5, color="#64748B")

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Architecture diagram generated at: {out}")

if __name__ == "__main__":
    create_architecture_diagram()
