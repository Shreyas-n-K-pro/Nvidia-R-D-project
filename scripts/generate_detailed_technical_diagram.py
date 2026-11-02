#!/usr/bin/env python3
"""Generate a clean, high-resolution, perfectly spaced technical architecture diagram.

Fixes all overlapping text, increases vertical and horizontal clearance,
routes arrows cleanly through dedicated corridors, and completely removes
the 'KLE Tech x NVIDIA' mention from the header.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_clean_technical_diagram(output_path="assets/asm_detailed_technical_architecture.png"):
    # Extra-wide canvas (22 x 12.5 inches) for generous spacing and zero overlaps
    fig, ax = plt.subplots(figsize=(22, 12.5), dpi=300)
    bg_color = "#F8FAFC"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 12.5)
    ax.axis("off")

    def draw_panel(x, y, w, h, title, subtitle="", box_color="#FFFFFF", border_color="#CBD5E1", header_color="#0F172A"):
        """Draw an outer section container panel."""
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.18",
            facecolor=box_color,
            edgecolor=border_color,
            linewidth=1.5,
            linestyle="--",
            zorder=1
        )
        ax.add_patch(rect)
        ax.text(x + 0.25, y + h - 0.35, title.upper(), ha="left", va="center", fontsize=9.5, fontweight="bold", color=header_color, zorder=2)
        if subtitle:
            ax.text(x + w - 0.25, y + h - 0.35, subtitle, ha="right", va="center", fontsize=8.0, style="italic", color="#64748B", zorder=2)

    def draw_node(x, y, w, h, title, tag="", lines=None, shape_info="", box_color="#FFFFFF", border_color="#94A3B8", title_color="#0F172A", zorder=3):
        """Draw an operational execution node with strict internal padding and spacing."""
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.14",
            facecolor=box_color,
            edgecolor=border_color,
            linewidth=1.4,
            zorder=zorder
        )
        ax.add_patch(rect)
        
        # Tag pill in top right
        if tag:
            tag_w = len(tag) * 0.11 + 0.25
            t_box = patches.FancyBboxPatch(
                (x + w - tag_w - 0.15, y + h - 0.42), tag_w, 0.28,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=border_color,
                edgecolor="none",
                zorder=zorder+1
            )
            ax.add_patch(t_box)
            ax.text(x + w - 0.15 - tag_w/2, y + h - 0.28, tag, ha="center", va="center", fontsize=7.0, fontweight="bold", color="#FFFFFF", zorder=zorder+2)

        # Title
        ax.text(x + 0.2, y + h - 0.30, title, ha="left", va="center", fontsize=10.0, fontweight="bold", color=title_color, zorder=zorder+1)

        # Shape information badge
        y_curr = y + h - 0.65
        if shape_info:
            ax.text(x + 0.2, y_curr, f"Shape: {shape_info}", ha="left", va="center", fontsize=8.0, fontweight="bold", color="#2563EB", zorder=zorder+1)
            y_curr -= 0.34

        # Bullet lines with clean line-height
        if lines:
            for l in lines:
                ax.text(x + 0.2, y_curr, f"• {l}", ha="left", va="center", fontsize=7.6, color="#334155", zorder=zorder+1)
                y_curr -= 0.28

    def draw_arrow(x1, y1, x2, y2, label="", color="#475569", lw=1.6, ls="-", rad=0.0, label_offset=(0, 0.20)):
        """Draw clean connection arrows with isolated label tags."""
        connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->,head_width=0.32,head_length=0.42", color=color, lw=lw, linestyle=ls, connectionstyle=connectionstyle),
            zorder=6
        )
        if label:
            mx = (x1 + x2) / 2 + label_offset[0]
            my = (y1 + y2) / 2 + label_offset[1]
            ax.text(mx, my, label, ha="center", va="center", fontsize=7.2, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec=color, lw=0.8, alpha=0.95), zorder=8)

    # =========================================================================
    # 0. HEADER & SUBTITLE BANNER (Completely clean of KLE / NVIDIA)
    # =========================================================================
    ax.text(0.8, 11.95, "AUTOMOTIVE SURVEILLANCE MODE (ASM) — DETAILED TECHNICAL ARCHITECTURE", fontsize=16, fontweight="bold", color="#0F172A")
    ax.text(0.8, 11.60, "End-to-End Tensor Dimensions, Memory Caching Subsystems, Multi-Task Heads & Low-Power PVA Integration", fontsize=9.5, color="#64748B")

    # =========================================================================
    # STAGE 1: CAMERA INGESTION & TEMPORAL DECIMATION (Column 1: x = 0.8 .. 4.6)
    # =========================================================================
    draw_panel(0.8, 5.8, 3.8, 5.5, "Stage 1: Camera Ingestion & Decimation", "Surround Video Ingestion", "#F1F5F9", "#CBD5E1")

    draw_node(
        1.0, 8.8, 3.4, 2.1,
        "Surround Fisheye Feeds", "RAW INPUT",
        ["4 synchronized camera streams", "(Front, Rear, Left, Right)", "Native Rate: 1280 × 720 @ 30 FPS", "Pixel Format: RGB (uint8)"],
        shape_info="[4, 30, 3, 720, 1280]",
        box_color="#FFFFFF", border_color="#64748B"
    )

    draw_node(
        1.0, 6.2, 3.4, 2.1,
        "Temporal Decimation Filter", "5 FPS CUT",
        ["Uniform temporal sampling: Δt=200ms", "Cuts frame volume by 83.3%", "Sliding chunk buffer (N=40 frames)", "Squash resize to 640 × 640"],
        shape_info="[T_sub, 3, 640, 640]",
        box_color="#EFF6FF", border_color="#3B82F6", title_color="#1D4ED8"
    )

    draw_arrow(2.7, 8.8, 2.7, 8.3, label="30 FPS Stream", color="#3B82F6", label_offset=(0, 0))

    # =========================================================================
    # STAGE 2: SHARED DINOv2 BACKBONE (Column 2: x = 5.0 .. 9.2)
    # =========================================================================
    draw_panel(5.0, 5.8, 4.3, 5.5, "Stage 2: Shared Vision Transformer", "DINOv2 ViT-S/14", "#F0FDF4", "#BBF7D0")

    draw_node(
        5.2, 8.8, 3.9, 2.1,
        "Patch Embedding & Norm", "PATCHIFIED",
        ["Patch kernel: 14 × 14, Stride: 14", "Grid resolution: 45 × 45 = 2,025 patches", "ImageNet channel normalization (μ, σ)", "Prepends [CLS] token (+1)"],
        shape_info="[B, 2026, 384]",
        box_color="#FFFFFF", border_color="#22C55E", title_color="#15803D"
    )

    draw_node(
        5.2, 6.1, 3.9, 2.4,
        "12-Layer Transformer Blocks", "FROZEN",
        ["12 Self-Attention + MLP Blocks (D=384)", "Extracts Multi-Scale Hidden States:", "  • Layer 3: Low-level edges & textures", "  • Layer 6: Mid-level part representations", "  • Layer 9: High-level semantic parts", "  • Layer 12: Spatial patch tokens + CLS"],
        shape_info="HS: 4 × [B, 2026, 384]",
        box_color="#FFFFFF", border_color="#22C55E", title_color="#15803D"
    )

    draw_arrow(4.4, 7.25, 5.2, 9.8, label="[B, 3, 640, 640]", color="#2563EB", rad=-0.08, label_offset=(0, 0.15))
    draw_arrow(7.15, 8.8, 7.15, 8.5, color="#15803D")

    # =========================================================================
    # CACHING SUBSYSTEMS (Bottom Left: x = 0.8 .. 9.2)
    # =========================================================================
    draw_panel(0.8, 0.6, 8.5, 4.8, "System Caching Subsystems (Disk & RAM Pool)", "Storage & Memory Layer", "#F8FAFC", "#E2E8F0")

    draw_node(
        1.0, 1.0, 3.9, 3.9,
        "Feature & Detection Disk Cache", "PERSISTENT",
        ["Directory: data/virat_cache_5fps/*.pt", "Stores serialized tensors:", "  • patch_tokens: [T, 2025, 384]", "  • cls_token: [T, 384]", "  • detections: List[List[Dict]]", "  • metadata: FPS, timestamps, dims", "Atomic write & integrity checksum", "Eliminates duplicate 45 GMAC backbone passes", "Reduces re-training time from 45m to 22s"],
        shape_info=".pt binary (torch.save)",
        box_color="#FFFFFF", border_color="#6366F1", title_color="#4338CA"
    )

    draw_node(
        5.2, 1.0, 3.9, 3.9,
        "Runtime In-Memory Model Cache", "RAM POOL",
        ["Module: unified_perception.heads.detection", "Dictionary: _PRADIPT_IMPL_CACHE[key]", "Key: (adapter_type, device, ckpt_hash)", "Preserves instantiated models in RAM:", "  • SharedBackboneDetector instance", "  • YOLO26 weight tensors & CUDA contexts", "Zero initialization latency across video clips", "Thread-safe lazy-singleton initialization"],
        shape_info="Object Reference",
        box_color="#FFFFFF", border_color="#0EA5E9", title_color="#0369A1"
    )

    # Connections into/out of cache
    draw_arrow(7.15, 6.1, 3.0, 4.9, label="Offline Extraction Save", color="#6366F1", ls="--", rad=0.15, label_offset=(-0.2, 0.2))

    # =========================================================================
    # STAGE 3A: ACTOR DETECTION & TRACKING (Top Right: x = 9.8 .. 15.6)
    # =========================================================================
    draw_panel(9.8, 6.5, 6.0, 4.8, "Stage 3A: Actor Detection & Spatiotemporal Tracking", "Head A (ViT-Adapter + YOLO)", "#FFFBEB", "#FDE68A")

    draw_node(
        10.1, 9.0, 5.4, 1.9,
        "ViT-Adapter-v2 Multiscale Bridge", "ADAPTER",
        ["Spatial Prior: 3-scale detail convolutions", "Deformable Cross-Attention injection", "Reshapes ViT tokens to Feature Pyramids", "Pyramid Levels: P3 (80×80), P4 (40×40), P5 (20×20)"],
        shape_info="P3, P4, P5 Multi-Scale",
        box_color="#FFFFFF", border_color="#F59E0B", title_color="#B45309"
    )

    draw_node(
        10.1, 6.8, 5.4, 1.9,
        "YOLO26 Detector & Hungarian Tracker", "TRACKING",
        ["Anchor-free bounding box & class regression", "NMS confidence threshold: 0.25, IoU: 0.45", "Bipartite Hungarian Tracker links frames", "Outputs continuous trajectories: Track_ID, [x1, y1, x2, y2]"],
        shape_info="Tracks: [N_actors, T, 4]",
        box_color="#FFFFFF", border_color="#F59E0B", title_color="#B45309"
    )

    draw_arrow(9.1, 7.3, 10.1, 9.9, label="Hidden States [3,6,9,12]", color="#B45309", rad=0.08, label_offset=(0, 0.2))
    draw_arrow(12.8, 9.0, 12.8, 8.7, color="#B45309")

    # =========================================================================
    # STAGE 3B: SPATIOTEMPORAL TUBELET ACTION HEAD (Bottom Middle: x = 9.8 .. 15.6)
    # =========================================================================
    draw_panel(9.8, 0.6, 6.0, 5.5, "Stage 3B: Spatiotemporal Tubelet Action Head", "Head B (virat_action.py)", "#EEF2FF", "#C7D2FE")

    draw_node(
        10.1, 4.2, 5.4, 1.6,
        "Spatial Tubelet Pooler", "ROI POOLING",
        ["Projects normalized boxes [x1, y1, x2, y2] to 45×45 grid", "Slices actor-specific patch tokens per frame", "Average-pools spatial region into d=384 vector", "Builds spatiotemporal tubelet across T frames"],
        shape_info="Tubelet: [N_actors, T, 384]",
        box_color="#FFFFFF", border_color="#6366F1", title_color="#4338CA"
    )

    draw_node(
        10.1, 2.4, 5.4, 1.6,
        "Temporal Action Transformer", "2-LAYER ATTN",
        ["2-Layer Temporal Self-Attention Encoder (2.47M params)", "Captures dynamic rhythm: walking vs standing vs enter", "Gaussian feature noise (σ=0.04) & feature dropout (p=0.15)", "Linear projection + Label Smoothing (0.10)"],
        shape_info="Logits: [N_actors, 18]",
        box_color="#FFFFFF", border_color="#6366F1", title_color="#4338CA"
    )

    draw_node(
        10.1, 0.9, 5.4, 1.3,
        "LOSO Calibrated Action Classifier", "18 CLASSES",
        ["Classes: Walking (75% F1), Standing (69.6% F1), Loitering", "Unseen Scene Accuracy: 70.83% Top-1, 69.16% Weighted F1", "Checkpointed in assets/virat_outputs/best.pth"],
        shape_info="Probabilities: [0..1]",
        box_color="#FFFFFF", border_color="#6366F1", title_color="#4338CA"
    )

    # Connections to Head B (carefully routed without crossing labels)
    draw_arrow(9.1, 6.4, 10.1, 5.0, label="Patch Tokens [T, 2025, 384]", color="#4338CA", rad=-0.08, label_offset=(-0.1, -0.2))
    draw_arrow(12.8, 6.8, 12.8, 5.8, label="Actor Trajectories {Track_ID, BBoxes}", color="#B45309", lw=1.8, label_offset=(0, 0))
    draw_arrow(12.8, 4.2, 12.8, 4.0, color="#4338CA")
    draw_arrow(12.8, 2.4, 12.8, 2.2, color="#4338CA")

    # =========================================================================
    # STAGE 4: EMBEDDED PVA TARGET & THREAT ENGINE (Far Right: x = 16.2 .. 21.2)
    # =========================================================================
    draw_panel(16.2, 0.6, 5.0, 10.7, "Stage 4: Embedded Target & Threat Assessment", "Automotive Hardware Execution", "#F8FAFC", "#BAE6FD")

    draw_node(
        16.5, 8.4, 4.4, 2.5,
        "Spatiotemporal Threat Engine", "SECURITY LOGIC",
        ["Joint evaluation of Actor ID + Action + Proximity:", "  • Walking in transit  →  Normal (Log only)", "  • Standing near vehicle > 15s  →  Loitering Alert", "  • Approaching door + Touch  →  Active Break-in Alert", "Preserves vehicle perimeter safety"],
        shape_info="Threat Score [0..100]",
        box_color="#FFFFFF", border_color="#0284C7", title_color="#0369A1"
    )

    draw_node(
        16.5, 4.5, 4.4, 3.6,
        "Low-Power PVA Accelerator", "HARDWARE TARGET",
        ["Target: Programmable Vision Accelerator (PVA)", "Thermal Envelope: 1.5W – 5.0W ultra-low power", "Strict Hardware KPI: < 10 GMACs per frame", "Compute Profiling Breakdown:", "  • Backbone unpruned frame: ~42 GMACs (exceeds)", "  • 5 FPS decimation: 6x average power reduction", "  • Head A (YOLO) + Head B (Tubelet): < 1.2 GMACs", "Battery State of Charge (SoC) preservation target"],
        shape_info="< 10 GMACs / Frame",
        box_color="#F0F9FF", border_color="#0284C7", title_color="#0369A1"
    )

    draw_node(
        16.5, 0.9, 4.4, 3.3,
        "Upcoming Milestone: Token Pruning", "FRAME CACHING",
        ["Temporal Patch Differencing (Frame t vs t-1)", "Background static mask: eliminates >80% patches", "Reuses cached tokens for static background", "Active dynamic tokens forwarded: ~300 of 2,025", "Brings per-frame compute down to ~6.8 GMACs", "Guarantees full hardware compliance (< 10 GMACs)"],
        shape_info="~6.8 GMACs (< 10)",
        box_color="#FEFCE8", border_color="#EAB308", title_color="#A16207"
    )

    # Connections to Stage 4 (Clean curved routing)
    draw_arrow(15.5, 7.8, 16.5, 9.6, label="Detections & Tracks", color="#B45309", rad=0.08, label_offset=(-0.1, 0.2))
    draw_arrow(15.5, 1.5, 16.5, 8.8, label="Action Probabilities", color="#4338CA", rad=-0.22, label_offset=(0.1, -0.2))
    draw_arrow(18.7, 8.4, 18.7, 8.1, color="#0284C7")
    draw_arrow(18.7, 4.5, 18.7, 4.2, label="Next Milestone", color="#A16207", ls="--", label_offset=(0, 0))

    # =========================================================================
    # FOOTNOTE
    # =========================================================================
    ax.text(0.8, 0.25, "Solid Lines: Active Implemented Dataflow | Dashed Lines: Storage Cache & Upcoming Hardware Token Pruning | Tensor Dimensions: [Batch, Time, Channels, Height, Width]", fontsize=8.0, color="#64748B")

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Clean technical diagram generated at: {out}")

if __name__ == "__main__":
    create_clean_technical_diagram()
