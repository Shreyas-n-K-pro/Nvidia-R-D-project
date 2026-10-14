"""
backbone — shared DINOv2 feature extractor for the sentry-mode perception pipeline.

Runs the frozen DINOv2 backbone ONCE per video, returning a BackboneOutput
(see perception_types.py) that all downstream heads consume. This is the
"call DINOv2 once" architecture — subsequent heads (detection, action,
alert, etc.) never re-run the backbone.

Design
------
- Model loaded lazily and cached at module level (singleton per size).
- Backbone stays frozen; no gradients.
- Extracts CLS + patch tokens from the last layer AND all requested
  intermediate layer hidden states. Heads pick what they need.
- Frames are the standard DINOv2 preprocessing: resize→center crop→normalize.

CLI
---
    python backbone.py --source render.mp4 --num-frames 8 --out /tmp/feats.pt

Programmatic
------------
    from skills.backbone import extract_features
    feats = extract_features(mp4_path, num_frames=8)  # BackboneOutput

Backbone alignment (as of 2026-07-03)
-------------------------------------
Default is ``vits14`` (DINOv2 ViT-S/14, 384-dim). This matches
Pradipt's Adapter-v2 checkpoint AND the retrained action-recog head.
Both heads consume features from the same backbone forward pass.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .perception_types import BackboneOutput


# =============================================================================
# Model cache (singleton per model_name)
# =============================================================================

_BACKBONE_CACHE: dict[str, Any] = {}


def _hf_name(model_name: str) -> str:
    """Resolve short size (vits14, vitb14, vitl14, vitg14) to HuggingFace repo id."""
    aliases = {
        "vits14": "facebook/dinov2-small",
        "vitb14": "facebook/dinov2-base",
        "vitl14": "facebook/dinov2-large",
        "vitg14": "facebook/dinov2-giant",
    }
    if model_name in aliases:
        return aliases[model_name]
    if model_name.startswith("facebook/") or "/" in model_name:
        return model_name
    raise ValueError(f"Unknown DINOv2 backbone: {model_name}")


def _load_backbone(model_name: str = "vits14", device: str | None = None):
    """Lazy-load + cache the DINOv2 backbone. Frozen, eval mode."""
    import torch
    from transformers import AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    key = f"{model_name}:{device}"
    if key in _BACKBONE_CACHE:
        return _BACKBONE_CACHE[key]

    hf_id = _hf_name(model_name)
    print(f"[backbone] Loading {model_name} ({hf_id}) on {device}", file=sys.stderr)
    model = AutoModel.from_pretrained(hf_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    _BACKBONE_CACHE[key] = model
    return model


# =============================================================================
# Frame sampling + preprocessing
# =============================================================================

def _sample_frames(
    mp4_path: str, num_frames: int, image_size: int = 640, stride: int = 1
):
    """Uniformly sample num_frames from an mp4, return BOTH raw and normalized frames.

    Returns:
        frames_raw:  [T, C, image_size, image_size] float32 in [0, 1]
                     (matches Pradipt's `img_tensor` — for the adapter's
                     detail branch, which needs low-level pixel values).
        frames_norm: [T, C, image_size, image_size] float32, ImageNet-normalized
                     (for the DINOv2 backbone forward).

    - num_frames = -1 : use ALL frames (stride applies).
    - num_frames > 0  : uniformly sample num_frames indices.
    - stride > 1      : subsamples by stride BEFORE selecting indices.
    """
    import numpy as np
    import torch
    from torchvision import transforms

    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(mp4_path, num_threads=1)
        total = len(vr)

        if num_frames == -1:
            indices = np.arange(0, total, stride)
        elif total < num_frames:
            indices = np.arange(total)
            indices = np.pad(indices, (0, num_frames - total), mode="edge")
        else:
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).permute(0, 3, 1, 2)  # [T, C, H, W] uint8
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(mp4_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames == -1:
            indices = list(range(0, total, stride))
        elif total < num_frames:
            indices = list(range(total)) + [total - 1] * (num_frames - total)
        else:
            indices = [int(x) for x in np.linspace(0, total - 1, num_frames, dtype=int)]

        idx_set = set(indices)
        frames_dict = {}
        cur_idx = 0
        while cap.isOpened() and len(frames_dict) < len(idx_set):
            ret, frame = cap.read()
            if not ret:
                break
            if cur_idx in idx_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_dict[cur_idx] = torch.from_numpy(rgb).permute(2, 0, 1)
            cur_idx += 1
        cap.release()
        fallback_frame = next(iter(frames_dict.values())) if frames_dict else torch.zeros(3, image_size, image_size, dtype=torch.uint8)
        frames_list = [frames_dict.get(i, fallback_frame) for i in indices]
        frames = torch.stack(frames_list, dim=0)  # [T, C, H, W] uint8

    # Resize (squash) + convert-to-float. This gives us the "raw [0,1]"
    # frames that the ViT-adapter's detail branch expects (Pradipt line 306).
    raw_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
    ])
    frames_raw = torch.stack([raw_tfm(f) for f in frames])

    # Normalize on top of raw for the DINOv2 backbone forward.
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    frames_norm = torch.stack([normalize(f) for f in frames_raw])

    return frames_raw, frames_norm


# =============================================================================
# Main API
# =============================================================================

def extract_features(
    mp4_path: str,
    num_frames: int = 8,
    model_name: str = "vits14",
    device: str | None = None,
    image_size: int = 640,   # Match Pradipt's adapter (built at imgsz=640)
    hidden_layer_indices: list[int] | None = None,
    keep_frames: bool = False,
    stride: int = 1,
) -> BackboneOutput:
    """Extract DINOv2 features from a single mp4.

    Parameters
    ----------
    mp4_path             : str
        Path to the input mp4.
    num_frames           : int
        Uniformly sample this many frames from the video (default: 8).
    model_name           : str
        Which DINOv2 to use — vits14/vitb14/vitl14/vitg14 or a HF repo id.
    device               : str
        "cuda" or "cpu". Default: auto-detect.
    image_size           : int
        Square DINOv2 preprocessing size. This package supports 640 (default,
        matches the detection adapter; DINOv2 patch grid 45x45) or 644 (a clean
        multiple of the ViT/14 patch size -> 46x46 grid). See ALLOWED_IMAGE_SIZES
        in orchestrator.py; the detection adapter is baked for 640.
    hidden_layer_indices : list[int] | None
        Which intermediate layer indices to keep. If None, keeps ALL layers
        (memory permitting). Pradipt's Adapter-v2 wants indices [3, 6, 9, 12]
        for ViT-S/14.
    keep_frames          : bool
        If True, keep the preprocessed frames tensor in the BackboneOutput.
        Useful for heads that need pixel-level input (e.g. rendering
        annotated overlays). Increases memory.

    Returns
    -------
    BackboneOutput
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _load_backbone(model_name, device)
    frames_raw, frames_norm = _sample_frames(mp4_path, num_frames, image_size, stride=stride)

    # Process in chunks (mini-batches) to prevent RAM exhaustion on CPU / GPU
    batch_size_fwd = 8
    num_total_frames = frames_norm.shape[0]
    all_cls = []
    all_patch_tokens = []
    all_hidden_states: dict[int, list[torch.Tensor]] = {}

    with torch.no_grad():
        for b_start in range(0, num_total_frames, batch_size_fwd):
            b_norm = frames_norm[b_start : b_start + batch_size_fwd].to(device)
            out_b = model(b_norm, output_hidden_states=True)
            lhs_b = out_b.last_hidden_state
            all_cls.append(lhs_b[:, 0].detach().cpu())
            all_patch_tokens.append(lhs_b[:, 1:].detach().cpu())

            for i, h in enumerate(out_b.hidden_states):
                if hidden_layer_indices is None or (0 <= i < len(out_b.hidden_states) and i in hidden_layer_indices):
                    all_hidden_states.setdefault(i, []).append(h.detach().cpu())

    cls = torch.cat(all_cls, dim=0) if all_cls else torch.empty(0, 384)
    patch_tokens = torch.cat(all_patch_tokens, dim=0) if all_patch_tokens else torch.empty(0, 2025, 384)
    hidden_states = {i: torch.cat(chunks, dim=0) for i, chunks in all_hidden_states.items()}

    # Source frame rate — used by the action head's sliding-window mode to
    # subsample to a target fps. Best-effort; 0.0 if unavailable.
    try:
        import decord as _decord
        fps = float(_decord.VideoReader(mp4_path, num_threads=1).get_avg_fps())
    except Exception:
        try:
            import cv2
            _cap = cv2.VideoCapture(mp4_path)
            fps = float(_cap.get(cv2.CAP_PROP_FPS) or 0.0)
            _cap.release()
        except Exception:
            fps = 0.0

    metadata = {
        "source":         mp4_path,
        "model_name":     model_name,
        "hf_id":          _hf_name(model_name),
        "num_frames":     num_frames,
        "image_size":     image_size,
        "feature_dim":    int(cls.shape[-1]),
        "num_patches":    int(patch_tokens.shape[1]),
        "num_layers":     len(out.hidden_states),
        "fps":            fps,
        "stride":         int(stride),   # CLS index i -> raw frame i*stride
    }

    return BackboneOutput(
        cls=cls,
        hidden_states=hidden_states,
        patch_tokens=patch_tokens,
        metadata=metadata,
        # frames = RAW [0,1] tensor (unnormalized). This is what Pradipt's
        # ViT-adapter detail branch consumes (see run_shared_backbone.py:316
        # — he passes `img_tensor` NOT `img_norm` to the adapter).
        frames=(frames_raw.detach().cpu() if keep_frames else None),
    )


# =============================================================================
# CLI
# =============================================================================

def _cli():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--source", required=True, help="Input mp4")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--model-name", default="vits14",
                   help="vits14 (default) | vitb14 | vitl14 | vitg14 | facebook/...")
    p.add_argument("--device", default="cuda")
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="Optional: only keep these hidden layer indices "
                        "(e.g. --layers 3 6 9 12 for Pradipt's adapter).")
    p.add_argument("--keep-frames", action="store_true",
                   help="Include the preprocessed frames tensor in the output.")
    p.add_argument("--out", help="Optional .pt path to save the BackboneOutput dict")
    args = p.parse_args()

    if not Path(args.source).exists():
        sys.exit(f"[backbone] source not found: {args.source}")

    feats = extract_features(
        args.source,
        num_frames=args.num_frames,
        model_name=args.model_name,
        device=args.device,
        hidden_layer_indices=args.layers,
        keep_frames=args.keep_frames,
    )
    print(f"[backbone] cls shape:       {tuple(feats.cls.shape)}")
    print(f"[backbone] patch_tokens:    {tuple(feats.patch_tokens.shape)}")
    print(f"[backbone] hidden_states:   {list(feats.hidden_states.keys())}")
    print(f"[backbone] feature_dim:     {feats.metadata['feature_dim']}")
    print(f"[backbone] backbone:        {feats.metadata['hf_id']}")

    if args.out:
        import torch
        torch.save(feats.__dict__, args.out)
        print(f"[backbone] saved -> {args.out}")


if __name__ == "__main__":
    _cli()
