#!/usr/bin/env python3
"""Chunked feature extraction and actor detection for 5 FPS VIRAT videos.

Processes videos in temporal chunks (e.g. 40 frames = 8 seconds at 5 FPS)
to prevent RAM exhaustion on CPU/Windows, while extracting full-resolution
DINOv2 patch tokens and YOLO Adapter-v2 detections.
"""
import gc
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_perception.backbone import _load_backbone, BackboneOutput
from unified_perception.heads.detection import DetectionHead


def extract_video_chunked(
    mp4_path: str,
    output_pt: str,
    chunk_size: int = 40,
    image_size: int = 640,
    device: str = "cpu",
):
    print(f"\nProcessing {mp4_path} -> {output_pt}")
    start_time = time.time()

    cap = cv2.VideoCapture(mp4_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(f"Total frames: {total_frames}, FPS: {fps:.1f}, Chunk size: {chunk_size}")

    # Transforms
    raw_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
    ])
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    # Load models
    print("Loading DINOv2 backbone and DetectionHead...")
    backbone = _load_backbone("vits14", device)
    backbone.eval()
    det_head = DetectionHead(mode="shared")

    all_patch_tokens = []
    all_cls_tokens = []
    all_detections = []

    frame_idx = 0
    chunk_id = 0

    while True:
        chunk_frames = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            chunk_frames.append(torch.from_numpy(rgb).permute(2, 0, 1))

        if not chunk_frames:
            break

        chunk_id += 1
        n_chunk = len(chunk_frames)
        t_start = time.time()

        frames_tensor = torch.stack(chunk_frames, dim=0)  # [B, 3, H, W] uint8
        frames_raw = torch.stack([raw_tfm(f) for f in frames_tensor])
        frames_norm = torch.stack([normalize(f) for f in frames_raw])

        # Run backbone in mini-batches of 8
        batch_size_fwd = 8
        chunk_cls = []
        chunk_patch = []
        chunk_hidden: dict[int, list[torch.Tensor]] = {}

        with torch.no_grad():
            for b_start in range(0, n_chunk, batch_size_fwd):
                b_norm = frames_norm[b_start : b_start + batch_size_fwd].to(device)
                out_b = backbone(b_norm, output_hidden_states=True)
                lhs_b = out_b.last_hidden_state
                chunk_cls.append(lhs_b[:, 0].detach().cpu())
                chunk_patch.append(lhs_b[:, 1:].detach().cpu())

                for i, h in enumerate(out_b.hidden_states):
                    chunk_hidden.setdefault(i, []).append(h.detach().cpu())

        cat_cls = torch.cat(chunk_cls, dim=0)
        cat_patch = torch.cat(chunk_patch, dim=0)
        cat_hidden = {i: torch.cat(tensors, dim=0) for i, tensors in chunk_hidden.items()}

        # Build BackboneOutput for detection head
        chunk_bo = BackboneOutput(
            cls=cat_cls,
            patch_tokens=cat_patch,
            hidden_states=cat_hidden,
            frames=frames_raw,
            metadata={"source": mp4_path, "fps": fps},
        )

        det_res = det_head.predict(chunk_bo)
        if det_res.error:
            print(f"  Chunk {chunk_id}: Detection error: {det_res.error}")
            chunk_dets = [[] for _ in range(n_chunk)]
        else:
            chunk_dets = det_res.primary

        all_patch_tokens.append(cat_patch)
        all_cls_tokens.append(cat_cls)
        all_detections.extend(chunk_dets)

        frame_idx += n_chunk
        t_elapsed = time.time() - t_start
        print(f"  Chunk {chunk_id}: frames {frame_idx-n_chunk}..{frame_idx}/{total_frames} ({t_elapsed:.1f}s) - {len(chunk_dets)} frames detected")

        # Free chunk memory
        del chunk_frames, frames_tensor, frames_raw, frames_norm
        del chunk_cls, chunk_patch, chunk_hidden, cat_cls, cat_patch, cat_hidden, chunk_bo, det_res
        gc.collect()

    cap.release()

    final_patch = torch.cat(all_patch_tokens, dim=0)
    final_cls = torch.cat(all_cls_tokens, dim=0)

    print(f"Concatenated features: patch_tokens {final_patch.shape}, detections: {len(all_detections)} frames")

    cache_data = {
        "patch_tokens": final_patch,
        "cls": final_cls,
        "metadata": {"source": mp4_path, "fps": fps, "num_frames": frame_idx},
        "detections": all_detections,
        "video_name": Path(mp4_path).name,
    }

    out_p = Path(output_pt)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_data, out_p)
    file_size_mb = out_p.stat().st_size / (1024 * 1024)
    total_time = time.time() - start_time
    print(f"Saved {out_p.name} ({file_size_mb:.1f} MB) in {total_time:.1f}s")


def main():
    video_dir = Path("data/virat_5fps")
    cache_dir = Path("data/virat_cache_5fps")
    cache_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(video_dir.glob("*.mp4")))
    print(f"Found {len(videos)} videos in {video_dir}")

    for vid in videos:
        out_pt = cache_dir / f"{vid.stem}.pt"
        if out_pt.exists():
            print(f"Skipping existing cache: {out_pt.name}")
            continue
        extract_video_chunked(
            str(vid),
            str(out_pt),
            chunk_size=40,
            image_size=640,
            device="cpu",
        )

    print("\nAll 5 FPS extractions complete!")


if __name__ == "__main__":
    main()
