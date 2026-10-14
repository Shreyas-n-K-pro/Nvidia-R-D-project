#!/usr/bin/env python3
"""Train the VIRAT temporal action head on VIRAT sample videos.

Phases:
  1. Extract & cache DINOv2 features + detections from sample videos
  2. Generate pseudo-labels from detection heuristics
  3. Train the TemporalActionClassifier
  4. Save checkpoint

Usage:
    python scripts/train_virat_head.py --data-dir data/virat_samples --output-dir assets/virat_outputs
    python scripts/train_virat_head.py --extract-only  # just cache features
    python scripts/train_virat_head.py --train-only    # train from cached features
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UnifiedPerception root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# =============================================================================
# Action Class Names (VIRAT 18 classes)
# =============================================================================

def load_virat_classes(path: Optional[str | Path] = None) -> list[str]:
    """Load VIRAT action class names from file or fallback to standard 18 classes."""
    if path is None:
        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "assets" / "virat_classes.txt"
    p = Path(path)
    if p.exists():
        classes = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if classes:
            return classes
    return [
        "Walking",
        "Running",
        "Standing",
        "Sitting",
        "Carrying_Object",
        "Gesturing",
        "Entering_Vehicle",
        "Exiting_Vehicle",
        "Opening_Trunk",
        "Closing_Trunk",
        "Loading_Vehicle",
        "Unloading_Vehicle",
        "Entering_Facility",
        "Exiting_Facility",
        "Talking",
        "Handing_Off_Object",
        "Riding_Bicycle",
        "Background",
    ]


VIRAT_CLASSES = load_virat_classes()


# =============================================================================
# Model & Tubelet Definitions (with fallback if virat_action module not found)
# =============================================================================

try:
    from unified_perception.heads.virat_action import (
        TemporalActionClassifier as _TemporalActionClassifier,
        TubeletPooler as _TubeletPooler,
        VIRAT_CLASSES as _IMPORTED_CLASSES,
    )
    if _IMPORTED_CLASSES:
        VIRAT_CLASSES = _IMPORTED_CLASSES

    # Wrap or alias
    class TemporalActionClassifier(_TemporalActionClassifier):
        def __init__(self, feature_dim: int = 384, num_classes: int = len(VIRAT_CLASSES), in_dim: Optional[int] = None, **kwargs):
            dim = in_dim if in_dim is not None else feature_dim
            super().__init__(feature_dim=dim, num_classes=num_classes, **kwargs)

        def classify(
            self,
            actor_features: Any,
            topk: int = 5,
            classes: Optional[list[str]] = None,
        ) -> list[tuple[str, float]]:
            """Classify a single actor tube [T, D] -> list[(label, prob)]."""
            import torch

            self.eval()
            cls_names = classes or VIRAT_CLASSES
            with torch.no_grad():
                inp = actor_features.unsqueeze(0) if actor_features.dim() == 2 else actor_features
                logits = self.forward(inp)
                probs = torch.softmax(logits, dim=-1)[0]
                k = min(topk, int(probs.shape[-1]))
                top_probs, top_idx = torch.topk(probs, k=k)
                return [(cls_names[int(i)], float(p)) for i, p in zip(top_idx, top_probs)]

    class TubeletPooler(_TubeletPooler):
        """Subclass of virat_action.TubeletPooler adding extract_tubes & __call__ helpers."""

        def __init__(
            self,
            feature_dim: int = 384,
            patch_grid_h: int = 45,
            patch_grid_w: int = 45,
            image_size: int = 640,
            iou_threshold: float = 0.25,
            target_tube_len: int = 8,
            min_tube_len: int = 3,
            **kwargs,
        ):
            super().__init__(
                feature_dim=feature_dim,
                patch_grid_h=patch_grid_h,
                patch_grid_w=patch_grid_w,
                image_size=image_size,
                iou_threshold=iou_threshold,
            )
            self.target_tube_len = target_tube_len
            self.min_tube_len = min_tube_len

        def extract_tubes(
            self,
            patch_tokens: Any,
            detections: list[list[dict]],
            image_size: int = 640,
            cls_tokens: Any = None,
            target_tube_len: Optional[int] = None,
        ) -> list[dict]:
            """Extract per-actor tubes and pooled features from detections and patch tokens."""
            import numpy as np
            import torch


            t_len = target_tube_len or self.target_tube_len

            # Extract vehicle boxes per frame
            vehicles_by_frame: dict[int, list[dict]] = {}
            for t, frame_dets in enumerate(detections):
                v_list = []
                if isinstance(frame_dets, list):
                    for d in frame_dets:
                        cname = str(d.get("class_name", "")).lower()
                        if cname in ("car", "truck", "bus", "vehicle", "van", "automobile"):
                            v_list.append(d)
                vehicles_by_frame[t] = v_list

            # Pool actor features using base pool_actor_features
            actor_feats = self.pool_actor_features(
                patch_tokens=patch_tokens,
                detections_per_frame=detections,
                image_size=image_size,
            )

            trajectories = getattr(self, "actor_trajectories", {})
            tubes: list[dict] = []

            for actor_id, raw_feat in actor_feats.items():
                traj = trajectories.get(actor_id, [])
                frames = [t for t, _ in traj]
                boxes = [box for _, box in traj]

                if len(frames) < self.min_tube_len and len(actor_feats) > 1:
                    continue

                # Standardize to t_len frames (subsample or pad)
                K = raw_feat.shape[0]
                dim = raw_feat.shape[-1]
                if K == t_len:
                    feat = raw_feat
                elif K > t_len:
                    idx = np.linspace(0, K - 1, t_len, dtype=int)
                    feat = raw_feat[idx]
                else:
                    pad = raw_feat[-1:].expand(t_len - K, dim)
                    feat = torch.cat([raw_feat, pad], dim=0)

                tubes.append({
                    "track_id": actor_id,
                    "frames": frames,
                    "boxes": boxes,
                    "features": feat.float(),  # [t_len, D]
                    "vehicles_by_frame": {t: vehicles_by_frame.get(t, []) for t in frames},
                })

            return tubes

        def __call__(
            self,
            patch_tokens: Optional[torch.Tensor],
            detections: list[list[dict]],
            image_size: int = 640,
            cls_tokens: Optional[torch.Tensor] = None,
        ) -> list[dict]:
            return self.extract_tubes(
                patch_tokens=patch_tokens,
                detections=detections,
                image_size=image_size,
                cls_tokens=cls_tokens,
            )


except (ImportError, ModuleNotFoundError, AttributeError):

    import torch
    import torch.nn as nn

    class TemporalActionClassifier(nn.Module):
        """Temporal action classifier over pooled actor tube features [B, T, D]."""

        def __init__(
            self,
            feature_dim: int = 384,
            num_classes: int = len(VIRAT_CLASSES),
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
            max_seq_len: int = 256,
            in_dim: Optional[int] = None,
            **kwargs,
        ):
            super().__init__()
            dim = in_dim if in_dim is not None else feature_dim
            self.feature_dim = dim
            self.num_classes = num_classes
            self.max_seq_len = max_seq_len

            self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, dim))
            nn.init.trunc_normal_(self.positional_encoding, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            import torch.nn.functional as F

            if x.dim() == 2:
                x = x.unsqueeze(0)

            B, T, D = x.shape
            max_pos = self.positional_encoding.shape[1]
            if T <= max_pos:
                pos = self.positional_encoding[:, :T, :]
            else:
                pos = F.interpolate(
                    self.positional_encoding.transpose(1, 2),
                    size=T,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            x_enc = x + pos
            out = self.transformer(x_enc)
            pooled = out.mean(dim=1)
            logits = self.classifier(pooled)
            return logits

    class TubeletPooler:
        """Extracts spatiotemporal actor tubes from detection boxes and pools DINOv2 features."""

        def __init__(
            self,
            feature_dim: int = 384,
            patch_grid_h: int = 45,
            patch_grid_w: int = 45,
            image_size: int = 640,
            iou_threshold: float = 0.25,
            **kwargs,
        ):
            self.feature_dim = feature_dim
            self.patch_grid_h = patch_grid_h
            self.patch_grid_w = patch_grid_w
            self.image_size = image_size
            self.iou_threshold = iou_threshold
            self.actor_trajectories: dict[int, list[tuple[int, list[float]]]] = {}
            self.last_bboxes: dict[int, list[float]] = {}


        @staticmethod
        def box_iou(b1: list[float], b2: list[float]) -> float:
            """Compute 2D bounding box IoU."""
            ix1 = max(b1[0], b2[0])
            iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2])
            iy2 = min(b1[3], b2[3])
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
            a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
            union = a1 + a2 - inter
            return inter / union if union > 0 else 0.0

        def track_persons(
            self,
            all_frames_dets: list[list[dict]],
        ) -> tuple[list[dict], dict[int, list[dict]]]:
            """Track person detections across frames and collect vehicle boxes per frame."""
            vehicles_by_frame: dict[int, list[dict]] = {}
            active_tracks: list[dict] = []
            finished_tracks: list[dict] = []
            next_track_id = 0

            for t, frame_dets in enumerate(all_frames_dets):
                persons = []
                vehicles = []
                for d in frame_dets:
                    cname = str(d.get("class_name", "")).lower()
                    if cname in ("person", "pedestrian"):
                        persons.append(d)
                    elif cname in ("car", "truck", "bus", "vehicle", "van", "automobile"):
                        vehicles.append(d)

                vehicles_by_frame[t] = vehicles

                matched_track_indices = set()
                matched_det_indices = set()

                if active_tracks and persons:
                    # Greedy IoU matching
                    iou_pairs = []
                    for ti, track in enumerate(active_tracks):
                        last_box = track["boxes"][-1]
                        for di, p in enumerate(persons):
                            iou = self.box_iou(last_box, p["bbox"])
                            if iou >= self.iou_threshold:
                                iou_pairs.append((iou, ti, di))

                    iou_pairs.sort(key=lambda x: x[0], reverse=True)
                    for _, ti, di in iou_pairs:
                        if ti not in matched_track_indices and di not in matched_det_indices:
                            matched_track_indices.add(ti)
                            matched_det_indices.add(di)
                            active_tracks[ti]["frames"].append(t)
                            active_tracks[ti]["boxes"].append(persons[di]["bbox"])
                            active_tracks[ti]["confs"].append(persons[di].get("conf", 1.0))
                            active_tracks[ti]["lost"] = 0

                # Increment lost counter for unmatched tracks
                for ti, track in enumerate(active_tracks):
                    if ti not in matched_track_indices:
                        track["lost"] += 1

                # Start new tracks for unmatched person detections
                for di, p in enumerate(persons):
                    if di not in matched_det_indices:
                        active_tracks.append({
                            "track_id": next_track_id,
                            "frames": [t],
                            "boxes": [p["bbox"]],
                            "confs": [p.get("conf", 1.0)],
                            "lost": 0,
                        })
                        next_track_id += 1

                # Move long-lost tracks to finished
                remaining_active = []
                for track in active_tracks:
                    if track["lost"] > self.max_lost:
                        finished_tracks.append(track)
                    else:
                        remaining_active.append(track)
                active_tracks = remaining_active

            finished_tracks.extend(active_tracks)

            # Filter tracks by min_tube_len
            valid_tracks = [tr for tr in finished_tracks if len(tr["frames"]) >= self.min_tube_len]
            if not valid_tracks and finished_tracks:
                valid_tracks = finished_tracks

            return valid_tracks, vehicles_by_frame

        def pool_actor_tube_features(
            self,
            patch_tokens: Optional[torch.Tensor],
            cls_tokens: Optional[torch.Tensor],
            track: dict,
            image_size: int = 640,
        ) -> torch.Tensor:
            """Pool spatial patch tokens for the actor's bounding box across tube frames."""
            import numpy as np

            frames = track["frames"]
            boxes = track["boxes"]
            K = len(frames)
            dim = 384
            if patch_tokens is not None and patch_tokens.numel() > 0:
                dim = patch_tokens.shape[-1]
            elif cls_tokens is not None and cls_tokens.numel() > 0:
                dim = cls_tokens.shape[-1]

            frame_features = []

            # Check if patch grid can be reshaped
            has_patch_grid = False
            grid_h = grid_w = 0
            if patch_tokens is not None and patch_tokens.numel() > 0:
                N = patch_tokens.shape[1]
                gh = gw = int(math.isqrt(N))
                if gh * gw == N:
                    has_patch_grid = True
                    grid_h, grid_w = gh, gw

            for i, t in enumerate(frames):
                box = boxes[i]
                if has_patch_grid and t < patch_tokens.shape[0]:
                    x1, y1, x2, y2 = box
                    # Map box coordinates to patch grid
                    c1 = max(0, min(grid_w - 1, int(x1 / image_size * grid_w)))
                    c2 = max(c1 + 1, min(grid_w, int(math.ceil(x2 / image_size * grid_w))))
                    r1 = max(0, min(grid_h - 1, int(y1 / image_size * grid_h)))
                    r2 = max(r1 + 1, min(grid_h, int(math.ceil(y2 / image_size * grid_h))))

                    grid = patch_tokens[t].view(grid_h, grid_w, dim)
                    actor_feat = grid[r1:r2, c1:c2, :].mean(dim=(0, 1))
                elif cls_tokens is not None and t < cls_tokens.shape[0]:
                    actor_feat = cls_tokens[t]
                elif patch_tokens is not None and t < patch_tokens.shape[0]:
                    actor_feat = patch_tokens[t].mean(dim=0)
                else:
                    actor_feat = torch.zeros(dim, dtype=torch.float32)

                frame_features.append(actor_feat.cpu())

            if not frame_features:
                return torch.zeros(self.target_tube_len, dim, dtype=torch.float32)

            stacked = torch.stack(frame_features, dim=0)  # [K, D]

            # Uniformly subsample or pad to target_tube_len
            T_target = self.target_tube_len
            if K == T_target:
                return stacked
            if K > T_target:
                idx = np.linspace(0, K - 1, T_target, dtype=int)
                return stacked[idx]
            pad = stacked[-1:].expand(T_target - K, dim)
            return torch.cat([stacked, pad], dim=0)

        def extract_tubes(
            self,
            patch_tokens: Optional[torch.Tensor],
            detections: list[list[dict]],
            image_size: int = 640,
            cls_tokens: Optional[torch.Tensor] = None,
        ) -> list[dict]:
            """Extract spatiotemporal actor tubes and their pooled features."""
            tracks, vehicles_by_frame = self.track_persons(detections)
            tubes = []

            for tr in tracks:
                features = self.pool_actor_tube_features(
                    patch_tokens=patch_tokens,
                    cls_tokens=cls_tokens,
                    track=tr,
                    image_size=image_size,
                )
                tubes.append({
                    "track_id": tr["track_id"],
                    "frames": tr["frames"],
                    "boxes": tr["boxes"],
                    "confs": tr["confs"],
                    "features": features,  # [target_tube_len, D]
                    "vehicles_by_frame": {t: vehicles_by_frame.get(t, []) for t in tr["frames"]},
                })

            return tubes

        def __call__(
            self,
            patch_tokens: Optional[torch.Tensor],
            detections: list[list[dict]],
            image_size: int = 640,
            cls_tokens: Optional[torch.Tensor] = None,
        ) -> list[dict]:
            return self.extract_tubes(
                patch_tokens=patch_tokens,
                detections=detections,
                image_size=image_size,
                cls_tokens=cls_tokens,
            )


# =============================================================================
# Training Sample Container
# =============================================================================

class TrainingSample(dict):
    """Training sample container supporting both dict access and tuple unpacking."""

    def __iter__(self):
        yield self["features"]
        yield self["label"]


# =============================================================================
# Phase 1: Feature Extraction and Caching
# =============================================================================

def extract_and_cache(
    video_dir: str | Path,
    cache_dir: str | Path,
    device: str,
    image_size: int = 640,
    stride: int = 6,
) -> bool:
    """Extract DINOv2 features and detections for all sample mp4 videos and cache to disk.

    Args:
        video_dir: Directory containing input mp4 videos.
        cache_dir: Directory to store cached .pt feature files.
        device: 'cuda' or 'cpu'.
        image_size: Square size for backbone extraction (default: 640).
        stride: Temporal frame sampling stride (default: 6).

    Returns:
        True if extraction completed successfully, False otherwise.
    """
    import torch

    v_dir = Path(video_dir)
    c_dir = Path(cache_dir)
    c_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(v_dir.glob("*.mp4")) + list(v_dir.glob("*.avi")) + list(v_dir.glob("*.mkv")))
    if not video_files:
        print(f"[extract] Warning: No video files found in {v_dir}")
        return False

    print(f"[extract] Found {len(video_files)} video(s) in {v_dir}")
    print(f"[extract] Caching features to {c_dir} (device={device}, stride={stride}, image_size={image_size})")

    try:
        from unified_perception.backbone import extract_features
    except Exception as e:
        print(f"[extract] Error importing backbone extractor: {e}")
        print("[extract] Note: transformers and/or decord might not be installed in this environment.")
        return False

    # Detection head
    det_head = None
    try:
        from unified_perception.heads.detection import DetectionHead
        det_head = DetectionHead(mode="shared")
    except Exception as e:
        print(f"[extract] Warning: Could not initialize DetectionHead ({e}). Empty detections will be cached.")

    success_count = 0
    for idx, mp4 in enumerate(video_files, 1):
        stem = mp4.stem
        cache_file = c_dir / f"{stem}.pt"

        if cache_file.exists():
            print(f"[extract] [{idx}/{len(video_files)}] Cache exists: {cache_file.name}. Skipping.")
            success_count += 1
            continue

        print(f"[extract] [{idx}/{len(video_files)}] Extracting features: {mp4.name} ...")
        try:
            features = extract_features(
                str(mp4),
                num_frames=-1,
                stride=stride,
                keep_frames=True,
                device=device,
                image_size=image_size,
            )
        except Exception as e:
            print(f"[extract] Error extracting features for {mp4.name}: {e}")
            continue

        # Run detection head on shared features
        detections = []
        if det_head is not None:
            try:
                det_res = det_head.predict(features)
                if det_res and not det_res.error:
                    detections = det_res.primary
                else:
                    err_msg = det_res.error if det_res else "unknown error"
                    print(f"[extract] Detection head returned error for {mp4.name}: {err_msg}")
            except Exception as e:
                print(f"[extract] Detection inference failed for {mp4.name}: {e}")

        # Save to cache
        cache_data = {
            "patch_tokens": features.patch_tokens.detach().cpu() if features.patch_tokens is not None else None,
            "cls": features.cls.detach().cpu() if features.cls is not None else None,
            "metadata": features.metadata if features.metadata else {},
            "detections": detections,
            "video_name": mp4.name,
        }

        try:
            torch.save(cache_data, cache_file)
            print(f"[extract] Saved cache to {cache_file} ({cache_file.stat().st_size / (1024*1024):.2f} MB)")
            success_count += 1
        except Exception as e:
            print(f"[extract] Failed saving cache for {mp4.name}: {e}")

    print(f"[extract] Extraction complete: {success_count}/{len(video_files)} video caches ready.")
    return success_count > 0


# =============================================================================
# Phase 2: Pseudo-Label Generation
# =============================================================================

def _compute_iou(b1: list[float], b2: list[float]) -> float:
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _center_distance(b1: list[float], b2: list[float]) -> float:
    """Euclidean distance between bbox centers."""
    c1x, c1y = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
    c2x, c2y = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
    return math.hypot(c1x - c2x, c1y - c2y)


def assign_pseudo_label_to_tube(tube: dict, image_size: int = 640) -> str:
    """Assign a VIRAT action pseudo-label to an actor tube using spatial-temporal heuristics.

    Heuristics:
      1. Vehicle interaction:
         - Person bbox overlaps significantly with vehicle (car/truck/bus)
         - If IoU with vehicle increases over time -> 'Entering_Vehicle'
         - If IoU with vehicle decreases over time -> 'Exiting_Vehicle'
         - If person near vehicle rear -> 'Opening_Trunk' or 'Loading_Vehicle'
         - Otherwise -> 'Standing'
      2. Motion & Speed:
         - Significant displacement across frames -> 'Walking' or 'Running'
         - Mostly stationary -> 'Standing'
      3. Aspect ratio:
         - Squarish / compressed height -> 'Sitting'
      4. Default:
         - 'Background'
    """
    boxes = tube.get("boxes", [])
    frames = tube.get("frames", [])
    vehicles_by_frame = tube.get("vehicles_by_frame", {})

    if not boxes:
        return "Background"

    num_frames = len(boxes)

    # ---------------------------------------------------------
    # 1. Vehicle Interaction Heuristics
    # ---------------------------------------------------------
    # Find vehicles overlapping or close to the person
    interacting_vehicle_ious: list[float] = []
    interacting_vehicle_dists: list[float] = []
    is_near_rear = False

    for i, t in enumerate(frames):
        p_box = boxes[i]
        v_list = vehicles_by_frame.get(t, [])
        max_iou_in_frame = 0.0
        min_dist_in_frame = float("inf")
        closest_v_box = None

        for v in v_list:
            v_box = v["bbox"]
            iou = _compute_iou(p_box, v_box)
            dist = _center_distance(p_box, v_box)
            if iou > max_iou_in_frame:
                max_iou_in_frame = iou
            if dist < min_dist_in_frame:
                min_dist_in_frame = dist
                closest_v_box = v_box

        interacting_vehicle_ious.append(max_iou_in_frame)
        interacting_vehicle_dists.append(min_dist_in_frame if min_dist_in_frame != float("inf") else 999.0)

        # Check if person is near vehicle rear / bumper area
        if closest_v_box is not None and max_iou_in_frame > 0.05:
            vx1, vy1, vx2, vy2 = closest_v_box
            vw = vx2 - vx1
            vh = vy2 - vy1
            pcx = (p_box[0] + p_box[2]) / 2.0
            pcy = (p_box[1] + p_box[3]) / 2.0
            # Near boundary of vehicle (top, bottom, or sides)
            if pcy > vy2 - 0.25 * vh or pcy < vy1 + 0.25 * vh or pcx > vx2 - 0.25 * vw or pcx < vx1 + 0.25 * vw:
                is_near_rear = True

    max_overlap = max(interacting_vehicle_ious) if interacting_vehicle_ious else 0.0

    if max_overlap >= 0.08 or (min(interacting_vehicle_dists) < 0.15 * image_size and max_overlap > 0.02):
        if num_frames >= 2:
            half = max(1, num_frames // 2)
            first_half_iou = sum(interacting_vehicle_ious[:half]) / half
            second_half_iou = sum(interacting_vehicle_ious[half:]) / (num_frames - half)
            delta_iou = second_half_iou - first_half_iou

            first_half_dist = sum(interacting_vehicle_dists[:half]) / half
            second_half_dist = sum(interacting_vehicle_dists[half:]) / (num_frames - half)
            delta_dist = second_half_dist - first_half_dist

            # Moving toward vehicle -> Entering
            if delta_iou > 0.03 or delta_dist < -10.0:
                return "Entering_Vehicle"

            # Moving away from vehicle -> Exiting
            if delta_iou < -0.03 or delta_dist > 10.0:
                return "Exiting_Vehicle"

        if is_near_rear:
            # Alternate between Opening_Trunk and Loading_Vehicle based on track id
            track_id = tube.get("track_id", 0)
            return "Opening_Trunk" if track_id % 2 == 0 else "Loading_Vehicle"

        return "Standing"

    # ---------------------------------------------------------
    # 2. Motion / Speed / Aspect Ratio Heuristics (no vehicle)
    # ---------------------------------------------------------
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
    displacements = [
        math.hypot(centers[k][0] - centers[k - 1][0], centers[k][1] - centers[k - 1][1])
        for k in range(1, len(centers))
    ]
    total_dist = sum(displacements)
    speed = total_dist / max(1, len(displacements))  # pixels per frame

    heights = [b[3] - b[1] for b in boxes]
    widths = [b[2] - b[0] for b in boxes]
    avg_h = sum(heights) / len(heights) if heights else 50.0
    avg_w = sum(widths) / len(widths) if widths else 20.0
    aspect_ratio = avg_h / max(1.0, avg_w)
    norm_speed = speed / max(1.0, avg_h)

    # Sitting check: squarish aspect ratio + stationary or very slow
    if aspect_ratio < 1.35 and speed < 2.0:
        return "Sitting"

    # Speed check
    if norm_speed > 0.16 or speed > 7.5:
        return "Running"
    if norm_speed > 0.03 or speed > 1.2:
        return "Walking"
    if speed <= 1.2:
        return "Standing"

    return "Background"


def generate_pseudo_labels(cache_dir: str | Path) -> list[dict]:
    """Load cached features and detections, run TubeletPooler, and assign pseudo-labels.

    Args:
        cache_dir: Directory containing cached .pt files.

    Returns:
        List of TrainingSample dicts, each containing:
          - 'features': torch.Tensor [T, D]
          - 'label': int (class index)
          - 'label_name': str
          - 'video': str
          - 'track_id': int
    """
    import torch

    c_dir = Path(cache_dir)
    cache_files = sorted(c_dir.glob("*.pt"))
    if not cache_files:
        print(f"[pseudo-label] Warning: No cached .pt files found in {c_dir}")
        return []

    print(f"[pseudo-label] Loading cached features from {len(cache_files)} file(s) in {c_dir} ...")
    pooler = TubeletPooler()
    all_samples: list[dict] = []
    class_to_idx = {c: i for i, c in enumerate(VIRAT_CLASSES)}

    for cache_path in cache_files:
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[pseudo-label] Failed to load {cache_path.name}: {e}")
            continue

        patch_tokens = data.get("patch_tokens")
        cls_tokens = data.get("cls")
        metadata = data.get("metadata", {})
        detections = data.get("detections", [])
        image_size = metadata.get("image_size", 640)
        video_name = data.get("video_name", cache_path.stem)

        # Extract actor tubes
        tubes = pooler.extract_tubes(
            patch_tokens=patch_tokens,
            detections=detections,
            image_size=image_size,
            cls_tokens=cls_tokens,
        )

        print(f"[pseudo-label] Video {video_name}: extracted {len(tubes)} actor tube(s).")

        for tube in tubes:
            label_name = assign_pseudo_label_to_tube(tube, image_size=image_size)
            label_idx = class_to_idx.get(label_name, class_to_idx.get("Background", 17))

            sample = TrainingSample({
                "features": tube["features"].float(),  # [target_tube_len, D]
                "label": label_idx,
                "label_name": label_name,
                "video": video_name,
                "track_id": tube["track_id"],
            })
            all_samples.append(sample)

    # Robust fallback: If no tubes were detected, generate background samples from CLS tokens
    if not all_samples:
        print("[pseudo-label] Notice: No actor tubes extracted from detections. Generating background samples from CLS.")
        for cache_path in cache_files:
            try:
                data = torch.load(cache_path, map_location="cpu", weights_only=False)
                cls_tokens = data.get("cls")
                if cls_tokens is not None and cls_tokens.numel() > 0:
                    T_all, D = cls_tokens.shape
                    target_len = pooler.target_tube_len
                    # Slice into non-overlapping windows of target_len
                    for start in range(0, T_all - target_len + 1, target_len):
                        chunk = cls_tokens[start : start + target_len].float()
                        all_samples.append(TrainingSample({
                            "features": chunk,
                            "label": class_to_idx.get("Background", 17),
                            "label_name": "Background",
                            "video": cache_path.stem,
                            "track_id": start,
                        }))
            except Exception as e:
                print(f"[pseudo-label] Fallback sample creation failed for {cache_path.name}: {e}")

    # Summary of pseudo-labels
    from collections import Counter
    label_counts = Counter(s["label_name"] for s in all_samples)
    print(f"[pseudo-label] Generated {len(all_samples)} training sample(s) total.")
    for lbl, cnt in label_counts.most_common():
        print(f"  - {lbl}: {cnt}")

    return all_samples


# =============================================================================
# Phase 3: Classifier Training
# =============================================================================

def train_classifier(
    samples: list[dict],
    output_dir: str | Path,
    num_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 8,
    device: str = "cpu",
) -> Optional[Path]:
    """Train the TemporalActionClassifier on pseudo-labeled actor tubes.

    Args:
        samples: List of training sample dicts.
        output_dir: Directory to save best.pth and configuration.
        num_epochs: Training epochs (default: 50).
        lr: Learning rate (default: 1e-3).
        batch_size: Batch size (default: 8).
        device: 'cuda' or 'cpu'.

    Returns:
        Path to best.pth checkpoint if saved, None otherwise.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    if not samples:
        print("[train] Error: No samples provided for training.")
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class TubeletDataset(Dataset):
        def __init__(self, data_samples: list[dict]):
            self.data = data_samples

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            s = self.data[idx]
            feat = s["features"]
            if not isinstance(feat, torch.Tensor):
                feat = torch.tensor(feat, dtype=torch.float32)
            else:
                feat = feat.float()
            label = torch.tensor(int(s["label"]), dtype=torch.long)
            return feat, label

    dataset = TubeletDataset(samples)
    eff_batch_size = max(1, min(batch_size, len(dataset)))
    dataloader = DataLoader(
        dataset,
        batch_size=eff_batch_size,
        shuffle=(len(dataset) > 1),
        drop_last=False,
    )

    in_dim = int(samples[0]["features"].shape[-1])
    num_classes = len(VIRAT_CLASSES)

    print(f"[train] Initializing TemporalActionClassifier (in_dim={in_dim}, num_classes={num_classes}) on {device}")
    model = TemporalActionClassifier(in_dim=in_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs),
        eta_min=lr * 0.01,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state_dict = None

    print(f"[train] Starting training for {num_epochs} epoch(s) (samples={len(samples)}, batch_size={eff_batch_size}, lr={lr})")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total_items = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total_items += batch_x.size(0)

        scheduler.step()
        epoch_loss = total_loss / max(1, total_items)
        epoch_acc = (correct / max(1, total_items)) * 100.0
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch [{epoch:3d}/{num_epochs:3d}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:5.1f}% - LR: {current_lr:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best checkpoint
    best_ckpt_path = out_dir / "best.pth"
    feat_dim = getattr(model, "feature_dim", getattr(model, "in_dim", in_dim))
    hidden_dim = getattr(model, "hidden_dim", feat_dim)
    n_layers = getattr(model, "num_layers", 2)

    checkpoint = {
        "model_state_dict": best_state_dict or model.state_dict(),
        "state_dict": best_state_dict or model.state_dict(),
        "classes": VIRAT_CLASSES,
        "num_classes": num_classes,
        "in_dim": feat_dim,
        "feature_dim": feat_dim,
        "hidden_dim": hidden_dim,
        "num_layers": n_layers,
        "epoch": best_epoch,
        "loss": best_loss,
        "architecture": "TemporalActionClassifier",
    }
    torch.save(checkpoint, best_ckpt_path)
    print(f"[train] Best model checkpoint saved to: {best_ckpt_path} (epoch {best_epoch}, loss {best_loss:.4f})")

    # Save training config
    config = {
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "batch_size": eff_batch_size,
        "device": str(device),
        "num_samples": len(samples),
        "in_dim": feat_dim,
        "feature_dim": feat_dim,
        "hidden_dim": hidden_dim,
        "num_layers": n_layers,
        "best_epoch": best_epoch,
        "best_loss": round(float(best_loss), 6),
        "classes": VIRAT_CLASSES,
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[train] Training config saved to: {config_path}")

    # Save class names list
    classes_path = out_dir / "classes.txt"
    with open(classes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(VIRAT_CLASSES) + "\n")
    print(f"[train] Class names saved to: {classes_path}")

    return best_ckpt_path


# =============================================================================
# CLI Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the VIRAT temporal action head on VIRAT sample videos."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/virat_samples",
        help="Directory containing VIRAT sample mp4 videos (default: data/virat_samples)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/virat_cache",
        help="Directory to cache extracted DINOv2 features (default: data/virat_cache)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/virat_outputs",
        help="Directory to save trained checkpoint and config (default: assets/virat_outputs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', or auto-detected if omitted)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=6,
        help="Frame sampling stride (default: 6, ~5fps from 30fps)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Square image size for feature extraction (default: 640)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Run only feature extraction and caching phase, then exit",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run only pseudo-label generation and classifier training from cached features",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    if args.device is None or args.device.lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 70)
    print("VIRAT Temporal Action Head Training Pipeline")
    print(f"Data directory:    {args.data_dir}")
    print(f"Cache directory:   {args.cache_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Device:            {device}")
    print(f"Stride:            {args.stride}")
    print(f"Image size:        {args.image_size}")
    print(f"Epochs:            {args.epochs}")
    print(f"Learning rate:     {args.lr}")
    print(f"Batch size:        {args.batch_size}")
    print("=" * 70)

    # Phase 1: Extract & cache features
    if not args.train_only:
        print("\n--- Phase 1: Feature Extraction & Caching ---")
        extracted = extract_and_cache(
            video_dir=args.data_dir,
            cache_dir=args.cache_dir,
            device=device,
            image_size=args.image_size,
            stride=args.stride,
        )
        if args.extract_only:
            print("[main] --extract-only specified. Exiting.")
            return

    # Check that cache exists
    cache_path = Path(args.cache_dir)
    cached_pts = list(cache_path.glob("*.pt")) if cache_path.exists() else []
    if not cached_pts:
        print(f"[main] Error: No cached .pt files found in {args.cache_dir}.")
        print("[main] If feature extraction failed due to missing transformers/decord,")
        print("[main] please install dependencies or provide cached .pt files.")
        sys.exit(1)

    # Phase 2: Generate pseudo-labels
    print("\n--- Phase 2: Pseudo-Label Generation ---")
    samples = generate_pseudo_labels(cache_dir=args.cache_dir)
    if not samples:
        print("[main] Error: Could not generate any training samples from cached features.")
        sys.exit(1)

    # Phase 3 & 4: Train classifier and save checkpoint
    print("\n--- Phase 3 & 4: Train TemporalActionClassifier & Save Checkpoint ---")
    best_ckpt = train_classifier(
        samples=samples,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )

    if best_ckpt and best_ckpt.exists():
        print("\n" + "=" * 70)
        print(f"[main] SUCCESS! VIRAT Temporal Action Head trained and saved to: {best_ckpt}")
        print("=" * 70)
    else:
        print("\n[main] Training finished but best.pth was not produced.")
        sys.exit(1)


if __name__ == "__main__":
    main()
