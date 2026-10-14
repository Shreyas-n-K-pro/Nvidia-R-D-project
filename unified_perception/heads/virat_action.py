"""
virat_action head — spatio-temporal action detection for VIRAT surveillance activities.

Replaces whole-scene Kinetics-400 action recognition with a per-actor VIRAT activity
classifier. Extracts per-person tubelet features from DINOv2 patch tokens using detection
bounding boxes from the upstream detection head, and classifies each actor tubelet with a
lightweight temporal Transformer encoder.

Architecture:
  1. Detection Head runs first → provides per-frame bounding boxes for actors ('person').
  2. TubeletPooler → associates actor boxes across frames (greedy IoU tracker) and
     average-pools DINOv2 spatial patch tokens inside each actor bbox into a [T_vis, D]
     feature tubelet.
  3. TemporalActionClassifier → processes tubelet features with positional encoding +
     temporal Transformer encoder + mean pooling + linear classifier → 18 VIRAT activity logits.
  4. Fallback → if no actor detections are available, aggregates whole-scene CLS tokens across
     time and predicts a global scene activity label.
  5. Annotator → burns color-coded actor bounding boxes and predicted action labels onto video.
"""
from __future__ import annotations

import math
import os as _os
import sys
from pathlib import Path
from typing import Any, Optional

from ..perception_types import BackboneOutput, HeadResult

try:
    import torch
    import torch.nn as nn
    _ModuleBase = nn.Module
except ImportError:
    torch = None
    nn = None
    _ModuleBase = object


def _repo_root() -> Path:
    """UnifiedPerception/ root (this file is at <root>/unified_perception/heads/virat_action.py)."""
    return Path(__file__).resolve().parents[2]


# Default assets and environment overrides
DEFAULT_CKPT_PATH = _repo_root() / "assets" / "virat_outputs" / "best.pth"


# =============================================================================
# 1. VIRAT Activity Classes & Color Coding
# =============================================================================

VIRAT_CLASSES: list[str] = [
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

# Distinct OpenCV BGR colors grouped by action semantics
ACTION_COLORS: dict[str, tuple[int, int, int]] = {
    # Locomotion / Posture — Green tones
    "Walking": (0, 255, 0),
    "Running": (50, 255, 0),
    "Standing": (0, 200, 0),
    "Sitting": (0, 180, 50),
    # Person-Vehicle Interactions — Orange / Amber tones
    "Entering_Vehicle": (0, 140, 255),
    "Exiting_Vehicle": (0, 140, 255),
    "Opening_Trunk": (0, 165, 255),
    "Closing_Trunk": (0, 165, 255),
    "Loading_Vehicle": (0, 120, 255),
    "Unloading_Vehicle": (0, 120, 255),
    # Facility Interactions — Magenta / Purple tones
    "Entering_Facility": (255, 0, 180),
    "Exiting_Facility": (255, 0, 180),
    # Object & Social Interactions — Cyan / Yellow-Green tones
    "Carrying_Object": (255, 200, 0),
    "Handing_Off_Object": (255, 180, 0),
    "Gesturing": (220, 220, 0),
    "Talking": (255, 220, 50),
    # Transportation — Yellow
    "Riding_Bicycle": (0, 255, 255),
    # Neutral / Background — Gray
    "Background": (160, 160, 160),
}
DEFAULT_ACTION_COLOR = (0, 255, 255)


def _compute_iou(
    boxA: tuple[float, float, float, float] | list[float],
    boxB: tuple[float, float, float, float] | list[float],
) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union_area = areaA + areaB - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _parse_detection(d: Any) -> Optional[tuple[float, float, float, float, float, str]]:
    """Extract (x1, y1, x2, y2, confidence, class_name) from a detection dictionary."""
    if not isinstance(d, dict):
        return None
    cls_name = str(d.get("class_name", "person")).lower()

    if "bbox" in d:
        b = d["bbox"]
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        else:
            return None
    elif "x1" in d and "y1" in d and "x2" in d and "y2" in d:
        x1 = float(d["x1"])
        y1 = float(d["y1"])
        x2 = float(d["x2"])
        y2 = float(d["y2"])
    else:
        return None

    # Sanity-order coordinates
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    conf = float(d.get("confidence", d.get("conf", 0.0)))
    return x1, y1, x2, y2, conf, cls_name


# =============================================================================
# 2. TubeletPooler
# =============================================================================

class TubeletPooler:
    """Extracts per-actor feature tubes from DINOv2 patch tokens using detection bounding boxes.

    Uses a greedy IoU tracker across frames to link detections of the same actor and
    average-pools spatial patch tokens that fall inside each actor's bounding box.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        patch_grid_h: int = 45,
        patch_grid_w: int = 45,
        image_size: int = 640,
        iou_threshold: float = 0.3,
    ):
        self.feature_dim = feature_dim
        self.patch_grid_h = patch_grid_h
        self.patch_grid_w = patch_grid_w
        self.image_size = image_size
        self.iou_threshold = iou_threshold

        # Populated during pool_actor_features
        self.last_bboxes: dict[int, list[float]] = {}
        self.actor_trajectories: dict[int, list[tuple[int, list[float]]]] = {}

    def pool_actor_features(
        self,
        patch_tokens: Any,
        detections_per_frame: list[list[dict]],
        image_size: int = 640,
    ) -> dict[int, Any]:
        """Extract per-actor tubelet features across frames.

        Parameters
        ----------
        patch_tokens : torch.Tensor [T, N, D] on CPU
            DINOv2 patch tokens without CLS.
        detections_per_frame : list[list[dict]]
            Per-frame detections from detection head. Each dict has bounding box,
            class_name, confidence.
        image_size : int
            Square image dimension used for coordinate mapping (default 640).

        Returns
        -------
        dict mapping actor_id -> torch.Tensor [T_visible, D] of pooled features.
        """
        import torch

        if patch_tokens is None:
            return {}

        if patch_tokens.dim() == 2:
            patch_tokens = patch_tokens.unsqueeze(0)

        T, N, D = patch_tokens.shape

        # Dynamically infer patch grid resolution
        grid_h = self.patch_grid_h
        grid_w = self.patch_grid_w
        if grid_h * grid_w != N:
            side = int(math.isqrt(N))
            if side * side == N:
                grid_h = grid_w = side
            else:
                p = max(1, image_size // 14)
                grid_h = grid_w = p

        cell_w = image_size / float(grid_w)
        cell_h = image_size / float(grid_h)

        # Simple greedy tracker state
        next_actor_id = 0
        active_actors: list[dict[str, Any]] = []  # [{"actor_id": int, "bbox": [x1,y1,x2,y2], "last_t": int}]
        actor_features: dict[int, list[torch.Tensor]] = {}
        actor_last_bboxes: dict[int, list[float]] = {}
        actor_trajectories: dict[int, list[tuple[int, list[float]]]] = {}

        for t in range(T):
            frame_dets = detections_per_frame[t] if (detections_per_frame and t < len(detections_per_frame)) else []

            # Filter for person detections
            person_dets: list[tuple[float, float, float, float, float, dict]] = []
            if isinstance(frame_dets, list):
                for d in frame_dets:
                    parsed = _parse_detection(d)
                    if parsed is not None:
                        x1, y1, x2, y2, conf, cls_name = parsed
                        if cls_name == "person":
                            person_dets.append((x1, y1, x2, y2, conf, d))

            # IoU greedy matching with active actors
            matched_actors: set[int] = set()
            matched_dets: set[int] = set()
            actor_assignments: dict[int, int] = {}  # det_idx -> actor_id

            if active_actors and person_dets:
                pairs: list[tuple[float, int, int]] = []
                for a_idx, act in enumerate(active_actors):
                    for d_idx, pdet in enumerate(person_dets):
                        cand_box = [pdet[0], pdet[1], pdet[2], pdet[3]]
                        iou = _compute_iou(act["bbox"], cand_box)
                        if iou >= self.iou_threshold:
                            pairs.append((iou, a_idx, d_idx))

                pairs.sort(key=lambda item: item[0], reverse=True)
                for iou, a_idx, d_idx in pairs:
                    if a_idx not in matched_actors and d_idx not in matched_dets:
                        matched_actors.add(a_idx)
                        matched_dets.add(d_idx)
                        act = active_actors[a_idx]
                        act["bbox"] = [person_dets[d_idx][0], person_dets[d_idx][1], person_dets[d_idx][2], person_dets[d_idx][3]]
                        act["last_t"] = t
                        actor_assignments[d_idx] = act["actor_id"]

            # Unmatched detections initialize new actors
            for d_idx, pdet in enumerate(person_dets):
                if d_idx not in matched_dets:
                    new_id = next_actor_id
                    next_actor_id += 1
                    box = [pdet[0], pdet[1], pdet[2], pdet[3]]
                    active_actors.append({"actor_id": new_id, "bbox": box, "last_t": t})
                    actor_assignments[d_idx] = new_id

            # Age out active actors not seen for more than 30 frames
            active_actors = [a for a in active_actors if (t - a["last_t"]) <= 30]

            # Extract and pool patch tokens for each person in frame t
            frame_patches = patch_tokens[t]  # [N, D]
            patch_grid = frame_patches.view(grid_h, grid_w, D)

            for d_idx, pdet in enumerate(person_dets):
                actor_id = actor_assignments[d_idx]
                x1, y1, x2, y2 = pdet[0], pdet[1], pdet[2], pdet[3]

                # Map bbox in image_size coords to patch grid indices
                gx1 = int(math.floor(x1 / cell_w))
                gx2 = int(math.ceil(x2 / cell_w))
                gy1 = int(math.floor(y1 / cell_h))
                gy2 = int(math.ceil(y2 / cell_h))

                # Clamp to grid boundaries
                gx1 = max(0, min(gx1, grid_w - 1))
                gx2 = max(gx1 + 1, min(gx2, grid_w))
                gy1 = max(0, min(gy1, grid_h - 1))
                gy2 = max(gy1 + 1, min(gy2, grid_h))

                # Average-pool patches inside the bounding box
                patches_in_box = patch_grid[gy1:gy2, gx1:gx2, :]
                actor_feat = patches_in_box.reshape(-1, D).mean(dim=0)  # [D]

                if actor_id not in actor_features:
                    actor_features[actor_id] = []
                    actor_trajectories[actor_id] = []

                actor_features[actor_id].append(actor_feat)
                actor_last_bboxes[actor_id] = [x1, y1, x2, y2]
                actor_trajectories[actor_id].append((t, [x1, y1, x2, y2]))

        self.last_bboxes = actor_last_bboxes
        self.actor_trajectories = actor_trajectories

        # Stack features per actor into [T_visible, D]
        result: dict[int, torch.Tensor] = {}
        for actor_id, feats in actor_features.items():
            if feats:
                result[actor_id] = torch.stack(feats, dim=0)

        return result


# =============================================================================
# 3. TemporalActionClassifier
# =============================================================================

class TemporalActionClassifier(_ModuleBase):
    """A lightweight temporal classifier over actor feature tubes."""

    def __init__(
        self,
        feature_dim: int = 384,
        num_classes: int = 18,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        import torch
        import torch.nn as nn

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, feature_dim))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: Any) -> Any:
        """Forward pass over actor feature tubelets.

        Parameters
        ----------
        x : torch.Tensor [B, T, D] or [T, D]
            Actor feature tube(s).

        Returns
        -------
        torch.Tensor [B, num_classes] logits.
        """
        import torch
        import torch.nn.functional as F

        squeeze_on_return = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_on_return = False

        B, T, D = x.shape
        max_pos = self.positional_encoding.shape[1]
        if T <= max_pos:
            pos = self.positional_encoding[:, :T, :]
        else:
            # Interpolate if sequence length exceeds max_seq_len
            pos = F.interpolate(
                self.positional_encoding.transpose(1, 2),
                size=T,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        x_enc = x + pos
        out = self.transformer(x_enc)       # [B, T, D]
        pooled = out.mean(dim=1)            # [B, D]
        logits = self.classifier(pooled)    # [B, num_classes]

        if squeeze_on_return:
            return logits[0]
        return logits


# =============================================================================
# 4. ViratActionHead
# =============================================================================

class ViratActionHead:
    """Spatio-temporal action detection head for VIRAT surveillance activities.

    Consumes DINOv2 patch tokens and CLS features from BackboneOutput along with
    detection bounding boxes from prior_results['detection'], extracting per-actor
    feature tubelets and classifying activities into 18 VIRAT categories.
    """

    name = "virat_action"
    required_features = ["patch_tokens", "cls"]

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        topk: int = 5,
        device: Optional[str] = None,
        allow_untrained: bool = False,
    ):
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.topk = topk
        self.allow_untrained = allow_untrained or (_os.environ.get("SENTRY_VIRAT_ALLOW_UNTRAINED") == "1")

        self.pooler = TubeletPooler(feature_dim=384, patch_grid_h=45, patch_grid_w=45, image_size=640)
        self.classifier = TemporalActionClassifier(feature_dim=384, num_classes=len(VIRAT_CLASSES))
        self.classes = list(VIRAT_CLASSES)

        if checkpoint_path is None:
            checkpoint_path = _os.environ.get("SENTRY_VIRAT_ACTION_CKPT", str(DEFAULT_CKPT_PATH))
        self.checkpoint_path = checkpoint_path
        self.checkpoint_loaded = False
        self.load_error: Optional[str] = None

        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                if isinstance(state, dict):
                    sd = state.get("state_dict", state.get("model_state_dict", state))
                else:
                    sd = state
                self.classifier.load_state_dict(sd, strict=False)
                self.checkpoint_loaded = True
            except Exception as e:
                self.load_error = f"Failed to load checkpoint at {checkpoint_path}: {e}"
                print(f"[virat_action] checkpoint load warning: {self.load_error}", file=sys.stderr)
        else:
            # Classifier is randomly initialized (for training)
            self.load_error = f"Checkpoint not found at: {checkpoint_path}"

        self.classifier.to(self.device)
        self.classifier.eval()

    def _try_lazy_load_checkpoint(self):
        """Attempt to load checkpoint if it became available on disk after __init__."""
        if not self.checkpoint_loaded and self.checkpoint_path and Path(self.checkpoint_path).exists():
            try:
                import torch
                state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                sd = state.get("state_dict", state.get("model_state_dict", state)) if isinstance(state, dict) else state
                self.classifier.load_state_dict(sd, strict=False)
                self.checkpoint_loaded = True
                self.load_error = None
            except Exception as e:
                self.load_error = str(e)

    def predict(
        self,
        features: BackboneOutput,
        prior_results: Optional[dict[str, Any]] = None,
        allow_untrained: Optional[bool] = None,
    ) -> HeadResult:
        """Run spatio-temporal action detection over shared backbone features.

        Parameters
        ----------
        features : BackboneOutput
            Shared DINOv2 features including patch_tokens [T, N, D] and cls [T, D].
        prior_results : dict, optional
            Results from prior heads. Reads prior_results['detection']['primary'].
        allow_untrained : bool, optional
            If True, permit inference with randomly initialized weights if checkpoint
            is missing.

        Returns
        -------
        HeadResult
        """
        try:
            return self._predict_impl(features, prior_results=prior_results, allow_untrained=allow_untrained)
        except Exception as e:
            return HeadResult(head_name=self.name, error=f"virat_action predict failed: {e!r}")

    def _predict_impl(
        self,
        features: BackboneOutput,
        prior_results: Optional[dict[str, Any]] = None,
        allow_untrained: Optional[bool] = None,
    ) -> HeadResult:
        import torch

        # Attempt to load checkpoint if it became available
        self._try_lazy_load_checkpoint()

        use_untrained = self.allow_untrained if allow_untrained is None else allow_untrained
        if not use_untrained:
            use_untrained = (_os.environ.get("SENTRY_VIRAT_ALLOW_UNTRAINED") == "1")

        if not self.checkpoint_loaded and not use_untrained:
            return HeadResult(
                head_name=self.name,
                error=(
                    f"head checkpoint not found: {self.checkpoint_path}. "
                    f"Set SENTRY_VIRAT_ACTION_CKPT to a trained checkpoint, "
                    f"or run with allow_untrained=True."
                ),
            )

        metadata = features.metadata or {}
        image_size = int(metadata.get("image_size", 640))
        cls_tokens = features.cls
        patch_tokens = features.patch_tokens

        # Extract upstream detection bboxes
        detection_primary = None
        if prior_results and "detection" in prior_results:
            det_head_data = prior_results["detection"]
            if isinstance(det_head_data, dict):
                detection_primary = det_head_data.get("primary")
            elif hasattr(det_head_data, "primary"):
                detection_primary = det_head_data.primary

        has_per_frame_detections = (
            isinstance(detection_primary, list)
            and len(detection_primary) > 0
            and isinstance(detection_primary[0], list)
            and any(len(frame_dets) > 0 for frame_dets in detection_primary)
        )

        actor_tubes: dict[int, torch.Tensor] = {}
        if has_per_frame_detections and patch_tokens is not None:
            actor_tubes = self.pooler.pool_actor_features(
                patch_tokens=patch_tokens,
                detections_per_frame=detection_primary,
                image_size=image_size,
            )

        # Fallback to global CLS-based classification if no actor tubes found
        if not actor_tubes:
            if cls_tokens is None or cls_tokens.numel() == 0:
                if patch_tokens is not None and patch_tokens.numel() > 0:
                    cls_input = patch_tokens.mean(dim=1).unsqueeze(0)  # [1, T, D]
                else:
                    return HeadResult(head_name=self.name, error="no cls or patch_tokens features available")
            else:
                if cls_tokens.dim() == 2:
                    cls_input = cls_tokens.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, D]
                else:
                    cls_input = cls_tokens.view(1, 1, -1)

            cls_input = cls_input.to(self.device)
            with torch.no_grad():
                logits = self.classifier(cls_input)  # [1, num_classes]
                probs = torch.softmax(logits, dim=-1)[0]
                k = min(self.topk, len(self.classes))
                top_probs, top_indices = torch.topk(probs, k=k)
                top_k = [(self.classes[int(idx)], float(p)) for idx, p in zip(top_indices, top_probs)]

            top_label = top_k[0][0]
            top_conf = top_k[0][1]

            return HeadResult(
                head_name=self.name,
                primary=top_label,
                confidence=float(top_conf),
                details={
                    "mode": "global_cls",
                    "top_k": top_k,
                    "checkpoint_path": self.checkpoint_path,
                    "checkpoint_loaded": self.checkpoint_loaded,
                    "device": self.device,
                },
                annotated_mp4=None,
            )

        # Per-actor activity classification
        primary_list: list[dict[str, Any]] = []
        per_actor_predictions: dict[int, dict[str, Any]] = {}

        for actor_id, tube in actor_tubes.items():
            tube_in = tube.unsqueeze(0).to(self.device)  # [1, T_visible, D]
            with torch.no_grad():
                logits = self.classifier(tube_in)  # [1, num_classes]
                probs = torch.softmax(logits, dim=-1)[0]
                k = min(self.topk, len(self.classes))
                top_probs, top_indices = torch.topk(probs, k=k)
                actor_top_k = [(self.classes[int(idx)], float(p)) for idx, p in zip(top_indices, top_probs)]

            top_action = actor_top_k[0][0]
            top_conf = actor_top_k[0][1]
            bbox_last = self.pooler.last_bboxes.get(actor_id, [0.0, 0.0, 0.0, 0.0])

            primary_item = {
                "actor_id": int(actor_id),
                "action": top_action,
                "confidence": float(top_conf),
                "bbox_last": [float(c) for c in bbox_last],
            }
            primary_list.append(primary_item)

            per_actor_predictions[int(actor_id)] = {
                "top_k": actor_top_k,
                "action": top_action,
                "confidence": float(top_conf),
                "bbox_last": [float(c) for c in bbox_last],
                "num_frames_visible": int(tube.shape[0]),
            }

        mean_conf = (
            sum(p["confidence"] for p in primary_list) / len(primary_list)
            if primary_list else 0.0
        )

        return HeadResult(
            head_name=self.name,
            primary=primary_list,
            confidence=float(mean_conf),
            details={
                "mode": "per_actor",
                "num_actors": len(primary_list),
                "per_actor_predictions": per_actor_predictions,
                "actor_trajectories": self.pooler.actor_trajectories,
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_loaded": self.checkpoint_loaded,
                "device": self.device,
            },
            annotated_mp4=None,
        )

    def annotate(
        self,
        source_mp4: str,
        head_result: HeadResult,
        out_path: str,
        image_size: int = 640,
    ) -> Optional[str]:
        """Burn per-actor bounding boxes and VIRAT action labels onto video frames.

        Parameters
        ----------
        source_mp4 : str
            Input video file path.
        head_result : HeadResult
            Result object from predict().
        out_path : str
            Destination path for annotated mp4.
        image_size : int
            Square image dimension used during feature extraction (default 640).

        Returns
        -------
        Path to annotated mp4, or None on failure.
        """
        try:
            import cv2
        except ImportError:
            print("[virat_action] cv2 not available — skipping annotate", file=sys.stderr)
            return None

        if head_result is None or head_result.error or not head_result.primary:
            return None

        cap = cv2.VideoCapture(source_mp4)
        if not cap.isOpened():
            print(f"[virat_action] annotate: could not open source video {source_mp4}", file=sys.stderr)
            return None

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        sx = src_w / float(image_size)
        sy = src_h / float(image_size)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (src_w, src_h))
        if not writer.isOpened():
            print(f"[virat_action] annotate: writer failed to open {out_path}", file=sys.stderr)
            cap.release()
            return None

        mode = head_result.details.get("mode", "per_actor")

        if mode == "global_cls":
            # Global scene action banner
            label = str(head_result.primary)
            conf = float(head_result.confidence or 0.0)
            text = f"Scene Action: {label} ({conf:.2f})"
            color = ACTION_COLORS.get(label, DEFAULT_ACTION_COLOR)

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                banner_h = th + 24
                cv2.rectangle(frame, (0, 0), (src_w, banner_h), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    text,
                    ((src_w - tw) // 2, th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)

        else:
            # Per-actor bounding box overlays
            actor_actions: dict[int, tuple[str, float]] = {}
            if isinstance(head_result.primary, list):
                for item in head_result.primary:
                    if isinstance(item, dict) and "actor_id" in item:
                        actor_actions[int(item["actor_id"])] = (
                            item.get("action", "?"),
                            float(item.get("confidence", 0.0)),
                        )

            # Map frame_idx -> [(actor_id, [x1, y1, x2, y2])]
            frame_actors: dict[int, list[tuple[int, list[float]]]] = {}
            trajectories = head_result.details.get("actor_trajectories", {})
            if isinstance(trajectories, dict):
                for a_id_str, frames_list in trajectories.items():
                    a_id = int(a_id_str)
                    if isinstance(frames_list, list):
                        for f_idx, box in frames_list:
                            frame_actors.setdefault(int(f_idx), []).append((a_id, box))

            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                actors_in_frame = frame_actors.get(frame_idx, [])
                for actor_id, box in actors_in_frame:
                    action, conf = actor_actions.get(actor_id, ("?", 0.0))
                    color = ACTION_COLORS.get(action, DEFAULT_ACTION_COLOR)

                    x1 = int(round(box[0] * sx))
                    y1 = int(round(box[1] * sy))
                    x2 = int(round(box[2] * sx))
                    y2 = int(round(box[3] * sy))

                    # Clamp to frame dimensions
                    x1 = max(0, min(x1, src_w - 1))
                    x2 = max(x1 + 1, min(x2, src_w))
                    y1 = max(0, min(y1, src_h - 1))
                    y2 = max(y1 + 1, min(y2, src_h))

                    # Bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label badge above or inside box
                    label_text = f"#{actor_id} {action} {conf:.2f}"
                    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    badge_y1 = max(0, y1 - th - baseline - 6)
                    badge_y2 = max(badge_y1 + th + baseline + 6, y1)
                    badge_x2 = min(x1 + tw + 8, src_w)

                    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
                    cv2.putText(
                        frame,
                        label_text,
                        (x1 + 4, badge_y2 - baseline - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                writer.write(frame)
                frame_idx += 1

        cap.release()
        writer.release()
        head_result.annotated_mp4 = out_path
        return out_path


# =============================================================================
# 5. Module-Level Export
# =============================================================================

HEAD = ViratActionHead()

__all__ = [
    "VIRAT_CLASSES",
    "ACTION_COLORS",
    "TubeletPooler",
    "TemporalActionClassifier",
    "ViratActionHead",
    "HEAD",
]
