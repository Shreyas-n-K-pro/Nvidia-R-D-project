#!/usr/bin/env python3
"""Resample VIRAT source videos to 5 FPS using OpenCV."""
import os
import cv2
from pathlib import Path

def resample_videos(src_dir="data/virat_samples", dst_dir="data/virat_5fps", target_fps=5.0):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    video_files = sorted(list(src_path.glob("*.mp4")))
    print(f"[resample] Found {len(video_files)} video(s) in {src_path}")
    
    resampled_files = []
    for vf in video_files:
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            print(f"[resample] Error: Cannot open {vf}")
            continue
            
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame step for target_fps (e.g. 30 / 5 = 6)
        step = max(1, round(orig_fps / target_fps))
        out_name = f"{vf.stem}_5fps.mp4"
        out_file = dst_path / out_name
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file), fourcc, target_fps, (src_w, src_h))
        
        written_count = 0
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                writer.write(frame)
                written_count += 1
            frame_idx += 1
            
        cap.release()
        writer.release()
        duration = written_count / target_fps
        print(f"[resample] {vf.name} ({orig_fps:.1f} fps, {total_frames} frames) -> {out_name}: {src_w}x{src_h}, {target_fps:.1f} fps, {written_count} frames ({duration:.1f}s)")
        resampled_files.append(out_file)
        
    return resampled_files

if __name__ == "__main__":
    resample_videos()
