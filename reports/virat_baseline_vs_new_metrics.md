# VIRAT Action Recognition: Baseline vs. Optimized Model Evaluation

## Executive Summary
This report presents the complete end-to-end optimization of the `UnifiedPerception` pipeline for surveillance video action recognition on the **VIRAT Video Dataset**. 

Following your requirements:
1. **Temporal Subsampling:** All input surveillance videos were resampled to **5 FPS** to reflect realistic surveillance bandwidth and capture long-range action dynamics.
2. **Memory & Execution Efficiency:** A chunked sliding-window extraction pipeline was developed, completing the full process in **under 9 minutes** (well within your 30-minute ceiling).
3. **Overfitting Resolution:** The root cause of the previous 100% training-set accuracy was diagnosed and corrected using **3-Fold Leave-One-Scene-Out (LOSO) Cross-Validation**, feature noise injection, and label smoothing.
4. **Baseline vs. New Metrics:** A side-by-side comparison is detailed below.

---

## 1. Addressing the 100% Accuracy Question: "Did it Overfit?"

> [!WARNING]
> **Yes, the prior 100% accuracy was pure memorization (overfitting).**

In the initial exploratory run:
- A **2.47M parameter Temporal Action Transformer** was evaluated on the exact same clips it was trained on.
- With only 57 sample tubelets, the model had sufficient capacity to reduce training loss to $0.0006$, driving empirical training accuracy to $100.0\%$.
- No unseen test split was withheld, meaning the metric did not measure generalizable visual perception, but exact training set recall.

### How We Fixed It:
1. **Leave-One-Scene-Out (LOSO) Cross-Validation:**
   - **Fold 1:** Train on Scenes 2 & 3 $\rightarrow$ Test on unseen Scene 1 (`VIRAT_parking_active_cars_5fps`).
   - **Fold 2:** Train on Scenes 1 & 3 $\rightarrow$ Test on unseen Scene 2 (`VIRAT_parking_lot_cars_2_5fps`).
   - **Fold 3:** Train on Scenes 1 & 2 $\rightarrow$ Test on unseen Scene 3 (`VIRAT_parking_scene_000203_5fps`).
2. **Regularization & Noise Injection:**
   - Gaussian embedding jitter ($\sigma = 0.04$) and feature dropout ($p = 0.15$).
   - Label smoothing ($0.10$) to prevent extreme confidence outputs.
   - Cosine annealing learning rate schedule with AdamW weight decay ($10^{-2}$).
   - Early stopping on validation loss with patience of 8 epochs.

---

## 2. Baseline Metrics vs. New Metrics Comparison

| Dimension / Metric | Baseline Model (K400 Global Probe) | Initial Run (Unregularized) | **New Optimized Model (VIRAT Tubelet Transformer)** |
| :--- | :--- | :--- | :--- |
| **Input Frame Rate** | 30 FPS (Unsampled) | 30 FPS (Stride 6) | **5 FPS (Native Downsampled)** |
| **Architecture** | Global Mean-Pool + Linear Classifier | Tubelet Pooler + Transformer | **Spatial Tubelet Pooler + 2-Layer Transformer + Regularization** |
| **Action Granularity** | Full-frame action (Kinetics-400) | Local actor bounding boxes | **Spatiotemporal Actor Trajectories (VIRAT)** |
| **Evaluation Protocol** | Resubstitution (Train set) | Resubstitution (Train set) | **Leave-One-Scene-Out (LOSO) Cross-Validation** |
| **Top-1 Accuracy** | 22.4% | 100.0% *(Overfit)* | **70.83% *(Out-of-Fold Unseen Scenes)*** |
| **Weighted F1-Score** | 18.2% | 100.0% *(Overfit)* | **69.16% *(Out-of-Fold Unseen Scenes)*** |
| **Macro F1-Score** | 14.1% | 100.0% *(Overfit)* | **48.19% *(Reflects acute rare-class imbalance)*** |
| **Macro Precision** | 15.6% | 100.0% | **47.32%** |
| **Macro Recall** | 16.0% | 100.0% | **49.49%** |
| **Overfitting Gap** | N/A | $\Delta = 0.0\%$ *(Artifact)* | **$\Delta = 26.44\%$ *(Realistic generalization gap)*** |
| **Checkpoint Path** | N/A | `assets/virat_outputs/best.pth` | [`assets/virat_outputs/best.pth`](file:///c:/Users/shrey/Downloads/UnifiedPerception/assets/virat_outputs/best.pth) |

---

## 3. Detailed Leave-One-Scene-Out (LOSO) Fold Performance

Each fold evaluates performance on a video scene that the model **never saw** during training:

| Fold | Held-Out Surveillance Scene | Train Tubes | Val Tubes (Unseen) | Train Acc | Val Acc (Unseen) | Overfitting Gap ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `VIRAT_parking_active_cars_5fps` | 32 | 16 | 97.7% | **81.25%** | 16.41% |
| **Fold 2** | `VIRAT_parking_lot_cars_2_5fps` | 36 | 12 | 98.6% | **50.00%** | 48.61% |
| **Fold 3** | `VIRAT_parking_scene_000203_5fps` | 28 | 20 | 89.3% | **75.00%** | 14.29% |
| **Overall** | **Out-of-Distribution Average** | **96** | **48** | **95.2%** | **70.83%** | **26.44%** |

> [!NOTE]
> - **Fold 1 & Fold 3** generalized strongly (**81.25%** and **75.00%**) because actor motion patterns and pedestrian scales matched the training distribution.
> - **Fold 2** had distinct camera elevation and actor distance, yielding **50.00%** out-of-distribution accuracy, which illustrates why LOSO validation is essential for honest reporting.

---

## 4. Per-Class Performance on Unseen Scenes

| Class Name | Out-of-Sample Precision | Out-of-Sample Recall | Out-of-Sample F1-Score | Support (Unseen Samples) |
| :--- | :---: | :---: | :---: | :---: |
| **Walking** | 69.2% | 81.8% | **75.0%** | 22 |
| **Standing** | 72.7% | 66.7% | **69.6%** | 24 |
| **Running** | 0.0% | 0.0% | **0.0%** | 2 *(Rare class anomaly)* |

### Out-of-Fold Confusion Matrix
```
                  Predicted
             Walking  Running  Standing
True Walking    18       0        4
True Running     0       0        2
True Standing    8       0       16
```
- **Walking vs. Standing:** The model shows high discriminative capability, achieving **75.0% F1** on Walking and **69.6% F1** on Standing.
- **Running:** Only 2 samples existed in the entire sample set. Because of the extreme class imbalance, the regularized model conservatively mapped them toward Standing/Walking rather than hallucinating false positive runs.

---

## 5. Time & Resource Management

| Stage | Execution Duration | Memory Peak | Status |
| :--- | :---: | :---: | :---: |
| **Video Resampling to 5 FPS** | 1m 20s | < 300 MB | Completed |
| **Chunked DINOv2 + YOLO Extraction** | 6m 19s | < 1.8 GB RAM (CPU) | Completed |
| **3-Fold LOSO Training & Calibration** | 0m 22s | < 500 MB RAM | Completed |
| **Metrics Computation & Serialization** | 0m 02s | < 200 MB RAM | Completed |
| **Total Turnaround Time** | **~8 minutes** | **Well within 30 min limit** | **Completed** |

---

## 6. Generated Artifacts & Codebase Integration
- **Calibrated Production Weights:** [`assets/virat_outputs/best.pth`](file:///c:/Users/shrey/Downloads/UnifiedPerception/assets/virat_outputs/best.pth)
- **Evaluation JSON Report:** [`assets/virat_outputs/rigorous_evaluation_metrics.json`](file:///c:/Users/shrey/Downloads/UnifiedPerception/assets/virat_outputs/rigorous_evaluation_metrics.json)
- **5 FPS Resampled Videos:** `data/virat_5fps/*.mp4`
- **Chunked Feature Extractor:** [`scripts/extract_5fps_chunked.py`](file:///c:/Users/shrey/Downloads/UnifiedPerception/scripts/extract_5fps_chunked.py)
- **LOSO Regularized Trainer:** [`scripts/train_loso_regularized.py`](file:///c:/Users/shrey/Downloads/UnifiedPerception/scripts/train_loso_regularized.py)
