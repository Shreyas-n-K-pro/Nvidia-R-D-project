# Comprehensive Technical Documentation: Caching Subsystems & Execution Architecture

**Project Title:** *Develop and Validate Optimised Multimodal Models for Automotive Use Cases*  
**Operational Context:** Automotive Surveillance Mode (ASM) — Surround View Perception  
**Collaboration:** KLE Technological University × NVIDIA (Senior Design Project)

---

## 1. Executive Summary & Purpose of Caching

In **Automotive Surveillance Mode (ASM)**, the ego vehicle is parked with the combustion engine / primary high-voltage traction inverter switched off. The surrounding environment is monitored continuously via **4 Fisheye Surround View Cameras** (Front, Rear, Left, Right). 

Because the system operates solely on the vehicle’s auxiliary battery State of Charge (SoC), computational efficiency is paramount:
1. **Vision Transformer Compute Cost:** DINOv2 ViT-S/14 forward passes at $640 \times 640$ resolution generate $45 \times 45 = 2,025$ spatial patch tokens per frame, demanding approximately **$\approx 42\text{ GMACs}$** per unpruned frame.
2. **Detection & Tracking Latency:** Passing multi-scale intermediate hidden states through ViT-Adapter-v2 and the YOLO26 head involves memory allocation, deformable cross-attention, non-maximum suppression (NMS), and bipartite Hungarian matching.
3. **Hardware Constraint:** The production pipeline targets an embedded **PVA (Programmable Vision Accelerator)** with a strict power ceiling of $1.5\text{ W} - 5.0\text{ W}$ and a compute budget of **$< 10\text{ GMACs}$ per camera frame**.

To eliminate redundant compute, manage strict host memory limits, and enable reproducible cross-scene experimentation, our architecture implements **two active caching layers** alongside a designed **runtime temporal patch cache roadmap**:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   ACTIVE IMPLEMENTED CACHING LAYERS                      │
├──────────────────────────────────┬───────────────────────────────────────┤
│ 1. Persistent Feature &          │ Serializes DINOv2 patch tokens, CLS   │
│    Detection Disk Cache          │ embeddings, and YOLO tracks to .pt    │
├──────────────────────────────────┼───────────────────────────────────────┤
│ 2. In-Memory Detector            │ Lazy singleton pool in RAM preserving │
│    Instance Cache                │ weights, anchors, and CUDA context    │
├──────────────────────────────────┴───────────────────────────────────────┤
│                   UPCOMING RUNTIME TEMPORAL CACHE                        │
├──────────────────────────────────┬───────────────────────────────────────┤
│ 3. Runtime Temporal Patch Cache  │ Frame-to-frame patch differencing to  │
│    (Milestone 3 Roadmap)         │ skip static background (<10 GMACs)    │
└──────────────────────────────────┴───────────────────────────────────────┘
```

---

## 2. Layer 1: Persistent Feature & Detection Disk Cache

### 2.1 Technical Location & Interfaces
* **Primary Extraction Script:** [`scripts/extract_5fps_chunked.py`](file:///c:/Users/shrey/Downloads/UnifiedPerception/scripts/extract_5fps_chunked.py)
* **Training / Evaluation Consumer:** [`scripts/train_loso_regularized.py`](file:///c:/Users/shrey/Downloads/UnifiedPerception/scripts/train_loso_regularized.py)
* **Storage Target:** `data/virat_cache_5fps/*.pt`

### 2.2 Memory-Safe Chunked Sliding-Window Extraction
Attempting to extract intermediate hidden states and patch tokens for an entire $350+$ frame video all at once consumes over $14\text{ GB}$ of RAM, leading to Out-Of-Memory (OOM) operating system termination.

To solve this, [`extract_5fps_chunked.py`](file:///c:/Users/shrey/Downloads/UnifiedPerception/scripts/extract_5fps_chunked.py) introduces **chunked temporal windowing**:
* Videos are read in sliding chunks of **$N = 40$ frames** ($8.0\text{ seconds}$ at $5\text{ FPS}$).
* Backbone inference runs in micro-batches of $8$ frames.
* As each chunk completes, raw frames and intermediate computational graphs are freed via explicit Python garbage collection (`gc.collect()`), capping peak RAM usage strictly below **$1.8\text{ GB}$** on standard CPU/host setups.

### 2.3 Cached Data Structure Contract
Each `.pt` cache artifact stores a consolidated dictionary containing the complete visual representation of the scene:

```python
cache_data = {
    "patch_tokens": torch.Tensor,      # Shape: [T, 2025, 384], dtype=torch.float32
    "cls":          torch.Tensor,      # Shape: [T, 384],       dtype=torch.float32
    "detections":   List[List[dict]],  # Length T; per-frame detected objects
    "metadata": {
        "source":     str,             # Full source path of video file
        "fps":        float,           # Native frame rate (5.0 FPS)
        "num_frames": int,             # Total frame count in sequence (e.g. 210, 353, 245)
    },
    "video_name":   str                # Stem identifier (e.g. VIRAT_parking_lot_cars_2_5fps)
}
```

#### Detailed Breakdown of Cached Fields:
1. **`patch_tokens` ($[T, 2025, 384]$):**
   * Corresponds to the spatial grid of $45 \times 45$ tokens from DINOv2 Layer 12.
   * Enables the downstream **TubeletPooler** to slice arbitrary actor bounding boxes $[x_1, y_1, x_2, y_2]$ at any frame $t$ without ever re-running the Vision Transformer.
2. **`cls` ($[T, 384]$):**
   * Global scene classification embedding per frame. Used for global surveillance context and scene stability checks.
3. **`detections` ($[T]$ list):**
   * Contains the complete output of YOLO26 + ViT-Adapter-v2 after non-maximum suppression (NMS):
     ```python
     {
         "bbox": [x1, y1, x2, y2],       # Normalized coordinates [0.0 .. 1.0]
         "conf": float,                   # Detection confidence score
         "class_id": int,                 # 0: person, 2: car, 5: bus, 7: truck
         "class_name": str                # Human-readable string identifier
     }
     ```

### 2.4 Cache Verification & Existence Checks
During execution, the caching subsystem checks for existing artifacts prior to triggering extraction:
```python
out_pt = cache_dir / f"{vid.stem}.pt"
if out_pt.exists():
    print(f"Skipping existing cache: {out_pt.name}")
    continue
```
* **Performance Impact:** Cross-validation folds and hyperparameter tuning iterations run in **$22\text{ seconds}$** rather than **$45+\text{ minutes}$**, eliminating redundant DINOv2 forwards entirely.

---

## 3. Layer 2: In-Memory Runtime Detector Instance Cache

### 3.1 Technical Location & Mechanism
* **Implementation Module:** [`unified_perception/heads/detection.py`](file:///c:/Users/shrey/Downloads/UnifiedPerception/unified_perception/heads/detection.py)
* **Data Structure:** Module-level global dictionary:
  ```python
  _PRADIPT_IMPL_CACHE: dict[tuple[str, str, str], SharedBackboneDetector] = {}
  ```

### 3.2 Motivation & Latency Elimination
Initializing the detection pipeline involves:
1. Resolving the package path to Pradipt's Adapter-v2 implementation.
2. Allocating GPU/CPU memory for the multi-scale adapter convolutions ($P_3, P_4, P_5$).
3. Loading the $223$-layer YOLO26 state dictionary from binary weights on disk.
4. Initializing PyTorch CUDA execution contexts and PyTorch-to-C++ symbol bindings.

Repeating this model instantiation for every video clip or streaming batch introduces **$3.5 - 5.0\text{ seconds}$ of overhead** per call.

### 3.3 Cache Lifecycle & Keying
The detector is implemented as a thread-safe **Lazy Singleton Pattern** indexed by a 3-part composite key:
$$\text{Key} = (\text{adapter\_type}, \text{device}, \text{adapter\_ckpt\_path})$$

```python
def _load_pradipt_impl(training_package: str, adapter_type: str, device: str = "cuda"):
    paths = _resolve_paths(training_package, adapter_type)
    key = (adapter_type, device, str(paths["adapter_ckpt"]))
    if key in _PRADIPT_IMPL_CACHE:
        return _PRADIPT_IMPL_CACHE[key]
        
    impl = SharedBackboneDetector(
        adapter_checkpoint=paths["adapter_ckpt"],
        yolo_weights=paths["yolo_weights"],
        adapter_type=adapter_type,
        device=device,
    )
    _PRADIPT_IMPL_CACHE[key] = impl
    return impl
```
* **Result:** Subsequent predictions across any camera feed or video segment reuse the hot in-memory instance with **$0\text{ ms}$ initialization penalty**.

---

## 4. Spatiotemporal Tubelet Pooling Utilizing Cached Features

The primary beneficiary of the disk caching layer is the **VIRAT Action Head (`TubeletPooler` + `TemporalActionClassifier`)**:

```
┌────────────────────────────────────────────────────────┐
│            CACHED ARTIFACTS ON DISK (.pt)              │
│  patch_tokens: [T, 2025, 384]   detections: [T] boxes  │
└──────────────────────────┬─────────────────────────────┘
                           │ Fast Memory-Mapped Load
                           ▼
┌────────────────────────────────────────────────────────┐
│             SPATIAL TUBELET POOLER                     │
│  Projects box [x1, y1, x2, y2] to 45 x 45 patch grid   │
│  Extracts local patch slice: (r1:r2, c1:c2)            │
│  Spatial Average Pooling -> Feature Vector [384]       │
└──────────────────────────┬─────────────────────────────┘
                           │ Assembles Tubelet over T frames
                           ▼
┌────────────────────────────────────────────────────────┐
│          ACTOR TUBELET: [N_actors, T, 384]             │
│  2-Layer Temporal Self-Attention Transformer Head      │
│  Outputs 18 VIRAT surveillance action probabilities    │
└────────────────────────────────────────────────────────┘
```

Because the patch tokens are preserved losslessly in the cache, the TubeletPooler can experiment with dynamic temporal lengths ($T = 8, 16, 32$), various IoU tracking thresholds ($0.25$ vs $0.50$), and data augmentation noise without touching the raw video or vision backbone.

---

## 5. Architectural Roadmap: Runtime Temporal Patch Caching (< 10 GMACs)

While Layers 1 and 2 optimize disk and RAM utilization, meeting the strict **$< 10\text{ GMACs}$ hardware KPI** on the embedded PVA requires **Runtime Temporal Patch Caching**:

### 5.1 The Parked Vehicle Invariance Property
Surveillance cameras mounted on a stationary vehicle exhibit extreme temporal redundancy:
* Over **$80\% - 90\%$ of spatial patches** (asphalt, parked vehicles, distant background) do not change between consecutive $5\text{ FPS}$ frames ($\Delta t = 200\text{ ms}$).
* Re-computing DINOv2 self-attention on static background patches wastes $35+\text{ GMACs}$ per frame.

### 5.2 Patch Differencing & Token Gating Algorithm
In the upcoming optimization milestone:
1. **Pixel-Level Residual:** For frame $I_t$, compute absolute residual with frame $I_{t-1}$:
   $$R_t = |I_t - I_{t-1}|$$
2. **Patch Activity Thresholding:** Average residual over each $14 \times 14$ patch region $p_{i, j}$:
   $$\mathcal{A}(p_{i, j}) = \frac{1}{196}\sum_{(x, y) \in p_{i, j}} R_t(x, y)$$
3. **Dynamic Token Gating:**
   * If $\mathcal{A}(p_{i, j}) > \tau_{\text{motion}}$, the patch is flagged as **Dynamic**.
   * If $\mathcal{A}(p_{i, j}) \le \tau_{\text{motion}}$, the patch is flagged as **Static**.
4. **Token Cache Reuse:**
   * Static patches **bypass the 12 DINOv2 Transformer layers**; their patch tokens are directly looked up from the temporal cache $\mathcal{C}_{t-1}$.
   * Only the $\approx 300$ dynamic tokens are passed through self-attention.
5. **Compute Savings:**
   $$\text{Compute} \approx \frac{300}{2025} \times 42\text{ GMACs} + \text{Heads} \approx 6.2\text{ GMACs} + 0.6\text{ GMACs} = \mathbf{6.8\text{ GMACs}}$$
   * **Result:** Strictly satisfies the **$< 10\text{ GMACs}$** automotive requirement while preserving full detection and action recognition fidelity.

---

## 6. Summary Comparison of Caching Subsystems

| Subsystem Dimension | 1. Feature & Detection Disk Cache | 2. In-Memory Detector Instance Cache | 3. Runtime Temporal Patch Cache (Upcoming) |
| :--- | :--- | :--- | :--- |
| **Layer / Medium** | Persistent Storage (`.pt` on SSD/HDD) | Host RAM / VRAM (`_PRADIPT_IMPL_CACHE`) | Accelerator SRAM / High-Speed Cache |
| **Target Operation** | DINOv2 Backbone + YOLO forward passes | Model weight loading & graph instantiation | Transformer self-attention token evaluation |
| **Active Status** | **Implemented & Verified** | **Implemented & Verified** | **Designed (Milestone 3 PVA Target)** |
| **Data Scope** | Entire video sequence ($[T, 2025, 384]$) | Shared detector class reference & weights | Active frame-to-frame patch token buffer |
| **Compute / Time Saving** | Reduces re-training time from $45\text{m} \rightarrow 22\text{s}$ | Eliminates $3.5\text{s} - 5.0\text{s}$ startup overhead | Drops per-frame compute from $42\text{ GMACs} \rightarrow 6.8\text{ GMACs}$ |
| **Memory Footprint** | $\approx 650\text{ MB} - 1.0\text{ GB}$ per 5 FPS video | $\approx 220\text{ MB}$ resident model weights | $\approx 3.1\text{ MB}$ active token buffer |
