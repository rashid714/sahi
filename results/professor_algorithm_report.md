# Professor Algorithm Report (Elaborate) — ROI-Gated SAHI Using Pseudo Labels to Avoid Background Scanning

Date: 2026-01-17

This document is written in the professor’s framing and focuses on the **algorithm** (not environment/stack):

- **What SAHI does while scanning** (baseline behavior)
- **Why that wastes compute** on irrelevant background
- **What we changed** to follow the professor’s requirement
- **How the new scanning works step-by-step**, including merging + coordinate shifting
- **Why it is faster**, what can go wrong, and how we guarantee quality

## Update log (maintain this)

- 2026-01-17 — First professor-style algorithm report.
- 2026-01-17 — Added safety mechanisms in ROI-gated mode (aligned proposer/refiner + optional fallback) and corrected image color order for SAHI to improve correctness.

## 1) Baseline: how SAHI scans (what happened previously)

### 1.1 SAHI is a sliding-window tile scanner

Given an image, SAHI divides it into overlapping tiles (slices). For each tile:

1) Run the detector on the tile
2) Collect all tile detections
3) Merge them (NMS) back into one set of detections

Conceptually:

```
Full image
┌──────────────────────────────────────┐
│ [tile][tile][tile][tile] ...         │  scan left→right, top→bottom
│ [tile][tile][tile][tile] ...         │
│ [tile][tile][tile][tile] ...         │
└──────────────────────────────────────┘
```

### 1.2 Why this is time-inefficient (professor’s point)

If most of the image is background (trees/sky/walls/empty road), SAHI still executes the detector on those tiles.

So the previous behavior is:

- **Scan everywhere** → strong recall for small objects
- **But slow** when objects occupy a small portion of the image

Professor requirement (in plain words):

> “Do not slide over the whole image. Find the foreground, and scan only that. Background should be ignored.”

## 2) Our change: foreground-first scanning using pseudo labels

We redesigned scanning into two stages:

1) **Proposer (fast)** → suggests where objects might exist (pseudo labels)
2) **Refiner (accurate)** → runs SAHI slicing only inside those suggested regions

This gives the desired behavior:

- Background is treated as **0** (no scanning)
- Foreground regions are **1** (scan)

Crucially, we avoid training a segmentation network by building the “foreground mask” from pseudo-label boxes.

## 3) The ROI-Gated SAHI algorithm (step-by-step, detailed)

### 3.1 Inputs and outputs

Inputs:

- Image $I$ with width $W$ and height $H$
- Proposer detector $M_p$ (fast; low-res)
- Refiner detector $M_r$ wrapped for SAHI slicing
- Slice size $S$, overlap ratio $o$
- ROI controls: margin $m$, max ROIs $K$, ROI-merge IoU threshold $\tau_{merge}$

Output:

- Final detections $D$ in full-image coordinates

### 3.2 Stage A — Proposer pass: produce pseudo labels

Run the proposer once on the whole image (cheap):

- Lower threshold (higher recall)
- Smaller input resolution

This outputs pseudo labels:

$$
B = \{b_i\},\quad b_i = (x_1, y_1, x_2, y_2, c_i, s_i)
$$

where $c_i$ is class id, $s_i$ is score.

Important note (quality):

- On custom datasets, the proposer should be **aligned to the same classes** as the refiner. If proposer uses unrelated classes, it proposes wrong ROIs.

### 3.2.1 Target class filtering (“Target class IDs”) — what it means

Sometimes we do not want *all* detectable classes. We want to focus the scan on only the classes relevant to the experiment (and ignore everything else).

**Target class IDs** means:

- Only keep detections whose class id is in a chosen list (e.g., only “person” and “car”).
- Use that same class filter consistently in both stages:
   - Proposer (pseudo labels → ROIs)
   - Refiner (SAHI inside ROIs)

This is useful because:

- It prevents ROIs from being created for irrelevant classes.
- It avoids wasting ROI compute on objects we do not care about.

Concrete examples for the common COCO label set (the default weights used in this project):

- `0` → person
- `2` → car
- `5` → bus
- `7` → truck
- `15` → cat
- `16` → dog

Examples of Target class IDs lists:

- `0` (only people)
- `2` (only cars)
- `0,2` (people + cars)
- `0,2,5,7` (people + car + bus + truck)

For a custom-trained model, the class ids depend on the dataset’s class order (typically `0..(N-1)`).

### 3.3 Stage B — Convert pseudo labels into ROIs (foreground)

We convert $B$ into a small set of ROIs:

1) **Expand** each box by margin $m$:

$$
(x_1, y_1, x_2, y_2) \to (x_1-m, y_1-m, x_2+m, y_2+m)
$$

2) **Clip** to image bounds
3) **Merge** overlapping ROIs using IoU threshold $\tau_{merge}$
4) **Keep top-$K$** ROIs (cap compute)

Result:

$$
R = \{r_j\},\quad r_j = (X_1, Y_1, X_2, Y_2)
$$

Interpretation:

- $R$ is the foreground mask (box-based). Only areas inside these ROIs will be scanned.

### 3.4 Stage C — Refiner pass: SAHI slicing inside ROIs only

For each ROI $r_j$:

1) Crop the image: $I_j = I[Y_1:Y_2, X_1:X_2]$
2) Run SAHI sliced inference on $I_j$ (tile scan within ROI)
3) Collect detections $D_j$ in ROI-local coordinates

Conceptually:

```
Full image
┌──────────────────────────────────────┐
│ background (ignored)                 │
│      ┌──────── ROI ────────┐         │
│      │ [tile][tile][tile]  │  scan only here
│      │ [tile][tile][tile]  │
│      └─────────────────────┘         │
└──────────────────────────────────────┘
```

### 3.5 Stage D — Shift ROI detections back to full-image coordinates

Each detection from ROI $r_j$ is predicted relative to ROI origin $(X_1, Y_1)$.

So for any predicted box $(x_1, y_1, x_2, y_2)$ inside ROI, we shift:

$$
(x_1, y_1, x_2, y_2) \to (x_1+X_1, y_1+Y_1, x_2+X_1, y_2+Y_1)
$$

Now all detections are in full-image coordinates.

### 3.6 Stage E — Global merge (final NMS)

Because multiple ROIs can overlap, the same object can be detected more than once.

We run a final full-image NMS with IoU threshold $\tau_{nms}$:

- Input: $D = \cup_j D_j$
- Output: de-duplicated final detections

### 3.7 Safety mechanisms (to avoid “only one corner” or “wrong region”)

ROI-gating is a *speed optimization* and depends on proposer recall. To keep quality high:

1) **Aligned proposer/refiner weights** (default)
   - Proposer uses the same weights as refiner so ROIs match the true target classes.

2) **Fallback when ROI-gated result is empty** (optional)
   - If proposer produces no ROIs or ROI refinement yields no detections, we optionally run full-image SAHI once.

These mechanisms ensure the method follows the professor’s intent without sacrificing correctness.

## 4) Why it is faster (scan complexity explained)

Let slice size be $S$ and overlap ratio $o$.

Naive SAHI full-image tile count is approximately:

$$
N \approx \left\lceil \frac{W}{S(1-o)} \right\rceil \cdot \left\lceil \frac{H}{S(1-o)} \right\rceil
$$

Let the merged ROIs cover fraction $r$ of the image area:

$$
r = \frac{\text{area}(\cup R)}{WH}
$$

Then ROI-gated SAHI scans roughly $rN$ tiles (plus small overhead from the proposer).

Key interpretation:

- If the image is mostly background, $r \ll 1$ → big speedup
- If objects are everywhere, $r \approx 1$ → speedup is small (expected)

## 5) What “world-class” demonstration looks like (what to show)

For each test image, show these artifacts side-by-side:

1) **Full SAHI (baseline)** result + time
2) **ROI-gated SAHI** result + time
3) **ROI overlay** visualization (cyan boxes)
4) Timing breakdown:
   - proposer time
   - refine time (SAHI inside ROIs)
   - postprocess time

This directly proves:

- We are not scanning background tiles
- We are scanning only foreground ROIs
- We are faster while keeping detection quality

## 6) Common failure modes (and how to fix them)

1) **Proposer misses objects → ROI-gated misses them**
   - Fix: use aligned proposer weights, increase proposer imgsz, lower proposer conf, increase ROI margin, increase max ROIs.

2) **ROIs cover only one area (e.g., corner)**
   - Fix: aligned proposer/refiner weights; increase max ROIs; reduce ROI merge threshold; increase ROI margin.

3) **Wrong-looking detections (nonsense boxes)**
   - Fix: ensure correct image color order is passed into the detector (NumPy inputs are typically BGR for YOLO); then re-test full SAHI vs ROI-gated.

4) **Dense scenes**
   - Expectation: speedup may be limited because the foreground covers most of the image.

## 7) Where this exists in the project

- Streamlit image backend: “SAHI (ROI-gated via pseudo labels)”
- Files:
  - ui/streamlit_app.py (implementation + timing + ROI overlay)
  - results/roi_sahi_summary.md (bench results table)

This algorithm matches the professor’s requirement: **avoid full-image sliding scan by defining foreground from pseudo labels and scanning only those regions.**
