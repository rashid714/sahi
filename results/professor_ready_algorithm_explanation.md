# Project Algorithm Explanation 

Date: 2026-01-18

Title: **Making SAHI Faster by Scanning Only Foreground (ROI-Gated Sliced Inference using YOLO Pseudo-Labels)**

## 1) What problem we are solving

SAHI works well for small objects because it **cuts the image into many overlapping tiles** and runs detection on each tile.

However, the key issue is:

- SAHI “slides over the whole image”
- That means it also scans large **background regions** (trees, sky, empty road, walls, etc.)
- This wastes compute and time when the real objects exist only in a small part of the image

So the goal is:

- **Do not scan background**
- **Only scan the foreground regions** where objects are likely to exist

We implemented exactly that.

---

## 2) Baseline: what SAHI does by default (why it can be slow)

SAHI’s default behavior is a sliding tiled scan:

1. Split the full image into many overlapping tiles
2. Run the detector on every tile
3. Merge all tile detections back into one final set (using NMS merging)

A simple visual idea:

```
Full image
┌──────────────────────────────────────────┐
│ tile tile tile tile tile tile ...        │
│ tile tile tile tile tile tile ...        │  scans everything
│ tile tile tile tile tile tile ...        │  even background
└──────────────────────────────────────────┘
```

Why this wastes time:

- If the image is mostly background, SAHI still runs the detector on the background tiles.
- The detector is the expensive part, so scanning irrelevant tiles wastes most compute.

---

## 3) The key idea: “Foreground-first scanning” without training a segmentation model

We changed the scanning strategy:

- First, do a **fast, cheap pass** to guess where objects might exist.
- Then, run SAHI **only inside those regions**.

We do NOT train a segmentation model.

Instead, we use **pseudo-label bounding boxes** from a fast YOLO pass.

Pseudo-label means:

- We run a detector once on the whole image.
- The bounding boxes it returns are used as “hints” to tell us where the foreground is.

Those hint boxes are then converted into a few bigger rectangles (regions of interest / ROIs).

---

## 4) The new algorithm 

### Step A — Fast proposer pass (create pseudo labels)

We run YOLO once on the full image as a “proposer”.

This proposer is configured for **high recall**:

- Lower confidence threshold (so it misses fewer objects)
- Optionally a smaller input resolution (faster)

Output from this step:

- A list of bounding boxes that probably contain objects

Important quality rule:

- The proposer should match the same classes as the final detector.
- Best option: **use the same weights for proposer and refiner** so the class meanings are aligned.

### Step B — Convert pseudo-label boxes into scanning regions (ROIs)

We take the proposer boxes and convert them into a small set of ROIs:

1. Expand each box slightly (add margin) so we include nearby pixels
2. Merge boxes that overlap, to avoid scanning many tiny regions
3. Limit the number of ROIs (so runtime stays bounded)

Result:

- A few ROIs that represent the “foreground”
- Everything outside these ROIs is considered background and is not scanned

### Step C — Run SAHI sliced inference only inside each ROI

For each ROI:

1. Crop that ROI area from the original image
2. Run SAHI sliced inference on that crop
   - SAHI still tiles (but only within the ROI)
3. Collect detections from ROI tiles

Visual idea:

```
Full image
┌──────────────────────────────────────────┐
│ background (ignored)                     │
│         ┌──────── ROI ────────┐          │
│         │ tile tile tile ...  │  scan only here
│         │ tile tile tile ...  │
│         └─────────────────────┘          │
└──────────────────────────────────────────┘
```

### Step D — Convert ROI detections back to full-image coordinates

Detections from a crop are measured relative to the crop’s top-left corner.

So we shift each ROI detection back into the full image coordinate system.

(Example: if the ROI begins 300 pixels from the left and 100 pixels from the top, every detection in that ROI is shifted by +300 in x and +100 in y.)

### Step E — Final merge across all ROIs (global NMS)

If ROIs overlap, the same object might be detected twice.

So we do one final merging step on the full image:

- run non-maximum suppression (NMS) across all ROI detections
- keep the best box per object

Final output:

- One clean set of detections for the full image

---

## 5) Why this is faster 

The expensive part is running the detector.

Baseline SAHI:

- Runs the detector on tiles covering the entire image area.

ROI-gated SAHI:

- Runs the detector only on tiles covering the ROIs.

So speed improvement depends on how much of the image is foreground.

### Practical example (simple numbers)

Suppose:

- A full image requires about 200 tiles for SAHI.
- The actual objects occupy only about 25% of the image.

Then ROI-gated SAHI roughly scans:

- 25% of 200 tiles = about 50 tiles

Plus a small overhead:

- One fast proposer pass

So runtime goes down dramatically when the background dominates.

If objects are everywhere, ROIs cover most of the image, and speedup naturally becomes smaller (expected).

---

## 6) How we guarantee the method doesn’t become worse (safety mechanisms)

ROI-gating is a speed optimization, but it relies on the proposer finding objects.

So we added safety mechanisms to prevent “missing everything” or scanning only a wrong corner:

1) **Use the same weights for proposer and refiner (recommended)**
- This prevents class mismatch.
- It makes ROIs follow the real target objects.

2) **Fallback option (when ROI-gated fails)**
- If ROI-gated scanning produces no detections (or no ROIs), we can automatically run full-image SAHI.
- This ensures ROI-gating never permanently hides objects.

This makes the method robust:

- It still follows the main requirement (avoid background scanning),
- but it has a correctness safety net.

---

## 7) “Target class IDs” (class filtering) — what it means and why it matters

Sometimes we only care about certain classes (for example, only cars and people).

“Target class IDs” means:

- Only keep detections whose class id is in the chosen list.
- Apply that filter consistently:
  - proposer stage (so ROIs come only from relevant classes)
  - refiner stage (so final detections show only relevant classes)

Examples for COCO weights (common default):

- 0 = person
- 2 = car
- 5 = bus
- 7 = truck
- 15 = cat
- 16 = dog

Example filters:

- `0` → detect only people
- `2` → detect only cars
- `0,2` → people + cars
- `0,2,5,7` → people + car + bus + truck

For a custom-trained model, ids depend on the dataset class order (usually starting from 0).

---

## 8) What to show in a presentation (the “proof” that background is ignored)

For each sample image, show these side-by-side:

1) Full SAHI result + total time
2) ROI-gated SAHI result + total time
3) ROI overlay image (cyan rectangles showing where scanning happened)
4) Time breakdown:
   - proposer time
   - ROI SAHI time
   - merge/postprocess time

This proves three things clearly:

- We are not scanning the whole image anymore.
- The foreground areas are scanned deeply (SAHI within ROIs).
- Runtime improves when background dominates.

---

## 9) Common failure modes and how we address them

1) Proposer misses objects
- Effect: ROIs do not include objects → ROI-gated misses them
- Fix: lower proposer confidence, increase proposer resolution, increase ROI margin, increase max ROIs

2) ROIs only appear in one corner
- Usually due to proposer mismatch (wrong model/classes) or overly aggressive merging
- Fix: use same weights for proposer/refiner, reduce ROI merge aggressiveness, increase max ROIs

3) Strange/wrong detections
- Often caused by wrong image color format in the inference pipeline
- Fix: ensure correct color order is passed to the model (this was corrected in the implementation)

4) Dense scenes
- Foreground is most of the image
- ROI-gated speedup will be smaller (this is expected and correct)

---

## 10) Summary 
We kept SAHI’s advantage for small objects but removed its biggest inefficiency by adding a foreground-first step. A fast YOLO proposer creates pseudo-label boxes, these are merged into a small number of ROIs, and SAHI slicing runs only inside those ROIs. Background tiles are skipped entirely, reducing wasted compute. We also added safeguards (aligned proposer/refiner and optional fallback) so the faster method stays reliable. This satisfies the main requirement: do not slide over the full image; scan only the foreground.
