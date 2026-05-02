# Scanner Identification Using Feature-Based Processing and Analysis
### Paper by Khanna, Mikkilineni, Delp — IEEE TIFS, Vol. 4, No. 1, March 2009
### Course Project Notes — IIT Bhilai

---

## Table of Contents
1. [Paper Overview](#1-paper-overview)
2. [Problem Statement](#2-problem-statement)
3. [Background & Motivation](#3-background--motivation)
4. [How a Flatbed Scanner Works](#4-how-a-flatbed-scanner-works)
5. [Sensor Noise — The Core Concept](#5-sensor-noise--the-core-concept)
6. [Method 1 — Correlation-Based Approach](#6-method-1--correlation-based-approach)
7. [Method 2 — Statistical Feature-Based Approach (Main Contribution)](#7-method-2--statistical-feature-based-approach-main-contribution)
8. [The Denoising Filterbank](#8-the-denoising-filterbank)
9. [Dimensionality Reduction and Classification](#9-dimensionality-reduction-and-classification)
10. [Experiments and Results](#10-experiments-and-results)
11. [Robustness Under Postprocessing](#11-robustness-under-postprocessing)
12. [Comparison with Existing Methods](#12-comparison-with-existing-methods)
13. [Key Findings Summary](#13-key-findings-summary)
14. [What You Need to Implement](#14-what-you-need-to-implement)
15. [Presentation Guide](#15-presentation-guide)
16. [Implementation Checklist](#16-implementation-checklist)

---

## Quick Start

Use the project venv before running the pipeline:

```powershell
.venv\Scripts\Activate.ps1
```

Then run the full workflow from the project root:

```bash
./run_pipeline.sh
```

If you want to run the stages manually:

```powershell
python reorganize_dataset.py
python src/experiments/phase3_features.py
python src/experiments/phase4_classification.py
```

Outputs:
- `dataset.csv` at the project root
- `features.csv` at the project root
- `results/confusion_matrix.png`
- `results/report.txt`

---

## 1. Paper Overview

| Attribute | Detail |
|-----------|--------|
| **Title** | Scanner Identification Using Feature-Based Processing and Analysis |
| **Authors** | Nitin Khanna, Aravind K. Mikkilineni, Edward J. Delp |
| **Journal** | IEEE Transactions on Information Forensics and Security |
| **Year** | 2009 |
| **Core Problem** | Given a scanned image, determine *which scanner* produced it |
| **Core Approach** | Extract statistical features from imaging sensor pattern noise → SVM classifier |
| **Dataset** | 11 flatbed desktop scanners (10 different models), multiple resolutions |
| **Best Result** | 99.9% classification accuracy across 10 scanner models at 200 dpi |

---

## 2. Problem Statement

> **Given a digital image that was produced by a flatbed scanner, can we determine the identity (or at minimum the make and model) of the scanner that produced it?**

### Why does this matter?

- Scanned documents are increasingly used as **legal duplicates** (e.g., the Check Clearing for the 21st Century Act in the US allows banks to use scanned check images instead of physical checks).
- Forged or tampered scanned documents are a real forensic concern.
- Digital forensic tools need to establish **origin**, **authenticity**, and **chain of custody** of scanned images.
- If a scanned document is disputed in court, being able to trace it back to a specific scanner (or prove it wasn't from a claimed scanner) is crucial.

### Two levels of the problem:
1. **Scanner Model Identification** — Which make and model produced this image? (e.g., "HP ScanJet 6300c")
2. **Unique Scanner Identification** — Which *specific* device (among multiple scanners of the same model) produced this image?

This paper addresses both, with stronger results at the model level.

---

## 3. Background & Motivation

### Prior Work on Source Camera Identification
Before this paper, most work focused on **cameras**, not scanners. The pioneering work by Lukas, Fridrich, and Goljan (2005–2008) showed that:

- CCD/CMOS sensors introduce unique **Photo Response Non-Uniformity (PRNU)** noise.
- This noise acts like a **fingerprint** of the sensor.
- Correlation of a test image's noise pattern with a known camera's reference pattern can identify the camera.

### Why Scanners are Different from Cameras

| Property | Digital Camera | Flatbed Scanner |
|----------|---------------|-----------------|
| Sensor type | 2D array (full frame) | **1D linear array** (single row) |
| Image capture | One shot, entire frame | Row-by-row scan |
| Scan location | Always uses full sensor | Uses only the portion of bed where paper is placed |
| Noise characteristics | 2D PRNU | Predominantly **row-wise periodic** pattern |
| Reference pattern | 2D array pattern | Must be adapted to 1D or statistical |

The key insight: because a flatbed scanner uses a **1D linear sensor** that sweeps across the document, every row of the image is produced by the **same set of sensor pixels**. This creates a characteristic row-wise periodicity in the noise.

### Problem with Direct Camera Approach Applied to Scanners

When you try to directly use the camera identification approach (Lukas et al.) on scanners, you hit a major problem: **desynchronization**.

- For cameras, the sensor always captures the full frame → reference pattern aligns perfectly.
- For scanners, the document can be placed **anywhere** on the scanner bed → the PRNU seen in the image depends on *which portion* of the scanner bed was used.
- If training images and test images are scanned from different locations, the noise patterns don't align and correlation breaks.

This is the key problem this paper solves.

---

## 4. How a Flatbed Scanner Works

### Imaging Pipeline (Fig. 1 in paper)

```
Original Document
       ↓
   Light Source (CCFL or Xenon lamp)
       ↓
   Mirror-Lens & Imaging Sensor (1D CCD or CIS array)
       ↓
   Amplifier, ADC
       ↓
   Software Post-processing (Color correction, Gamma correction, ...)
       ↓
   Digital Image
```

### Key Mechanical Details

- The scan head (containing lens, mirrors, filters, and 1D sensor) **translates linearly** along the document using a stepper motor, stabilizer bar, and belt.
- The stabilizer bar ensures no wobble in the scan head.
- **Horizontal resolution** = number of elements in the linear CCD (e.g., 1200 sensor elements → 1200 dpi).
- **Vertical resolution** = step size of the motor.

### Scanning at Non-Native Resolution

If you want a 600 dpi scan from a 1200 dpi scanner, there are two approaches:
1. **Subsampling**: read only every other sensor pixel.
2. **Full-res scan + downsampling**: scan at 1200 dpi in memory, then downsample to 600 dpi.

Most quality scanners use method 2 (more accurate). This downsampling significantly affects the sensor noise characteristics.

### Sensor Types Used
- Most desktop scanners use **CCD** (Charge-Coupled Device) sensors.
- Some use **CIS** (Contact Image Sensor) or CMOS.
- The paper shows the method works on **both CCD and CIS** scanners.

---

## 5. Sensor Noise — The Core Concept

### Two Types of Sensor Noise

#### Type 1: Array Defects
- Point defects, hot pixels, dead pixels, pixel traps, column defects, cluster defects
- Cause **large deviations** in pixel values (dead pixels = black, hot pixels = very bright)
- Easily corrected by modern cameras/scanners
- **Not used** by this paper

#### Type 2: Pattern Noise (What this paper uses)
- Any spatial pattern that does **not change significantly** from image to image
- Two main sources:
  - **Fixed Pattern Noise (FPN)**: caused by dark currents (stray currents from sensor substrate into pixels). Varies pixel-to-pixel due to differences in detector size, doping density, and foreign matter from fabrication.
  - **Photo Response Non-Uniformity (PRNU)**: variation in pixel responsivity when illuminated. Due to differences in detector size, spectral response, coating thickness, and manufacturing imperfections.
- Does **not** cause large pixel deviations → hard to correct in-camera
- Most digital cameras do **not** compensate for PRNU
- This makes it a persistent, **unique fingerprint**

### Noise Model

The sensor noise is modeled as:

```
Noise = Fixed Component + Random Component
```

- **Fixed component**: same across all images from the same scanner → acts as a signature
- **Random component**: changes from image to image (lighting fluctuations, etc.)

### Noise Estimation

To extract noise from an image:

```
I_noise = I_original - I_denoised
```

Where `I_denoised` is obtained by applying a denoising filter to the original image. The idea: the denoiser removes "real" image content and what remains is mostly noise.

### Why Row-Periodicity Matters for Scanners

Since every row of a scanned image is produced by the **same physical sensor pixels**, the PRNU pattern repeats identically across all rows. This creates a **strong row-wise periodicity** in the noise. This is a key difference from camera images and is exploited heavily in the feature design.

---

## 6. Method 1 — Correlation-Based Approach

This is the baseline/prior approach adapted from camera identification.

### Training Phase

For each known scanner `s`:
1. Collect `K` training images from that scanner.
2. For each image `I^k`, compute noise: `I^k_noise = I^k - I^k_denoised`
3. Average all noise patterns to get the **2D reference pattern**:

```
Ĩ^array_noise(i,j) = (1/K) * Σ_k I^k_noise(i,j)
```

4. For scanners, also compute the **1D linear row reference pattern** by averaging all rows:

```
Ĩ^linear_noise(1,j) = (1/M) * Σ_i Ĩ^array_noise(i,j)
```

This 1D pattern is more appropriate because scanners use a 1D sensor.

### Testing Phase

For a new image from an unknown scanner:
1. Compute its noise pattern.
2. Compute correlation between the test image's noise and each known scanner's reference pattern:

```
C(X, Y) = (X - X̄)·(Y - Ȳ) / (||X - X̄|| * ||Y - Ȳ||)
```

3. The scanner with the **highest correlation** is declared the source.

### Why This Fails for Scanners

The fundamental problem: If a training image and a test image are scanned from **different locations** on the scanner bed, the PRNU patterns don't overlap. The correlation measure breaks down.

For example, if training images are always placed at the top-left corner of the scanner bed but the test image was scanned from the center of the bed, the PRNU fingerprints seen in both images come from completely different physical sensor regions → zero useful correlation.

### Attempted Fix: Full-Bed Reference Pattern + NCC

One solution: estimate reference patterns for the **entire scanner bed** using large images or multiple tiled images, then use Normalized Cross Correlation (NCC) to find the best match regardless of placement location.

**Problem with this fix**: For a 1200 dpi scanner, the full reference pattern is approximately **10,800 × 14,400 pixels ≈ 500 MB**. This is impractical in terms of storage and computation. This motivates the statistical feature-based approach.

---

## 7. Method 2 — Statistical Feature-Based Approach (Main Contribution)

### Core Insight

Instead of trying to match the actual noise pattern (which fails due to location desynchronization), extract **statistical properties** of the noise that:

1. Are **independent of image content**
2. **Characterize a specific scanner** (differ between scanners, even of the same model)
3. Are **independent of scan location** on the scanner bed

The key realization: even if images are scanned from different locations, the statistical distribution of the noise (mean, std, skewness, kurtosis, etc.) remains characteristic of the scanner, because these statistics are determined by the scanner's manufacturing properties, not by which portion of the bed was used.

### Step-by-Step Feature Extraction

Given an input image `I` of size `M × N` (M rows, N columns):

**Step 1: Compute noise**
```
I_noise = I - I_denoised
```

**Step 2: Compute row average of noise**

This is the average across all M rows for each column position j:
```
Ĩ^r_noise(1, j) = (1/M) * Σ_i I_noise(i, j),   for 1 ≤ j ≤ N
```

This is the estimated "row pattern" — since every row is produced by the same sensor pixels, averaging enhances the fixed component and suppresses the random component.

**Step 3: Compute column average of noise**

This is the average across all N columns for each row position i:
```
Ĩ^c_noise(i, 1) = (1/N) * Σ_j I_noise(i, j),   for 1 ≤ i ≤ M
```

This is used as a comparison (row direction should have higher periodicity than column direction for scanners).

**Step 4: Compute row correlation vector**

For each row `i`, compute correlation between row `i` of the noise and the row average:
```
ρ_row(i) = C(Ĩ^r_noise, I_noise(i, .))
```

This measures how similar each row is to the "average row pattern". For a scanner with consistent row-wise PRNU, this should be **high** — all rows look similar to the average row pattern.

**Step 5: Compute column correlation vector**

For each column `j`, compute correlation between column `j` and the column average:
```
ρ_col(j) = C(Ĩ^c_noise, I_noise(., j))
```

For scanners, this should be **lower** than ρ_row because there's no equivalent periodicity in the column direction.

### Why ρ_row > ρ_col is the Key Discriminator

- **High-quality scanner**: strong consistent row-wise PRNU → ρ_row values are large and relatively uniform. ρ_col is small.
- **Low-quality scanner**: more random noise (e.g., from lighting fluctuations) → ρ_row values drop closer to ρ_col values.
- Different scanners have different characteristic ratios of ρ_row to ρ_col — this ratio encodes the scanner's "quality signature".

### The 15 Features Per Color Channel

From ρ_row, ρ_col, Ĩ^r_noise, and Ĩ^c_noise, extract:

| Feature # | Description |
|-----------|-------------|
| 1 | Mean of ρ_row |
| 2 | Std Dev of ρ_row |
| 3 | Skewness of ρ_row |
| 4 | Kurtosis of ρ_row |
| 5 | Mean of ρ_col |
| 6 | Std Dev of ρ_col |
| 7 | Skewness of ρ_col |
| 8 | Kurtosis of ρ_col |
| 9 | Std Dev of Ĩ^r_noise |
| 10 | Skewness of Ĩ^r_noise |
| 11 | Kurtosis of Ĩ^r_noise |
| 12 | Std Dev of Ĩ^c_noise |
| 13 | Skewness of Ĩ^c_noise |
| 14 | Kurtosis of Ĩ^c_noise |
| 15 | f₁₅ = relative row-column periodicity difference (see below) |

**Feature f₁₅** — Relative Periodicity Difference:
```
f₁₅ = (1 - (mean of ρ_col) / (mean of ρ_row)) × 100
```

Interpretation:
- High positive value → much more row periodicity than column periodicity → characteristic of a scanner.
- Near zero → ρ_row ≈ ρ_col → heavy postprocessing or very low-quality scanner.

### Extending to All Color Channels

- Apply the above to **each of R, G, B channels separately** → 15 × 3 = **45 features**
- Add **6 cross-channel correlation features** (mutual correlations of Ĩ^r_noise across RGB channels, same for Ĩ^c_noise) to capture differences between scanners that use 3 separate linear sensors vs. a single sensor with a tricolor light source.
- Total from one denoising filter: **51-D feature vector**

---

## 8. The Denoising Filterbank

A single denoising filter may miss some types of noise. The paper uses a **bank of 4 different denoising algorithms** applied in parallel:

| Filter | Description |
|--------|-------------|
| **LPA-ICI** | Local Polynomial Approximation — Intersection of Confidence Intervals. Anisotropic, directional multiscale denoiser. Complex but powerful. |
| **Median Filter (3×3)** | Non-linear, good at removing salt-and-pepper noise while preserving edges. |
| **Wiener Filter (3×3)** | Adaptive filter that minimizes mean squared error. Uses local statistics. |
| **Wiener Filter (5×5)** | Same as above but larger neighborhood for smoother denoising. |

Each filter independently processes each color channel. Features from all 4 are **concatenated**:

```
Final feature vector = [51 features from LPA-ICI] + [51 from Median] + [51 from Wiener-3] + [51 from Wiener-5]
                     = 204-D feature vector per image
```

**Why use a filterbank?** Different types of noise are better captured by different filters. The combined filterbank captures the full spectrum of scanner noise characteristics. The paper notes that **linear filters (averaging, Gaussian)** are significantly less effective than the non-linear and adaptive filters used.

---

## 9. Dimensionality Reduction and Classification

### Linear Discriminant Analysis (LDA)

- 204-D is high-dimensional → risk of overfitting, slow training.
- Apply **LDA** to reduce from 204-D to **10-D**.
- LDA finds linear combinations of features that maximize between-class separation while minimizing within-class variance.
- Each of the 10 components is a weighted combination of all 204 original features.
- Choice of 10 dimensions: for 11 scanners, LDA gives at most `num_classes - 1 = 10` discriminant dimensions.

### Support Vector Machine (SVM)

- After LDA, feed the 10-D feature vectors into an **SVM classifier**.
- Uses **Radial Basis Function (RBF) kernel**: `K(x, y) = exp(-γ||x-y||²)`
- Grid search is performed to find optimal `C` (regularization) and `γ` (RBF kernel width) parameters.
- Features are **scaled to [-1, 1]** before SVM training (standard practice for SVMs).
- Library used: **LIBSVM** (open source, widely available in Python via `scikit-learn` or directly).
- For native resolution images: majority voting over subimage decisions gives the final scanner prediction.

### Complete Pipeline Diagram

```
Input Scanned Image
        ↓
┌───────────────────────┐
│  Denoising Filterbank │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│  │LPA  │ │Med  │ │Wie3 │ │Wie5 │ │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ │
└─────┼────────┼────────┼────────┼───┘
      ↓        ↓        ↓        ↓
  51-D feat  51-D feat  51-D feat  51-D feat
      └────────┴────────┴────────┘
                    ↓
           Concatenate → 204-D
                    ↓
              LDA → 10-D
                    ↓
           SVM Classifier
                    ↓
           Source Scanner ID
```

---

## 10. Experiments and Results

### Scanner Dataset (Table I)

| ID | Make/Model | Sensor | Native Resolution |
|----|-----------|--------|-------------------|
| S₁ | Epson Perfection 4490 Photo | CCD | 4800 DPI |
| S₂ | HP ScanJet 6300c-1 | CCD | 1200 DPI |
| S₃ | HP ScanJet 6300c-2 | CCD | 1200 DPI |
| S₄ | HP ScanJet 8250 | CCD | 4800 DPI |
| S₅ | Mustek 1200 III EP | CCD | 1200 DPI |
| S₆ | Visioneer OneTouch 7300 | CIS | 1200 DPI |
| S₇ | Canon LiDE 25 | CIS | 1200 DPI |
| S₈ | Canon LiDE 70 | CIS | 1200 DPI |
| S₉ | OpticSlim 2420 | CIS | 1200 DPI |
| S₁₀ | Visioneer OneTouch 7100 | CCD | 1200 DPI |
| S₁₁ | Mustek ScanExpress A3 | CCD | 600 DPI |

**Note**: S₂ and S₃ are **identical make and model** — used to test whether the method can distinguish two units of the same device.

### Experiment 1: Scan-Area Independence (Native Resolution)

- Split each scanner image into 1024×768 pixel blocks.
- **Test**: Train on images from Column 1 of scanner bed, test on Column 2.
- **Result**: 95% average classification accuracy despite using completely different scan areas.
- **Conclusion**: The proposed features are largely independent of scan location.

### Experiment 2: Native Resolution TIFF (Table V)

- 7 scanners, 400 subimages each, 50% train / 50% test.
- **Result: 100% final classification accuracy** (subimage accuracy ≈ 95-100%).
- Final decision by majority voting over all subimage decisions.

### Experiment 3: Native Resolution JPEG (Table VIII)

- Same setup but images JPEG compressed at Q=70.
- Dedicated classifier (trained on JPEG images).
- **Result: 92% accuracy** — significant but manageable degradation.

### Experiment 4: Nonnative Resolution 200 dpi TIFF (Table XIII)

- All 11 scanners, 108 images each, 80 training / 28 testing.
- S₂ and S₃ (same model) merged into one class.
- **Result: 99.9% accuracy across 10 scanner models**.

### Experiment 5: Same-Model Discrimination (Tables XI, XII)

- Can S₂ and S₃ (identical HP ScanJet 6300c units) be distinguished?
- **Native resolution**: Yes — high accuracy.
- **200 dpi TIFF**: ~90% — degraded but possible.
- **200 dpi JPEG**: ~75% — difficult, dropping to model-level only.

---

## 11. Robustness Under Postprocessing

### JPEG Compression

Average classification accuracy (proposed scheme):

| Format | Accuracy |
|--------|----------|
| TIFF (uncompressed) | 99.9% |
| JPEG Q=90 | 97.4% |
| JPEG Q=80 | 95.7% |
| JPEG Q=70 | 93.3% |

The general classifier (trained on all quality factors, tested on unknown quality) achieves **92.3%** — practical and useful.

### Image Sharpening

- Uses weighted median filter-based sharpening (τ=0.2).
- **Result**: Proposed scheme is **unaffected** (99.8% general classifier accuracy).
- IQM-based scheme shows significant drop.

### Contrast Stretching

- Uses a piecewise contrast curve with threshold T=20.
- **Result**: Proposed scheme is **unaffected** (99.8% general classifier accuracy, same as above — tested jointly with sharpening).

### Why are the Features Robust?

The statistical features (mean, std, skewness, kurtosis of noise correlations) capture deep structural properties of the noise pattern. Operations like sharpening and contrast stretching change pixel values globally but do not destroy the fundamental row-periodicity relationship in the noise, because the noise is only a small fraction of the total signal.

---

## 12. Comparison with Existing Methods

### Three Methods Compared

1. **Proposed scheme** (Khanna et al.): 204-D noise features from filterbank → LDA → SVM
2. **IQM-based scheme** (Kharrazi et al.): 28-D Image Quality Metrics + wavelet features → LDA → SVM (originally designed for cameras)
3. **Gou et al.'s scheme**: 60-D features (noise stats + wavelet + prediction error) → LDA → SVM

### Summary of Classification Accuracies (200 dpi images, 10 scanner models)

| Condition | Proposed | Gou et al. | IQM-based |
|-----------|----------|------------|-----------|
| TIFF (uncompressed) | **99.9%** | 96.6% | 88.4% |
| JPEG Q=90 dedicated | **97.4%** | ~85% | ~70% |
| JPEG Q=80 dedicated | **95.7%** | ~82% | ~65% |
| JPEG Q=70 dedicated | **93.3%** | 80.8% | 68.6% |
| JPEG general classifier | **92.3%** | 57.7% | 75.0% |
| Sharpening + Contrast (general) | **99.8%** | 95.4% | 79.7% |

**Key finding**: Better feature selection is the source of higher accuracy. The proposed features outperform both alternatives consistently, especially under JPEG compression where Gou et al.'s general classifier drops dramatically to 57.7%.

### Effect of Training Set Size (Figs. 10–11)

- With just **20 training images** per scanner class, the proposed scheme achieves ~90% accuracy on nonnative resolution images.
- With 80 training images, accuracy plateaus near 99.9%.
- The IQM-based scheme requires many more training images to reach comparable performance.

---

## 13. Key Findings Summary

1. **Statistical features of sensor pattern noise are effective scanner fingerprints** — they capture the characteristic PRNU and scanning noise of each device.

2. **The denoising filterbank matters** — using all 4 denoising algorithms together consistently outperforms any single one, especially under JPEG compression.

3. **Scan-area independence is achieved** — the proposed features do not require images to be scanned from the same location on the scanner bed.

4. **Model-level identification is highly accurate** — 99.9% for 10 scanner models at 200 dpi.

5. **Unique scanner identification is possible at native resolution** — even two HP ScanJet 6300c units can be distinguished.

6. **Robustness to postprocessing** — JPEG compression degrades performance gracefully; sharpening and contrast stretching have minimal impact.

7. **Practical with limited training data** — good results achievable with as few as 20 training images per scanner.

---

## 14. What You Need to Implement

### Core Implementation Tasks

#### A. Data Collection (if doing from scratch)
- Scan the same set of images using multiple different scanners.
- Save both as TIFF (uncompressed) and JPEG (at Q=70, 80, 90).
- Scan from different locations on the scanner bed (for scan-area independence experiments).

> **If you don't have access to multiple physical scanners**, use the publicly available scanner forensics datasets. Search for "Dresden Image Database" or similar forensics image datasets that include scanner images.

#### B. Preprocessing
```python
# Pseudocode
image = load_image(path)  # Load as RGB
image_float = image / 255.0  # Normalize

# For native resolution: slice into 1024×768 blocks
blocks = slice_image(image, block_size=(1024, 768))
```

#### C. Denoising (Filterbank)
Implement or use library versions of all 4 denoising filters:

```python
from scipy.ndimage import median_filter
from scipy.signal import wiener
# LPA-ICI is harder — use BM3D or a suitable substitute, or find Python implementations

def apply_filterbank(channel):
    denoised_lpaci = lpa_ici_denoise(channel)  # or BM3D approximation
    denoised_median = median_filter(channel, size=3)
    denoised_wiener3 = wiener(channel, mysize=3)
    denoised_wiener5 = wiener(channel, mysize=5)
    return [denoised_lpaci, denoised_median, denoised_wiener3, denoised_wiener5]
```

#### D. Noise Extraction
```python
def extract_noise(image_channel, denoised_channel):
    return image_channel - denoised_channel
```

#### E. Feature Extraction (51-D per denoising filter)

```python
import numpy as np
from scipy.stats import skew, kurtosis

def extract_features_from_noise(noise):
    """
    noise: 2D array of shape (M, N) — single color channel
    returns: 15-D feature vector
    """
    M, N = noise.shape
    
    # Row average
    row_avg = np.mean(noise, axis=0)  # shape (N,)
    # Col average
    col_avg = np.mean(noise, axis=1)  # shape (M,)
    
    # Row correlation: correlation of each row with the row average
    rho_row = np.array([
        np.corrcoef(row_avg, noise[i, :])[0, 1]
        for i in range(M)
    ])
    
    # Col correlation: correlation of each col with the col average
    rho_col = np.array([
        np.corrcoef(col_avg, noise[:, j])[0, 1]
        for j in range(N)
    ])
    
    # Replace NaNs (constant rows/cols)
    rho_row = np.nan_to_num(rho_row)
    rho_col = np.nan_to_num(rho_col)
    
    # Features 1-8: stats of rho_row and rho_col
    f1 = np.mean(rho_row)
    f2 = np.std(rho_row)
    f3 = skew(rho_row)
    f4 = kurtosis(rho_row)
    f5 = np.mean(rho_col)
    f6 = np.std(rho_col)
    f7 = skew(rho_col)
    f8 = kurtosis(rho_col)
    
    # Features 9-11: stats of row_avg
    f9  = np.std(row_avg)
    f10 = skew(row_avg)
    f11 = kurtosis(row_avg)
    
    # Features 12-14: stats of col_avg
    f12 = np.std(col_avg)
    f13 = skew(col_avg)
    f14 = kurtosis(col_avg)
    
    # Feature 15: relative row-col periodicity difference
    mean_rho_row = f1 if f1 != 0 else 1e-9
    f15 = (1 - f5 / mean_rho_row) * 100
    
    return np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15])

def extract_51d_features(noise_rgb):
    """
    noise_rgb: (M, N, 3) noise for all 3 channels
    returns: 51-D feature vector
    """
    feats = []
    channel_row_avgs = []
    channel_col_avgs = []
    
    for c in range(3):
        noise_c = noise_rgb[:, :, c]
        feats.append(extract_features_from_noise(noise_c))
        channel_row_avgs.append(np.mean(noise_c, axis=0))
        channel_col_avgs.append(np.mean(noise_c, axis=1))
    
    # 6 cross-channel correlations (row avgs: R-G, R-B, G-B; col avgs: R-G, R-B, G-B)
    cross_row = [
        np.corrcoef(channel_row_avgs[0], channel_row_avgs[1])[0, 1],
        np.corrcoef(channel_row_avgs[0], channel_row_avgs[2])[0, 1],
        np.corrcoef(channel_row_avgs[1], channel_row_avgs[2])[0, 1],
    ]
    cross_col = [
        np.corrcoef(channel_col_avgs[0], channel_col_avgs[1])[0, 1],
        np.corrcoef(channel_col_avgs[0], channel_col_avgs[2])[0, 1],
        np.corrcoef(channel_col_avgs[1], channel_col_avgs[2])[0, 1],
    ]
    
    all_feats = np.concatenate(feats + [cross_row, cross_col])  # 45 + 6 = 51
    return all_feats
```

#### F. Final Feature Vector Assembly (204-D)

```python
def extract_204d_feature(image_rgb):
    """Complete pipeline for one image"""
    features_all = []
    for denoised in apply_filterbank_rgb(image_rgb):  # 4 denoised versions
        noise = image_rgb - denoised
        features_all.append(extract_51d_features(noise))
    return np.concatenate(features_all)  # 204-D
```

#### G. LDA + SVM Classification

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Build pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(-1, 1))),
    ('lda', LinearDiscriminantAnalysis(n_components=10)),
    ('svm', SVC(kernel='rbf', C=best_C, gamma=best_gamma))
])

# Grid search for SVM parameters
from sklearn.model_selection import GridSearchCV
param_grid = {'svm__C': [0.1, 1, 10, 100], 'svm__gamma': [0.001, 0.01, 0.1, 1]}
grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)
```

#### H. Native Resolution: Majority Voting

```python
def predict_scanner_for_full_image(full_image, model, block_size=(1024, 768)):
    blocks = slice_image(full_image, block_size)
    block_features = [extract_204d_feature(b) for b in blocks]
    block_predictions = model.predict(block_features)
    # Majority vote
    from scipy.stats import mode
    return mode(block_predictions).mode[0]
```

### Experiments to Reproduce

| Priority | Experiment | Details |
|----------|-----------|---------|
| ⭐ Essential | 200 dpi TIFF classification | Train on 80, test on remaining |
| ⭐ Essential | Comparison with baseline | Show your method vs. simpler one |
| ⭐ Essential | JPEG robustness | Test at Q=70, 80, 90 |
| 🔶 Important | Scan-area independence | Train col 1, test col 2 |
| 🔶 Important | Training size effect | Vary from 10-80 and plot accuracy |
| 🔵 Optional | Sharpening/contrast robustness | If time permits |

---

## 15. Presentation Guide

### Suggested Slide Structure (15-20 slides)

| Slide | Title | Content |
|-------|-------|---------|
| 1 | Title slide | Paper name, authors, your name, course |
| 2 | Motivation | Why scanner forensics? Legal documents, fraud detection |
| 3 | Problem Statement | Source scanner identification — two levels |
| 4 | How Scanners Work | Pipeline diagram, 1D sensor concept |
| 5 | Sensor Noise Types | FPN vs PRNU, why pattern noise is useful |
| 6 | Prior Work: Camera ID | Lukas et al. correlation approach |
| 7 | Why Camera → Scanner Doesn't Work | Desynchronization problem, 500 MB reference |
| 8 | Key Insight: Statistical Features | Location-invariant statistics of noise |
| 9 | Feature Extraction Pipeline | ρ_row, ρ_col, the 15 features (with equations) |
| 10 | The Denoising Filterbank | 4 filters, 204-D concatenation |
| 11 | LDA + SVM | Dimensionality reduction, classification |
| 12 | Dataset | 11 scanners table, scanning setup |
| 13 | Main Results | Tables V, XIII — highlight key numbers |
| 14 | Robustness Results | Fig 8 bar chart, JPEG/sharpening results |
| 15 | Comparison with Baselines | Your table from Section 12 above |
| 16 | Your Implementation | What you implemented, code snippets |
| 17 | Your Results | Your confusion matrices, accuracy plots |
| 18 | Conclusions | Key takeaways, limitations, future work |

### Key Points to Emphasize in Presentation

1. **The desynchronization problem** — this is the crux of why a naive extension of camera identification fails for scanners. Make sure you explain this clearly with a diagram.

2. **Why row-periodicity is the key feature** — spend time on the intuition behind ρ_row > ρ_col for scanners. This is the core insight of the paper.

3. **The filterbank design** — mention Fig. 12 from the paper which shows each individual filter's performance vs. the combined filterbank, especially under JPEG compression. The combined filterbank maintains >90% even at Q=70.

4. **The practical implications** — this method can work with images as small as a few hundred pixels (200 dpi scan), which is what you'd typically find in real-world documents.

5. **Limitations to mention**:
   - Same-model discrimination degrades at low resolution
   - Heavy JPEG compression (Q < 70) would likely cause further degradation
   - The LPA-ICI filter is complex to implement — you may substitute with BM3D
   - Adversarial attacks (explicitly adding/removing noise) would defeat the method

---

## 16. Implementation Checklist

### Must-Do (for passing)
- [ ] Implement noise extraction (denoising + subtraction) for at least 1-2 filters
- [ ] Implement the 15-feature extraction per color channel
- [ ] Build the 51-D feature vector (including cross-channel correlations)
- [ ] Train and evaluate an SVM classifier
- [ ] Generate a confusion matrix
- [ ] Report classification accuracy

### Should-Do (for good marks)
- [ ] Implement all 4 denoising filters in the filterbank
- [ ] Implement LDA dimensionality reduction
- [ ] Test on JPEG-compressed images (robustness experiment)
- [ ] Plot accuracy vs. training set size
- [ ] Compare with at least one baseline method (e.g., just using raw noise statistics without the filterbank)
- [ ] Document scan-area independence test

### Nice-to-Have (for excellent marks)
- [ ] Implement the correlation-based approach and show why it fails for different scan locations
- [ ] Full pipeline end-to-end with majority voting for native resolution images
- [ ] Visualization of ρ_row and ρ_col distributions for different scanners
- [ ] t-SNE or LDA scatter plots showing feature cluster separation

---

## Quick Reference: Libraries You'll Need

```bash
pip install numpy scipy scikit-learn matplotlib pillow opencv-python
# For LPA-ICI approximation or BM3D:
pip install bm4d  # or bm3d
```

```python
# Core imports
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import median_filter
from scipy.signal import wiener
from scipy.stats import skew, kurtosis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
```

---

*Notes compiled from: Khanna, N., Mikkilineni, A.K., Delp, E.J. "Scanner Identification Using Feature-Based Processing and Analysis." IEEE Transactions on Information Forensics and Security, Vol. 4, No. 1, March 2009.*