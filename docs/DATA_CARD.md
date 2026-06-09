# Data Card: TrashNet (Garbage Classification)

**Project**: SDG 12 Responsible Consumption and Production — AI-Based Smart Waste Classification System

---

## 1. Dataset Identity

| Field | Details |
|-------|---------|
| Name | TrashNet (dataset-resized version) + Kaggle Garbage Classification (trash class supplement) |
| Owner | Gary Thung (gthung@stanford.edu), Mindy Yang (mindyang@stanford.edu), Stanford University CS229 |
| Version | GitHub commit hash of download; no official version update since 2017 |
| Download Date | June 2026 |
| Business Purpose | Original: Stanford CS229 final project. This project: evaluate MobileNetV2 vs. custom CNN on binary image classification in support of SDG 12 |
| Data Card Owner | Group 5 |

---

## 2. Provenance and Licensing

| Field | Details |
|-------|---------|
| Primary Source | GitHub: https://github.com/garythung/trashnet |
| Supplementary Source | Kaggle: https://www.kaggle.com/datasets/mostafaabla/garbage-classification (trash class, 697 images) |
| Original Reference | Thung & Yang, 2016, CS229 Project Report |
| Collection Method | Collected by Gary Thung and Mindy Yang using Apple iPhone; white posterboard background; natural or indoor lighting; Stanford University campus area, California, USA; items self-owned and self-labeled |
| License | MIT License — permits academic and commercial use with attribution |

### Allowed Uses

- Academic research and teaching demonstrations
- Derivative model development (disclosure of dataset use required)
- Commercial use under MIT License terms

### Restrictions

- This project uses the dataset for academic coursework only
- Images may contain commercial logos (Coca-Cola, Nestle, etc.); legal review recommended before commercial deployment
- Not recommended for high-risk decisions (medical, legal, law enforcement)
- Brand logos visible in images remain protected by their respective trademark holders; the dataset license does not cover commercial logo use

---

## 3. Schema and Labels

### 3.1 Image Tensor Schema

| Property | Value |
|----------|-------|
| Color Channels | 3 (RGB) |
| Native Resolution | 512 x 384 pixels |
| Bit Depth | 24-bit (8-bit per channel) |
| DPI | 96 |
| File Format | JPG |
| Model Input Shape | (224, 224, 3) after resize |
| Pixel Value Range | [0.0, 1.0] |

### 3.2 Target Labels

| Label | Index | Source Classes |
|-------|-------|---------------|
| non_garbage (recyclable) | 1 | glass, paper, cardboard, plastic, metal |
| garbage (non-recyclable) | 0 | trash (TrashNet 137 + Kaggle 697) |

Original 6-class structure was remapped to binary classification via `scripts/step1.py`. Labels were originally assigned by the dataset authors based on material type at the time of collection.

### 3.3 Missing Value Rules

This dataset consists of image files rather than tabular data. The following checks apply in place of standard missing value rules:

- No corrupted image files (all .jpg files verified as openable)
- No zero-byte files
- All images have a corresponding class folder label
- Standard tabular NaN / Null handling is not applicable

---

## 4. Protected Attributes

This project performs image classification. Images do not contain personal attributes. The following are equivalent protected attributes at the image level.

### 4.1 Equivalent Protected Attributes

| Attribute | Details |
|-----------|---------|
| Geographic Origin | All images from Stanford University campus, California, USA. May not generalize to Taiwan or other regions. |
| Object Type Bias | garbage class represents 25.9% of the augmented dataset (834 of 3,224), improved from 5.4% in the original TrashNet. Class imbalance ratio is 2.87:1 after Kaggle augmentation. |
| Capture Device | All images captured with Apple iPhone. Performance on Android, DSLR, or industrial cameras is untested. |
| Lighting Condition | Natural sunlight and indoor lighting only. Night, fluorescent, and neon lighting conditions are not covered. |
| Background | White posterboard only. Real-world environments (bins, streets, homes) will cause significant accuracy drop. |
| Brand and Cultural Context | Images contain North American consumer brand packaging. Recognition of non-North American brands and local products is unknown. |

### 4.2 Slice Dimensions for Fairness Testing

Downstream models should report per-slice F1 across the following dimensions:

- Slice by original class: cardboard, glass, metal, paper, plastic, trash (6 slices)
- Slice by class balance: majority class (non_garbage) vs. minority class (garbage)
- Slice by region: if Taiwan local data is added in v2.0, slice by region

---

## 5. Privacy Controls

| Field | Details |
|-------|---------|
| PII Fields | Dataset contains single-object photos only. No faces, names, addresses, or contact information. EXIF metadata may contain capture location and camera serial number. |
| Sensitive Content | No violent, explicit, political, or religious imagery. |
| Masking | EXIF metadata stripped using Pillow or ExifTool during preprocessing. No pixel-level masking required. |
| Retention | Retained until end of semester. Personal cloud storage copies deleted after course completion. Original GitHub repository remains publicly available under MIT License. |
| Deletion Process | Submit deletion request via course platform; confirm data is not included in any downstream release; delete from personal cloud storage and local machines; record deletion log (who, when, why). |
| Model Inversion Risk | Model Inversion Attack may allow partial reconstruction of training images from model weights. |

---

## 6. Quality and Bias

### 6.1 Class Balance

**Original 6-class distribution (TrashNet only, 2,527 images):**

| Class | Count | Percentage |
|-------|-------|------------|
| paper | 594 | 23.5% |
| glass | 501 | 19.8% |
| plastic | 482 | 19.1% |
| metal | 410 | 16.2% |
| cardboard | 403 | 15.9% |
| trash | 137 | 5.4% |

**After binary remapping with Kaggle supplement (3,224 images total):**

| Label | Count | Percentage |
|-------|-------|------------|
| non_garbage (recyclable) | 2,390 | 74.1% |
| garbage (non-recyclable) | 834 | 25.9% |

Class imbalance ratio: approximately 2.87:1 (non_garbage to garbage).

**Impact of imbalance:**

- Overall accuracy can be misleading if the model overpredicts the majority class
- Garbage Recall may be lower without countermeasures

**Mitigations applied in this project:**

- Custom Focal Loss (gamma = 2.0)
- Class weighting with garbage penalty multiplier (1.5x)
After applying these mitigations, MobileNetV2 achieved SPD = -0.0006 and DI = 0.9994 on the test set, both within acceptable fairness thresholds.

### 6.2 Dataset Scale

Total images: 3,224. This is a small dataset relative to common benchmarks:

| Dataset | Training Samples |
|---------|-----------------|
| This project | ~2,257 (train split) |
| CIFAR-10 | 50,000 |
| MNIST | 60,000 |
| ImageNet | 1,200,000 |

Transfer learning (MobileNetV2 pretrained on ImageNet) is strongly recommended over training from scratch given this scale.

### 6.3 Outliers

- Severely blurred or low-quality images: fewer than 1%
- Heavily occluded objects: present in small numbers
- Multiple objects in frame: rare; dataset is designed for single-object images
- Non-white background images: fewer than 1%

### 6.4 Known Limitations

- Geographic bias: all images from the United States; Taiwan-specific items (bubble tea cups, bento boxes, betel nut bags) are absent
- Scene bias: white background only; real-world environments cause significant accuracy drop
- Device bias: iPhone capture characteristics differ from other cameras
- Temporal bias: collected in 2016; new packaging materials (biodegradable plastics) are not included
- Brand bias: images feature North American consumer brands; recognition of local or non-Western brands is unknown

---

## 7. Splits and Lineage

### 7.1 Train / Validation / Test Split

| Split | Proportion | Approximate Count |
|-------|-----------|------------------|
| Train | 70% | 2,257 |
| Validation | 15% | 484 |
| Test | 15% | 483 |

- Split method: random shuffle via `scripts/step1.py` with random seed 2150
- Stratified sampling: not applied in v1.0 (planned for v2.0)

### 7.2 Storage Path

| Location | Path |
|----------|------|
| Primary source | https://github.com/garythung/trashnet |
| Supplementary source | https://www.kaggle.com/datasets/mostafaabla/garbage-classification |
| Project repository | https://github.com/ericliugood/TrashNetMobileNetV2VsCnn |
| Split output directory | `trashnet/trashnew/train/`, `trashnet/trashnew/val/`, `trashnet/trashnew/test/` |

### 7.3 Transformation Lineage

1. Original capture: JPG files at 512 x 384 pixels, 96 dpi, 24-bit RGB
2. Kaggle trash images (697) added to `trashnet/trash/trash/`
3. Six-class folder structure remapped to binary: glass, paper, cardboard, plastic, metal to non_garbage; trash to garbage
4. Random shuffle with seed 2150; 70/15/15 split into train, val, test directories
5. Resize to 224 x 224 pixels for model input
6. Pixel values rescaled to [0.0, 1.0]
7. Training set augmentation only: RandomRotation(20 degrees), RandomHorizontalFlip(), RandomResizedCrop(224, scale=(0.8, 1.0))

---

## 8. Security Controls

### 8.1 Access Control

| Scope | Details |
|-------|---------|
| Data source | Public GitHub repository; anyone may download or fork, including potentially modified versions |
| Training pipeline | Restricted to project team members |

### 8.2 Checksums

| Field | Details |
|-------|---------|
| Checksums provided by authors | No SHA-256 or MD5 provided in the original repository |
| Risk | If the GitHub account is compromised, a poisoned version could replace the original |
| Mitigation | Re-verify dataset integrity on each download |

### 8.3 Poisoning Checks

Based on Machado et al. (2021) Section 3.2.1 (Causative / Poisoning attacks), the following risks and checks apply:

| Risk | Description | Check |
|------|-------------|-------|
| Label Flipping | Attacker modifies class labels of training samples | Manual spot-check of 5% of images to verify label correctness |
| Backdoor Trigger | Attacker embeds hidden trigger patterns in samples | Activation Clustering or Spectral Signatures detection (v2.0 planned) |
| Clean-Label Attack | Imperceptible perturbations added without changing labels | Anomaly detection using isolation forest (v2.0 planned) |
| Distribution Shift Injection | Out-of-distribution samples deliberately mixed into training data | t-SNE or UMAP visualization of class distributions (v2.0 planned) |

### 8.4 Approval

| Role | Party |
|------|-------|
| Data Card Author | Group 5 |
| Data Card Reviewer | Course Instructor |
| Final Approval | Course Instructor |
| Approval Date | Pending instructor sign-off |
| Next Review | Before v2.0 development or upon any dataset change |
