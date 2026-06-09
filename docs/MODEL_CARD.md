# Model Card: MobileNetV2 TrashBinary v1.0

**Project**: SDG 12 Responsible Consumption and Production — AI-Based Smart Waste Classification System

---

## 1. Model Identity

| Field | Details |
|-------|---------|
| Name | MobileNetV2 TrashBinary v1.0 |
| Owner | Group 5 |
| Version | v1.0 |
| Algorithm | Deep Learning — Convolutional Neural Network (CNN) binary classifier. Base architecture: MobileNetV2 (Sandler et al., 2018, CVPR) with ImageNet pretrained weights and custom classification head |
| Created | June 2026 |
| License | Academic use only; non-commercial |
| Repository | https://github.com/ericliugood/TrashNetMobileNetV2VsCnn |

### Architecture

**Base model**: MobileNetV2 pretrained on ImageNet (IMAGENET1K_V1). All feature extraction layers frozen.

**Custom classification head**:

```
AdaptiveAvgPool2d(1x1)
Dropout(p=0.3)
Linear(1280 -> 1)
Sigmoid activation
```

| Parameter | Value |
|-----------|-------|
| Trainable parameters | ~1,281 |
| Total parameters | ~2,259,265 |
| Optimizer | Adam |
| Loss function | Custom Focal Loss + Class Weight + Garbage Penalty |
| Batch size | 64 |
| Epochs | 10 |
| Learning rate | 0.001 |

---

## 2. Intended Use

### Approved Users

Any user with access to the GitHub repository.

### Decision Context

The model outputs a sigmoid probability value. Classification threshold is set at 0.5:

- sigmoid output > 0.5: model predicts non_garbage (recyclable)
- sigmoid output <= 0.5: model predicts garbage (non-recyclable)

This output is an assistive suggestion only and should not be used as a final decision without human review. Actual classification should defer to human verification or local recycling regulations.

### Prohibited Uses

| Use Case | Reason |
|----------|--------|
| Commercial deployment | Not tested in real-world environments; no safety evaluation completed |
| High-risk decisions | Medical diagnosis, legal judgment, law enforcement, personnel evaluation |
| Direct deployment in Taiwan | TrashNet is a US dataset; local data fine-tuning required before Taiwan deployment |
| Fine-grained recycling judgment | Model performs binary classification only; cannot distinguish PET, HDPE, PP, and other plastic subtypes |
| Adversarial environments | No adversarial training applied; model is vulnerable to FGSM, PGD, and similar attacks |
| Fully automated decisions without human review | Human-in-the-loop mechanism is required |
| Training data reconstruction | Reverse-engineering training data from model weights violates original authors' intellectual property |

---

## 3. Training Data

For full dataset details, see `docs/DATA_CARD.md`.

| Field | Details |
|-------|---------|
| Dataset | TrashNet (dataset-resized) + Kaggle Garbage Classification trash supplement |
| Download date | June 2026 |
| Input feature | Single image tensor, shape (224, 224, 3), pixel values in [0.0, 1.0] |
| Target label | Binary: 0 = garbage (non-recyclable), 1 = non_garbage (recyclable) |
| Train set size | 70% of 3,224 images = approximately 2,257 images |
| Validation set size | 15% of 3,224 images = approximately 484 images |
| Data augmentation (train only) | RandomRotation(20 degrees), RandomHorizontalFlip(), RandomResizedCrop(224, scale=(0.8, 1.0)) |

---

## 4. Performance

### 4.1 Overall Metrics (Test Set, n = 483)

| Metric | MobileNetV2 | Custom CNN |
|--------|-------------|------------|
| Accuracy | 93.37% | 90.48% |
| Garbage Precision | 82.35% | 81.36% |
| Garbage Recall | 93.33% | 80.00% |
| Garbage F1 | 87.50% | 80.67% |
| Non-garbage Precision | 97.69% | 93.42% |
| Non-garbage Recall | 93.39% | 93.94% |
| Non-garbage F1 | 95.49% | 93.68% |
| Macro F1 | 91.50% | 87.18% |

Garbage Recall is the primary evaluation metric. Missing actual garbage (false negatives) carries a higher real-world cost than false positives.

### 4.2 Confusion Matrix

**MobileNetV2:**

|  | Predicted garbage | Predicted non_garbage |
|--|------------------|-----------------------|
| Actual garbage | 112 | 8 |
| Actual non_garbage | 24 | 339 |

**Custom CNN:**

|  | Predicted garbage | Predicted non_garbage |
|--|------------------|-----------------------|
| Actual garbage | 96 | 24 |
| Actual non_garbage | 22 | 341 |

### 4.3 Training Observations

- MobileNetV2: training and validation accuracy converge stably; no significant overfitting observed
- Custom CNN: training curve shows minor oscillation; overall convergence achieved
- Both models reach a stable range within 5 to 6 epochs

---

## 5. Fairness and Explainability

### 5.1 Statistical Parity Difference (Adapted)

Standard SPD is defined for tabular classification as the difference in positive prediction rates between privileged and unprivileged groups. For this image classification task, SPD is adapted as follows:

```
SPD (adapted) = P(correct prediction | trash class) - P(correct prediction | other classes)
```

| Metric | Value |
|--------|-------|
| SPD (adapted) | -0.0006 |
| Acceptable threshold | |SPD| < 0.1 (ideal); |SPD| > 0.2 (severely unfair) |
| Status | PASS — model treats garbage and non_garbage classes with near-identical recall |

### 5.2 Disparate Impact (Adapted)

Standard DI measures the ratio of positive prediction rates between unprivileged and privileged groups. Adapted for this task:

```
DI (adapted) = Recall(garbage / trash class) / Recall(non_garbage class)
```

| Metric | Value |
|--------|-------|
| DI (adapted) | 0.9994 |
| Acceptable threshold | DI >= 0.8 (four-fifths rule) |
| Status | PASS — well above the four-fifths threshold |

### 5.3 Slice Metrics

Per-slice F1 across dimensions defined in Data Card Section 4.2 is not implemented in v1.0. Required for v2.0.

### 5.4 Explainability

Standard tabular explainability tools (SHAP, LIME) are not directly applicable to image classification. The following image-adapted tools are planned for v2.0:

- Grad-CAM: visualizes which regions of the image the model attends to when making a prediction
- Integrated Gradients: measures per-pixel contribution to the prediction
- SmoothGrad: noise-reduced gradient visualization

---

## 6. Security Testing

### 6.1 Poisoning Checks

Four poisoning risks are documented in Data Card Section 8.3. The v1.0 status for each:

| Risk | v1.0 Status |
|------|------------|
| Label Flipping | No poisoning checks performed on training data |
| Backdoor Trigger | Not checked |
| Clean-Label Attack | Not checked |
| Distribution Shift Injection | Not checked |

Overall risk assessment: low to medium, as the primary data source is a public GitHub repository with no self-collected augmentation beyond the Kaggle supplement.

v2.0 plan: implement Activation Clustering detection before any local data augmentation.

### 6.2 Adversarial Vulnerability

Based on Machado et al. (2021), CNNs without adversarial training are highly vulnerable to evasion attacks. The attack types relevant to this system are described below, with expected impact based on the literature.

**Attack types and expected impact on an undefended MobileNetV2:**

| Attack | Type | Expected Impact |
|--------|------|----------------|
| FGSM (Goodfellow et al., 2015) | White-box, one-step, gradient-based | Significant accuracy drop even at small perturbation budgets (e.g., epsilon = 8/255) |
| BIM (Kurakin et al., 2017) | White-box, iterative, gradient-based | Greater accuracy drop than FGSM at equivalent epsilon; smaller perturbations |
| PGD (Madry et al., 2018) | White-box, iterative, gradient-based | Near-complete model failure at moderate epsilon; considered the strongest first-order attack |
| CW Attack (Carlini & Wagner, 2017) | White-box, iterative, gradient-based | State-of-the-art attack; finds minimal perturbation to cause misclassification with high confidence |
| Black-box transfer attack | Black-box, surrogate model | Adversarial examples crafted on a surrogate model (e.g., ResNet-50) transfer partially to MobileNetV2 due to adversarial transferability |

No adversarial experiments were conducted on this model in v1.0. The above reflects expected behavior based on Machado et al. (2021) survey findings for undefended CNNs. Actual robustness evaluation is planned for v2.0.

**Planned defense (v2.0)**: PGD Adversarial Training (PGD-AT), identified by Machado et al. (2021) as the most effective first-order defense at the time of publication.

### 6.3 API Abuse Assumptions

If deployed as a REST API, the following abuse scenarios are anticipated:

| Attack | Description | Planned Defense (v2.0) |
|--------|-------------|------------------------|
| Model Extraction | Attacker queries API to collect input-output pairs and trains a surrogate model | Limit sigmoid output precision; return hard labels only |
| Rate-based DoS | High-frequency queries exhaust compute resources | API gateway rate limiting |
| Adversarial Query Injection | Repeated adversarial image uploads to probe model weaknesses | Anomaly query detection (e.g., PRADA) |
| Membership Inference | Attacker infers whether a specific image was in the training set | Differential Privacy (v2.0 planned) |

---

## 7. Limitations and Residual Risk

### 7.1 Known Failure Modes

| Failure Mode | Description | Mitigation (v2.0) |
|-------------|-------------|-------------------|
| Out-of-distribution input | Non-white background or complex scenes cause large accuracy drop | OOD detection using Mahalanobis distance or ODIN |
| Minority class bias | Garbage class Recall may degrade with distribution shift | Class weighting and Focal Loss already applied in v1.0; monitor via drift detection |
| Adversarial examples | No adversarial training applied; model is vulnerable to FGSM and PGD | PGD Adversarial Training (PGD-AT) |
| Geographic bias | Taiwan-specific items absent from training data | Collect Taiwan local data and fine-tune |
| Brand bias | Non-North American brand packaging may be misclassified | Expand training data with diverse brand imagery |
| Low-light or blurred input | Not tested on low-quality images | Add GaussianBlur and ColorJitter augmentation in v2.0 |

### 7.2 Unacceptable Use Cases

- Fully automated decisions in real commercial recycling operations
- Any decision affecting health or safety
- Decisions affecting individual or group rights
- High-risk environments (medical waste, radioactive material classification)
- Legal contexts (recycling enforcement, penalty evidence)

### 7.3 Required Mitigations Before Any Deployment

- Implement all missing metrics from Section 4 (Precision, Recall, F1 per class)
- Implement per-slice F1 across dimensions defined in Data Card Section 4.2 (SPD and DI already computed in v1.0)
- Conduct adversarial testing with at least FGSM and PGD
- Add OOD detection module
- Add adversarial training
- Add Grad-CAM or equivalent XAI mechanism
- Supplement with Taiwan local data
- Implement human-in-the-loop review mechanism

---

## 8. Deployment and Monitoring

### 8.1 Decision Threshold

| Setting | Value |
|---------|-------|
| v1.0 threshold | 0.5 (sigmoid output > 0.5 predicts non_garbage) |
| Basis for threshold | Default value; no ROC or PR curve analysis performed |
| v2.0 improvement | Select threshold via PR curve analysis to improve Garbage Recall (e.g., raise threshold for non_garbage prediction to reduce false negatives on garbage) |

### 8.2 Human Review Triggers

Human review must be triggered under the following conditions:

- Sigmoid output falls in the uncertain range (e.g., 0.35 to 0.65)
- Model predicts garbage (minority class; misclassification cost is high)
- OOD detection score exceeds alert threshold
- Batch processing reveals abnormal distribution shift in predictions

### 8.3 Drift Detection

| Drift Type | Monitoring Method | Alert Threshold |
|------------|------------------|-----------------|
| Input Drift | Weekly KS-test on image statistics (brightness, contrast, color distribution) vs. training distribution | p-value < 0.01 |
| Concept Drift | Monthly agreement rate between human labels and model predictions | Agreement rate drop > 5% |
| Performance Drift | Monthly evaluation on a small human-annotated holdout set | Accuracy drop > 3% |

### 8.4 Rollback

| Field | Details |
|-------|---------|
| Version management | All model weights saved with timestamp-based filenames; retain at least 3 stable versions |
| Rollback trigger | Any drift alert unresolved for 2 consecutive weeks, or any major security incident |
| Rollback process | Suspend live service; load previous stable version; notify users; begin root cause analysis |

### 8.5 Retraining Triggers

Retraining must be initiated when any of the following conditions occur:

- Performance drift exceeds 3% for 2 consecutive weeks
- 500 or more new human-labeled images have been accumulated
- A major distribution shift is detected (e.g., new packaging materials become widespread)
- A new adversarial attack pattern is identified
- Data Card undergoes a major update (e.g., class definition changes)
- Fixed schedule: mandatory review every 6 months

### 8.6 Approval Gate

| Role | Party |
|------|-------|
| Model Card Author | Group 5 |
| Technical Reviewer | Course Instructor |
| Final Approver | Course Instructor |
| Current Status | v1.0 for academic final report only; not approved for actual deployment |
| Next Review | Before v2.0 development or after any major incident |
