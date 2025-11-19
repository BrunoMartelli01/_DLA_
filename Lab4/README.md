# Lab 4 - Adversarial Learning and OOD Detection

## Overview

This lab develops methodologies for **detecting Out-of-Distribution (OOD) samples** and enhancing model **robustness to adversarial attacks**. The exercises progress from baseline OOD detection through adversarial training to advanced ODIN (Out-of-Distribution detector using Input preprocessing and temperature scaling) implementation.

## Files Description

- **Lab4.ipynb** - Complete notebook with all three exercises and implementations
- **cifar10_best_P.pth** - Pre-trained CNN model checkpoint for baseline experiments

## Lab Structure

The lab is divided into three main exercises:

1. **Exercise 1**: OOD Detection Pipeline and Performance Evaluation
2. **Exercise 2**: Enhancing Robustness with Adversarial Training (FGSM)
3. **Exercise 3**: Advanced OOD Detection with ODIN

---

## Exercise 1: OOD Detection and Performance Evaluation

### Objectives

- Build a baseline OOD detection pipeline
- Implement performance metrics (AUROC, AUPR, FPR@95TPR)
- Compare different detection approaches

### Exercise 1.1: Build OOD Detection Pipeline

**Dataset Setup:**
- **In-Distribution (ID)**: CIFAR-10 training and test sets
- **Out-of-Distribution (OOD)**: FakeData dataset (randomly generated images)
- **Validation Split**: 90% training, 10% validation from CIFAR-10 train set

**Models Implemented:**

#### 1. CNN Classifier (Baseline)
```python
class CNN(nn.Module):
    # 5 convolutional layers + 3 fully connected layers
    # Outputs logits for 10 classes
```

**Detection Method**: Maximum Softmax Probability (MSP)
- Score: `logit.max(dim=1)[0]`
- Higher scores indicate ID samples

#### 2. Autoencoder
```python
class Autoencoder(nn.Module):
    # Encoder: 3 conv layers (3→12→24→48 channels)
    # Decoder: 3 transposed conv layers (48→24→12→3)
```

**Detection Method**: Reconstruction Error
- Score: `-MSE(x, x_reconstructed)`
- Lower reconstruction error indicates ID samples

**Training Configuration:**
```python
# CNN Classifier
optimizer = SGD(lr=0.01, weight_decay=5e-4)
epochs = 100
loss = CrossEntropyLoss()

# Autoencoder
optimizer = Adam(lr=0.001)
epochs = 100
loss = MSELoss()
```

### Exercise 1.2: Performance Metrics

**Implemented Metrics:**

1. **AUROC (Area Under ROC Curve)**
   - Measures overall separation capability
   - Range: [0, 1], higher is better
   - 1.0 = perfect separation, 0.5 = random

2. **AUPR (Area Under Precision-Recall Curve)**
   - Particularly useful for imbalanced datasets
   - Range: [0, 1], higher is better

3. **FPR@95TPR (False Positive Rate at 95% True Positive Rate)**
   - Practical metric: false alarms when accepting 95% of ID samples
   - Range: [0, 1], lower is better

4. **Detection Error@95TPR**
   - Average of FPR and FNR at 95% TPR threshold
   - Formula: `0.5 * (1 - 0.95) + 0.5 * FPR@95`

**Evaluation Function:**
```python
def eval_OOD(model, idloader, oodloader, device, score_fn):
    # Returns: AUROC, AUPR, FPR@95TPR, Detection Error
    # Plus ROC and PR curve data for plotting
```

**Expected Results (Baseline CNN):**
- AUROC: ~0.90-0.95
- FPR@95TPR: ~0.10-0.20
- Autoencoder generally shows different trade-offs

**Key Findings:**
- CNN provides stable, consistent detection
- Autoencoder achieves higher peak discrimination but with sharper transitions
- Reconstruction-based methods concentrate errors in narrow regions

---

## Exercise 2: Adversarial Training with FGSM

### Objectives

- Implement Fast Gradient Sign Method (FGSM)
- Generate adversarial examples
- Train robust models using adversarial augmentation

### Exercise 2.1: FGSM Implementation

**FGSM Attack Formula:**
```
η(x) = ε · sign(∇ₓ L(θ, x, y))
x_adv = x + η(x)
```

**Implementation:**
```python
def fgsm_attack_batch(model, x, y, eps=1/255):
    # 1. Compute gradient of loss w.r.t. input
    # 2. Take sign of gradient
    # 3. Add scaled perturbation to input
    # 4. Clip to valid range [-1, 1]
```

**Epsilon (ε) Sensitivity:**
- `ε = 1/255`: Subtle, nearly imperceptible perturbations
- `ε = 8/255`: Stronger attacks, some visual artifacts
- `ε = 32/255`: Significant perturbations, visible changes

**Targeted vs Untargeted Attacks:**
- **Untargeted**: Make model misclassify (any wrong class)
- **Targeted**: Force model to predict specific target class

### Exercise 2.2: Adversarial Training

**Training Strategy:**
```python
def train_with_fgsm(model, trainloader, valloader, 
                     epochs=30, eps=32/255, alpha=0.5):
    # For each batch:
    # 1. Generate adversarial examples
    # 2. Combined loss: α·L(x_clean) + (1-α)·L(x_adv)
    # 3. Backpropagate and update
```

**Key Parameters:**
- `alpha`: Balance between clean and adversarial loss (typically 0.5)
- `eps`: Attack strength during training
- Learning rate schedule: Cosine annealing recommended

**Evaluation Metrics:**
```python
def evaluate_model(model, loader, device, eps):
    # Returns: clean_accuracy, adversarial_accuracy, avg_loss
```

**Expected Improvements:**
- Adversarial accuracy increases from ~0% to 40-60%
- Clean accuracy may decrease slightly (2-5%)
- OOD detection performance often improves
- Model learns more robust features

**TensorBoard Logging:**
Monitor training progress:
```bash
tensorboard --logdir=fgsm/
```

Tracked metrics:
- Train/validation loss
- Clean/adversarial accuracy
- AUROC for OOD detection
- Learning rate schedule

---

## Exercise 3: ODIN for OOD Detection

### Objective

Implement and evaluate an advanced OOD detection method using ODIN (Out-of-Distribution detector for Neural Networks). Compare its performance versus baseline methods on CIFAR-10 (ID) vs. FakeData (OOD).

### 3.1: Building the ODIN Score Function

**Goal:**
- Enhance separation between ID and OOD samples through temperature scaling and gradient-based input preprocessing.

**Method:**
- **Temperature scaling:** Divide the output logits by a temperature parameter `T > 1` before applying softmax. This increases the sensitivity of the model to subtle differences, resulting in a stronger separation of confidence scores.
- **Input preprocessing:** Perturb the test input with a tiny adversarial step directed to increase softmax score, computed by `x_perturbed = x - eps · sign(∇ₓ loss)`, with `loss = CrossEntropy(logits_scaled, predicted_labels)`.

**Implementation Highlights:**
- Tune `T` in the range 1–1000.
- Tune `eps` in the range 0–0.004 (adjust according to normalization).
- Apply scoring to both ID and OOD datasets.

```python
def compute_scores_ODIN(model, dataloader, device, T, eps):
    # Step 1: forward, get predicted class
    # Step 2: backward on temp-scaled logits for gradient
    # Step 3: negative eps sign(gradient) perturbation
    # Step 4: forward+softmax on perturbed, temp-scaled input
    # Return: max softmax score (ODIN score) per sample
```

### 3.2: Hyperparameter Tuning and Metrics

**Goal:**
- Systematically search for the best values of `T` and `eps` to optimize detection, minimizing FPR@95TPR.

**Method:**
- Perform grid search [T in {1, 2, 5, 10, ..., 1000}, eps in [0, 0.004]]
- For each pair, compute ODIN scores on ID and OOD, get evaluation metrics:
    - AUROC: Area under ROC curve
    - AUPR: Area under Precision-Recall curve
    - FPR@95TPR: False Positives at 95% True Positive Rate

**Implementation:**
- Use `sklearn.metrics` routines as in Exercises 1/2.
- For each pair, track and store best `T` and `eps`.

```python
def eval_OOD(model, idloader, oodloader, device, score_fn):
    # Returns: AUROC, AUPR, FPR@95TPR, Detection Error
    # Plus ROC and PR curve data for plotting
```

### 3.3: Results and Practical Insights

**Expected Outcomes:**
- ODIN produces higher AUROC and much lower FPR@95TPR than vanilla MSP.
- Performance improvement is significant after tuning: FPR@95TPR commonly drops from 10–20% (baseline) to 2–6% (ODIN-tuned).
- AUROC typically improves by 3–5 points.

**Example Result Table:**

| Method           | AUROC   | FPR@95TPR | AUPR  |
|------------------|---------|-----------|-------|
| Baseline (MSP)   | 0.92    | 0.14      | 0.95  |
| ODIN (Tuned)     | 0.97    | 0.04      | 0.98  |

**Key Best Practices:**
- Always grid search `T` and `eps` on validation data, never test
- Ensure normalization is consistent for correct `eps` scaling
- Only consider correctly classified ID samples for scoring

**Troubleshooting:**
- No improvement: check if gradients are being computed on input!
- Instabilities: Use reasonable `T` values (≤1000), clip NaNs
- Overlap/poor separation: Retrain base model, increase data, check bug in perturbation direction

**Visualization:**
- Plot ROC and PR curves for both baseline and ODIN; compare visually.

### 3.4: Summary

- ODIN uses gradient-based preprocessing + temperature scaling to expand separation between ID/OOD samples
- When tuned properly, it provides a robust, practical OOD detector that outperforms MSP and Autoencoder baselines.
- Applying ODIN (particularly after adversarial training) sets a solid foundation for reliable OOD-aware neural network models.
