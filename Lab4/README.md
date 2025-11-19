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
