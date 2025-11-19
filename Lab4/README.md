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

## Exercise 3: ODIN for Enhanced OOD Detection ⭐

### Overview

ODIN (Out-of-Distribution detector using Input preprocessing and temperature scaling) significantly improves OOD detection by combining two techniques:

1. **Temperature Scaling** - Softens confidence distributions
2. **Input Preprocessing** - Uses gradients to amplify ID/OOD separation

### Theoretical Foundation

**Key Insight**: A small gradient-based perturbation designed to *increase* model confidence affects ID samples much more than OOD samples.

**ODIN Score Formula:**
```
# Step 1: Temperature scaling
logits_scaled = logits / T

# Step 2: Compute gradient
loss = CrossEntropy(logits_scaled, y_pred)
η = -ε · sign(∇ₓ loss)

# Step 3: Perturb input
x_perturbed = x + η

# Step 4: Final score
logits_final = model(x_perturbed) / T
score = max(softmax(logits_final))
```

### Implementation

#### ODIN Score Function

```python
def compute_scores_ODIN(model, dataloader, device, T, eps, 
                        filter_correct_only=False):
    """
    Args:
        T: Temperature parameter (typically 100-1000)
        eps: Input perturbation magnitude (typically 0.001-0.004)
        filter_correct_only: For ID data, only score correctly classified samples
    
    Returns:
        scores: ODIN confidence scores
    """
```

**Critical Implementation Details:**

1. **Temperature Scaling (`T`)**:
   - Divides logits before softmax
   - `T > 1` softens probability distribution
   - Increases gradient magnitudes for all classes
   - Typical range: 100-1000

2. **Input Preprocessing (`eps`)**:
   - Gradient descent on input (not ascent!)
   - Direction: `-ε · sign(∇ₓ loss)`
   - Pushes sample toward predicted class
   - Effects more pronounced on ID samples
   - Typical range: 0.0-0.004 (in normalized space)

3. **Normalization Consideration**:
   ```python
   # If images normalized to [-1, 1] with std=0.5:
   gradient_sign = grad_correct.data.sign()
   perturbed_x = x_correct - eps * gradient_sign / 0.5
   ```

### Hyperparameter Tuning

**Grid Search Strategy:**

```python
# Search space
temperatures = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
epsilons = np.linspace(0, 0.004, 21)

# Objective: Minimize FPR@95TPR
for T in temperatures:
    for eps in epsilons:
        score = eval_OOD(model, testloader, oodloader, 
                        device, score_fn=compute_scores_ODIN)
        # Track best_T, best_eps based on FPR@95TPR
```

**Typical Optimal Values:**
- **Temperature (`T`)**: Often in range [100, 1000]
  - Higher T → softer distributions → larger gradients
  - Too high → numerical instability
  
- **Epsilon (`eps`)**: Often in range [0.001, 0.004]
  - Higher eps → stronger perturbation
  - Too high → visual artifacts, unstable scores

### Performance Comparison

**Visualization:**
```python
def plot_curve_comparison(metrics_dict1, metrics_dict2, 
                          label1="ODIN", label2="Baseline"):
    # Plots side-by-side ROC and PR curves
    # Includes AUROC/AUPR in legend
```

**Expected Improvements (CIFAR-10 vs FakeData):**

| Metric | Baseline MSP | ODIN (Tuned) | Improvement |
|--------|--------------|--------------|-------------|
| AUROC | 0.90-0.95 | 0.95-0.98 | +3-5% |
| FPR@95TPR | 0.10-0.20 | 0.02-0.08 | 50-60% reduction |
| AUPR | 0.92-0.96 | 0.96-0.99 | +2-4% |

### Key Insights

**Why ODIN Works:**

1. **Temperature Scaling Effect**:
   - Amplifies gradient magnitudes uniformly
   - Makes subsequent perturbation more effective
   - Doesn't change prediction, only confidence calibration

2. **Input Preprocessing Effect**:
   - ID samples: Model is confident, small push increases confidence significantly
   - OOD samples: Model is uncertain, same push has minimal effect
   - Creates larger separation in confidence distributions

3. **Combined Synergy**:
   - Temperature enables stronger gradient signal
   - Preprocessing amplifies existing ID/OOD differences
   - Both techniques are complementary

**Practical Considerations:**

- **Computational Cost**: ~2-3x slower than baseline (requires backward pass)
- **Hyperparameter Sensitivity**: Requires tuning per dataset/model
- **Normalization Matters**: Adjust eps based on input scaling
- **Correct-Only Filtering**: For ID data, only score correctly classified samples

### Common Issues and Solutions

**Issue**: Poor performance with default hyperparameters
**Solution**: Always perform grid search; optimal values vary by dataset

**Issue**: Numerical instability with high temperature
**Solution**: Limit T to reasonable range (≤1000), check for NaN values

**Issue**: No improvement over baseline
**Solution**: 
- Verify gradient computation is enabled (`x.requires_grad=True`)
- Check perturbation direction (should be `-eps * sign`)
- Ensure temperature scaling is applied correctly

**Issue**: Inconsistent results across runs
**Solution**: 
- Set random seeds for reproducibility
- Use enough samples in evaluation
- Verify correct-only filtering for ID data

---

## Running the Lab

### Complete Notebook

```bash
jupyter notebook Lab4.ipynb
```

The notebook guides you through:
1. Dataset preparation and splitting
2. Model training (CNN and Autoencoder)
3. Baseline OOD detection
4. FGSM attack implementation
5. Adversarial training
6. ODIN implementation and tuning
7. Comprehensive performance comparison

### Key Dependencies

```python
import torch
import torchvision
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
```

---

## Results Summary

### Baseline Methods

**CNN Maximum Softmax:**
- Fast, simple baseline
- AUROC: 0.90-0.95
- Good starting point

**Autoencoder Reconstruction:**
- Different failure modes than CNN
- Higher peak performance
- More sensitive to threshold choice

### Adversarial Training

**Benefits:**
- Improved robustness to attacks
- Often better OOD detection
- More meaningful learned features

**Trade-offs:**
- Slightly lower clean accuracy
- Longer training time
- Requires careful epsilon tuning

### ODIN (Exercise 3)

**Optimal Configuration (Typical):**
- Temperature: 100-500
- Epsilon: 0.001-0.003
- 50-60% reduction in FPR@95TPR

**Best Use Cases:**
- When computational cost is acceptable
- High-stakes applications requiring reliability
- After thorough hyperparameter tuning

---

## Best Practices

1. **Always perform train/val/test split properly**
   - Never tune on test set
   - Use validation set for hyperparameter search

2. **Visualize distributions**
   - Plot histograms of scores
   - Check for overlap between ID/OOD
   - Use both sorted curves and histograms

3. **Report multiple metrics**
   - AUROC for overall performance
   - FPR@95TPR for practical deployment
   - AUPR for imbalanced scenarios

4. **Consider computational budget**
   - Baseline methods: Fast, reasonable performance
   - ODIN: Slower, best performance after tuning
   - Adversarial training: Moderate cost, robust models

5. **Document hyperparameters**
   - Record all settings used
   - Note dataset-specific optimal values
   - Share tuning insights with team

---

## References

### Papers

- [ODIN: Enhancing the Reliability of Out-of-distribution Image Detection in Neural Networks](https://arxiv.org/abs/1706.02690)
- [Explaining and Harnessing Adversarial Examples (FGSM)](https://arxiv.org/abs/1412.6572)
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)

### Additional Resources

- [OpenOOD Benchmark](https://github.com/Jingkang50/OpenOOD)
- [Adversarial Robustness Toolbox](https://adversarial-robustness-toolbox.readthedocs.io/)
- [PyTorch Adversarial Training Tutorial](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

---

## Next Steps

After completing this lab:

1. **Try different OOD datasets** (CIFAR-100 subsets, SVHN, Textures)
2. **Explore other detection methods** (Mahalanobis distance, Energy-based)
3. **Implement stronger attacks** (PGD, C&W)
4. **Experiment with ensemble methods**
5. **Deploy ODIN in a real application**
