# Lab 3 - Advanced Deep Learning Exercises

## Overview

This lab is organized in three main exercises, each exploring advanced deep learning concepts and modern practical techniques. For each exercise you'll find: the objective, method/implementation, and expected results.

---

## Exercise 1: Transformers and Attention

**Objective:**
- Understand and implement self-attention and transformer architectures for sequence modeling and vision.

**Method:**
- Develop from scratch the Multi-Head Attention mechanism and integrate it into a Transformer block.
- Use positional encodings and test encoder-decoder setups on simple sequence-to-sequence tasks.
- Apply vision transformer (ViT) on image data.

**Implementation Notes:**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        # ... standard transformer code ...
```
- Leverage HuggingFace `transformers` for fine-tuning large models.

**Expected Results:**
- Attention mechanisms improve sequence modeling.
- Transformers perform better than simple RNNs/CNNs, especially on longer or more complex sequences.

---

## Exercise 2: Transfer Learning

**Objective:**
- Apply, fine-tune, and evaluate pre-trained models on custom tasks/datasets.

**Method:**
- Use a pre-trained model (BERT, ViT, ResNet, EfficientNet, etc.) from HuggingFace or TorchVision.
- Explore full fine-tuning vs. feature extraction strategies.
- Practice domain adaptation (i.e., transfer from ImageNet to CIFAR-10).

**Implementation Example:**
```python
from transformers import AutoModel
model = AutoModel.from_pretrained('bert-base-uncased')
# ... fine-tuning code ...
```
**Evaluation:**
- Compare training time, data requirements, and final performance between training from scratch and transfer learning.

**Expected Results:**
- Pre-trained models yield much better performance on small datasets and require less training time.
- Domain-related pre-trained models transfer knowledge more effectively.

---

## Exercise 3: Advanced Optimization & Regularization

**Objective:**
- Optimize deep neural networks for speed, memory, and generalization.

**Method:**
- Implement and compare optimizers (Adam, AdamW, SGD with momentum).
- Integrate learning rate scheduling (cosine annealing, warmup).
- Use mixed precision training and gradient accumulation for large models.
- Apply regularization methods: dropout, batch/layer normalization, data augmentation.

**Implementation Example:**
```python
from torch.cuda.amp import autocast, GradScaler
with autocast():
    # ... training loop ...
```
**Expected Improvements:**
- Faster training with mixed precision and schedulers
- Better generalization and more stable convergence with regularization

---

## Results & Best Practices

- Always monitor CUDA utilization (use `torch.cuda.is_available()` and print GPU details)
- For out-of-memory issues: reduce batch size, enable gradient checkpointing, or use smaller model variants
- Monitor metrics with TensorBoard for all experiments
- Visualize attention weights, confusion matrices, and performance curves to interpret your results

## Dependencies
- `transformers`, `torch`, `torchvision`, `datasets`, `matplotlib`, `scikit-learn`, `tensorboard`

## Next Steps
- Experiment with few-shot or self-supervised learning
- Try neural architecture search or model distillation
- Explore multi-modal architectures

---

For detailed code and working examples, open the notebook in this folder:

```bash
jupyter notebook Lab3.ipynb
```
