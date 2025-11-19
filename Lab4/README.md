# Lab 4 - Computer Vision with CIFAR-10

## Overview

This lab focuses on image classification using the CIFAR-10 dataset. The goal is to build, train, and evaluate convolutional neural networks (CNNs) for recognizing objects in 32x32 color images across 10 different classes.

## Files Description

- **Lab4.ipynb** - Main Jupyter notebook with CNN implementations, training, and evaluation
- **cifar10_best_P.pth** - Pre-trained model checkpoint with best performance

## Running the Lab

### Interactive Notebook

```bash
jupyter notebook Lab4.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Model architecture definition
- Training loops with visualization
- Evaluation and testing
- Results analysis and error cases

### Using the Pre-trained Model

The included `cifar10_best_P.pth` file contains a trained model that can be loaded:

```python
import torch

# Load the pre-trained model
model = YourModelClass()  # Replace with actual model architecture
model.load_state_dict(torch.load('cifar10_best_P.pth'))
model.eval()

# Use for inference
with torch.no_grad():
    predictions = model(test_images)
```

## CIFAR-10 Dataset

### Dataset Overview

- **Size:** 60,000 images (50,000 training + 10,000 test)
- **Image dimensions:** 32x32 pixels, RGB (3 channels)
- **Classes:** 10 categories
- **Distribution:** 6,000 images per class

### Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

### Dataset Characteristics

- **Low resolution:** 32x32 pixels makes recognition challenging
- **Variety within classes:** Different poses, lighting, backgrounds
- **Similar classes:** Some classes are visually similar (cat/dog, deer/horse)
- **Real-world images:** Natural variation in object appearance

## Model Architecture

The lab likely explores various CNN architectures:

### Basic CNN
- Multiple convolutional layers with ReLU activation
- Max pooling for spatial dimension reduction
- Fully connected layers for classification
- Batch normalization for training stability
- Dropout for regularization

### Advanced Architectures
- **ResNet-inspired:** Residual connections for deeper networks
- **VGG-inspired:** Small 3x3 convolutions stacked deeply
- **Modern designs:** Depthwise separable convolutions, global average pooling

### Example Architecture
```
Conv2D (32 filters, 3x3) → ReLU → BatchNorm
Conv2D (32 filters, 3x3) → ReLU → BatchNorm → MaxPool
Conv2D (64 filters, 3x3) → ReLU → BatchNorm
Conv2D (64 filters, 3x3) → ReLU → BatchNorm → MaxPool
Conv2D (128 filters, 3x3) → ReLU → BatchNorm → MaxPool
Flatten
Dense (256) → ReLU → Dropout(0.5)
Dense (10) → Softmax
```

## Training Process

### Data Preprocessing

```python
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                       std=[0.2470, 0.2435, 0.2616])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                       std=[0.2470, 0.2435, 0.2616])
])
```

### Training Configuration

Typical hyperparameters:
- **Optimizer:** SGD with momentum or Adam
- **Learning rate:** 0.01-0.001 with decay/scheduling
- **Batch size:** 64-128
- **Epochs:** 100-200
- **Weight decay:** 1e-4 to 5e-4
- **Loss function:** Cross-Entropy Loss

### Data Augmentation

Key techniques to improve generalization:
- **Random horizontal flip** - Doubles effective dataset size
- **Random cropping** - Helps with translation invariance
- **Color jittering** - Robustness to lighting variations
- **Cutout/Mixup** - Advanced regularization (optional)

## Experimental Results

### Performance Metrics

- **Top-1 Accuracy** - Primary metric for 10-class classification
- **Per-class Accuracy** - Identify which classes are challenging
- **Confusion Matrix** - Understand misclassification patterns
- **Training/Validation Curves** - Monitor overfitting

### Expected Performance

| Model Type | Test Accuracy | Parameters | Training Time |
|------------|---------------|------------|---------------|
| Basic CNN | 70-75% | ~100K | ~30 min |
| Deeper CNN | 80-85% | ~500K | ~1 hour |
| ResNet-20 | 85-88% | ~270K | ~2 hours |
| ResNet-56 | 88-91% | ~850K | ~4 hours |
| State-of-the-art | 95%+ | Variable | Extended |

### Key Findings

Common observations:

**Challenging Classes:**
- Cat vs Dog confusion is common
- Deer vs Horse similarity causes errors
- Airplane vs Ship occasionally confused

**Training Insights:**
- Data augmentation provides 5-10% accuracy boost
- Batch normalization significantly stabilizes training
- Learning rate scheduling improves final accuracy
- Deeper networks (with skip connections) perform better

**Overfitting Indicators:**
- Training accuracy >> Test accuracy
- Mitigated by dropout, weight decay, and augmentation

## Monitoring Training

### TensorBoard Visualization

```bash
tensorboard --logdir=../logs/
```

View at `http://localhost:6006` to track:
- Training and validation loss
- Training and validation accuracy
- Learning rate schedule
- Per-class accuracy
- Gradient statistics

### Jupyter Notebook Plots

The notebook includes visualizations for:
- Loss curves over epochs
- Accuracy progression
- Sample predictions with confidence scores
- Misclassified examples analysis
- Filter visualizations (first layer)

## Model Evaluation

### Testing the Model

```python
model.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

test_accuracy = 100. * correct / len(test_loader.dataset)
print(f'Test Accuracy: {test_accuracy:.2f}%')
```

### Error Analysis

Analyze misclassifications:
- Identify consistently confused class pairs
- Visualize images the model gets wrong
- Examine prediction confidence on errors
- Consider additional data augmentation strategies

## Dependencies

Key packages:
- `torch` (2.7.1) - Deep learning framework
- `torchvision` (0.22.1) - Vision datasets and models
- `matplotlib` - Visualization
- `numpy` - Numerical operations
- `tensorboard` - Training monitoring
- `scikit-learn` - Metrics and evaluation

## GPU Acceleration

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
```

## Common Issues and Solutions

### Overfitting

**Symptoms:**
- High training accuracy (>95%) but low test accuracy (<80%)
- Widening gap between train and validation curves

**Solutions:**
```python
# Increase dropout
model.add_dropout(p=0.5)

# Add weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# More data augmentation
transforms.Compose([...])  # Add more transformations

# Early stopping
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter > patience:
        break
```

### Underfitting

**Symptoms:**
- Both training and test accuracy are low (<70%)
- Loss plateaus early

**Solutions:**
- Increase model capacity (more layers/filters)
- Train for more epochs
- Reduce regularization (lower dropout/weight decay)
- Increase learning rate
- Check data preprocessing is correct

### Slow Training

**Solutions:**
- Verify GPU is being used
- Increase batch size (if memory allows)
- Use DataLoader with `num_workers=4`
- Reduce image operations in augmentation
- Consider mixed precision training

### Poor Accuracy (<70%)

**Checklist:**
- [ ] Data normalization is correct
- [ ] Model has sufficient capacity
- [ ] Learning rate is appropriate
- [ ] Data augmentation is enabled
- [ ] Training for enough epochs
- [ ] No bugs in loss calculation

## Expected Outcomes

After completing this lab, you should:

- Understand CNN architectures for image classification
- Know how to preprocess and augment image data
- Be able to train and evaluate vision models
- Recognize overfitting and apply regularization
- Interpret confusion matrices and error patterns
- Optimize hyperparameters for better performance
- Use pre-trained models for transfer learning

## Advanced Techniques

For further improvement:

### Transfer Learning
```python
from torchvision import models

# Load pre-trained ResNet
model = models.resnet18(pretrained=True)

# Replace final layer for CIFAR-10
model.fc = nn.Linear(model.fc.in_features, 10)

# Fine-tune or freeze early layers
for param in model.parameters():
    param.requires_grad = False  # Freeze
model.fc.requires_grad = True     # Only train classifier
```

### Learning Rate Scheduling
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=200, eta_min=1e-6
)

# Or use ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Resources

### Papers
- [Going deeper with convolutions (GoogLeNet)](https://arxiv.org/abs/1409.4842)
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- [Batch Normalization](https://arxiv.org/abs/1502.03167)

### Datasets
- [CIFAR-10/100 Official Page](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch CIFAR Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

### Leaderboards
- [Papers With Code - CIFAR-10](https://paperswithcode.com/sota/image-classification-on-cifar-10)

## Next Steps

After completing this lab:
- Try other datasets (CIFAR-100, ImageNet)
- Implement more advanced architectures (EfficientNet, Vision Transformer)
- Experiment with ensemble methods
- Explore model interpretability techniques
- Deploy your model as a web service
