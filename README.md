# Deep Learning Applications (DLA) - Lab Projects

This repository contains laboratory assignments for the Deep Learning Applications course. Each lab focuses on different aspects of deep learning and reinforcement learning.

## Repository Structure

```
_DLA_/
├── Lab2/              # Deep Q-Learning & PPO
├── Lab3/              # Advanced Deep Learning Topics
├── Lab4/              # Computer Vision with CIFAR-10
├── requirements.txt   # Python dependencies
└── logs/              # Training logs and results
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- CUDA 12.8 (for GPU acceleration)
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/BrunoMartelli01/_DLA_.git
cd _DLA_
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

**Note**: This project uses PyTorch with CUDA 12.8. If you have a different CUDA version, modify the torch installation accordingly.

## Running the Labs

### Lab 2 - Reinforcement Learning (DQL & PPO)

Location: `Lab2/`

This lab implements two reinforcement learning algorithms:
- **Deep Q-Learning (DQL)** - Value-based RL algorithm
- **Proximal Policy Optimization (PPO)** - Policy gradient method

**Run the notebook:**
```bash
cd Lab2
jupyter notebook Lab2.ipynb
```

**Run standalone scripts:**
```bash
python Lab2/DQL.py        # Train DQL agent
python Lab2/PPO.py        # Train PPO agent
python Lab2/Eval.py       # Evaluate trained models
python Lab2/Results.py    # Visualize results
```

See [Lab2/README.md](Lab2/README.md) for detailed information.

### Lab 3 - Advanced Deep Learning

Location: `Lab3/`

This lab explores advanced deep learning techniques including transformers, attention mechanisms, and transfer learning.

**Run the notebook:**
```bash
cd Lab3
jupyter notebook Lab3.ipynb
```

See [Lab3/README.md](Lab3/README.md) for detailed information.

### Lab 4 - Computer Vision with CIFAR-10

Location: `Lab4/`

This lab focuses on image classification using the CIFAR-10 dataset with modern CNN architectures.

**Run the notebook:**
```bash
cd Lab4
jupyter notebook Lab4.ipynb
```

**Pre-trained model:**
The directory includes `cifar10_best_P.pth`, a pre-trained model checkpoint that can be loaded for inference or fine-tuning.

See [Lab4/README.md](Lab4/README.md) for detailed information.

## Key Dependencies

- **PyTorch 2.7.1** (with CUDA 12.8) - Deep learning framework
- **Stable-Baselines3 2.6.0** - Reinforcement Learning algorithms
- **Gymnasium 1.1.1** - RL environments
- **TensorBoard 2.19.0** - Training visualization
- **Transformers 4.53.0** - Hugging Face models
- **OpenCV 4.11** - Computer vision utilities
- **torchvision 0.22.1** - Vision datasets and models

## Monitoring Training

Many labs use TensorBoard for monitoring training progress. To launch TensorBoard:

```bash
tensorboard --logdir=logs/
```

Then open your browser at `http://localhost:6006`

## Project Overview

### Lab 2 Focus
- Reinforcement Learning fundamentals
- Value-based methods (DQL)
- Policy gradient methods (PPO)
- Environment interaction and reward optimization

### Lab 3 Focus
- Transformer architectures
- Attention mechanisms
- Transfer learning with pre-trained models
- Advanced optimization techniques

### Lab 4 Focus
- Image classification on CIFAR-10
- Convolutional Neural Networks
- Model training and evaluation
- Performance optimization

## Troubleshooting

### CUDA Issues
If you encounter CUDA-related errors:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Should return `True` if CUDA is properly configured.

### Missing Dependencies
If imports fail, ensure all packages are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Jupyter Kernel Issues
If Jupyter doesn't recognize the environment:
```bash
python -m ipykernel install --user --name=dla-env
```

### Out of Memory (OOM) Errors
- Reduce batch size in the training scripts
- Close other applications using GPU memory
- Use gradient checkpointing for large models

## Best Practices

- **Save your work frequently** - Use version control and commit regularly
- **Monitor training** - Use TensorBoard to track metrics
- **Document experiments** - Keep notes on hyperparameters and results
- **Use GPU efficiently** - Close notebooks when not in use to free memory

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Hugging Face Hub](https://huggingface.co/)

## License

This repository is for educational purposes as part of the Deep Learning Applications course.

## Author

Bruno Martelli ([BrunoMartelli01](https://github.com/BrunoMartelli01))
