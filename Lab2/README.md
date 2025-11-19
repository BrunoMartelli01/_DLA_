# Lab 2 - Reinforcement Learning: DQL & PPO — Exercises Structure

## Overview

The Lab is divided in two main exercises, each focused on exploring a state-of-the-art reinforcement learning algorithm and directly comparing their behavior on different environments and evaluation metrics.

---

## Exercise 1: Deep Q-Learning (DQL)

**Objective:**
- Implement and evaluate the Deep Q-Learning algorithm on classic control problems with discrete actions.

**Method:**
- Build a value-based RL agent using Q-learning with experience replay and target networks.
- Use epsilon-greedy exploration with decay.
- Train and assess on environments such as CartPole-v1, LunarLander-v2, and MountainCar-v0.

**Implementation Example:**
```python
python DQL.py       # Train DQL agent
python Eval.py      # Evaluate trained DQL model
```

**Key Parameters:**
```python
learning_rate = 1e-3
buffer_size = 100000
batch_size = 64
gamma = 0.99
...
```

**Expected Results:**
- DQL learns efficient policies on simpler, discrete environments
- High sample efficiency but more sensitivity to hyperparameters (especially exploration decay)
- Training curves may show higher variance

---

## Exercise 2: Proximal Policy Optimization (PPO)

**Objective:**
- Implement and experiment with PPO, analyzing its advantages for stable policy gradient learning (including continuous action support).

**Method:**
- Policy-based RL, training actor and value networks in parallel
- Clipped surrogate objective with GAE for variance reduction
- Run experiments on same environments but extend also to BipedalWalker-v3 (continuous)

**Implementation Example:**
```python
python PPO.py       # Train PPO agent
python Eval.py      # Evaluate trained PPO model
```
**Key Parameters:**
```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
...
```

**Expected Results:**
- PPO exhibits smoother, more stable training, handling both discrete and continuous actions gracefully
- Final performance often higher than DQL, at the cost of more samples and compute

---

## Comparing DQL vs PPO — Metrics & Analysis

- Main metrics:
    - **Episode reward curves**
    - **Sample efficiency**
    - **Training stability (variance)**
    - **Convergence speed**
    - **Final policy score**

| Metric              | DQL      | PPO        |
|---------------------|----------|------------|
| Sample Efficiency   | ⭐⭐⭐⭐     | ⭐⭐⭐       |
| Stability           | ⭐⭐⭐      | ⭐⭐⭐⭐⭐      |
| Continuous Actions  | ⭐⭐       | ⭐⭐⭐⭐⭐      |
| Discrete Actions    | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐       |
| Complexity          | ⭐⭐⭐      | ⭐⭐⭐⭐       |
| Final Performance   | ⭐⭐⭐⭐     | ⭐⭐⭐⭐⭐      |

- Use `Results.py` for visualization and plotting after experiments.

---

## General Instructions
- Use `Lab2.ipynb` for interactive, step-by-step coding and advanced visualizations
- Scripts allow for fast re-training and batch evaluation/comparison
- All dependencies are in `requirements.txt`
- Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=../logs/
```

## References & Further Reading
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [OpenAI Spinning Up RL Docs](https://spinningup.openai.com/)
- [Original DQN paper](https://arxiv.org/abs/1312.5602)
- [Original PPO paper](https://arxiv.org/abs/1707.06347)

---

For complete code and all details, use the notebook or scripts provided in Lab2.
