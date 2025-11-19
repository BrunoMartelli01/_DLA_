# Lab 2 - Reinforcement Learning: DQL & PPO

## Overview

This lab implements and compares two fundamental reinforcement learning algorithms:
- **Deep Q-Learning (DQL)** - A value-based approach that learns action-value functions
- **Proximal Policy Optimization (PPO)** - A policy gradient method with improved stability

## Files Description

- **Lab2.ipynb** - Main Jupyter notebook with experiments and comparative analysis
- **DQL.py** - Implementation of Deep Q-Learning algorithm
- **PPO.py** - Implementation of Proximal Policy Optimization
- **Eval.py** - Script for evaluating trained agents
- **Results.py** - Utilities for visualizing training results and performance metrics

## Running the Lab

### Option 1: Interactive Notebook

```bash
jupyter notebook Lab2.ipynb
```

This is the recommended way to explore both algorithms step-by-step with visualizations and comparisons.

### Option 2: Standalone Scripts

**Training DQL Agent:**
```bash
python DQL.py
```

**Training PPO Agent:**
```bash
python PPO.py
```

**Evaluation:**
```bash
python Eval.py
```

**Results Visualization:**
```bash
python Results.py
```

## Algorithms Overview

### Deep Q-Learning (DQL)

**Type:** Value-based reinforcement learning

**Key Concepts:**
- Learns Q-values Q(s,a) representing expected future rewards
- Uses experience replay for sample efficiency
- Employs target network for training stability
- Epsilon-greedy exploration strategy

**Advantages:**
- Sample efficient with experience replay
- Well-suited for discrete action spaces
- Relatively simple to implement

**Challenges:**
- Overestimation bias
- Can be unstable with continuous actions
- Requires careful tuning of exploration parameters

### Proximal Policy Optimization (PPO)

**Type:** Policy gradient method

**Key Concepts:**
- Learns policy π(a|s) directly
- Clipped surrogate objective prevents destructive updates
- Uses Generalized Advantage Estimation (GAE)
- Multiple epochs on collected trajectories

**Advantages:**
- More stable training than vanilla policy gradients
- Works well with both continuous and discrete actions
- Good balance between simplicity and performance

**Challenges:**
- Requires more samples than DQL
- Sensitive to hyperparameter choices
- May converge to local optima

## Experimental Results

### Comparison Metrics

Both algorithms are evaluated on:

1. **Episode Rewards** - Total reward per episode over training
2. **Sample Efficiency** - Steps required to reach target performance
3. **Training Stability** - Variance in performance across runs
4. **Convergence Speed** - Time to reach optimal policy
5. **Final Performance** - Maximum achieved reward

### Expected Findings

**DQL typically shows:**
- Faster initial learning on simpler tasks
- More sample-efficient early training
- Potential for higher variance in performance
- Better suited for discrete action environments

**PPO typically shows:**
- More stable training curves
- Better final performance on complex tasks
- Smoother convergence
- Superior performance on continuous control

### Environments Tested

Common Gymnasium environments used:
- **CartPole-v1** - Classic control with discrete actions
- **LunarLander-v2** - Landing task with discrete actions
- **MountainCar-v0** - Exploration challenge
- **BipedalWalker-v3** - Continuous control task (PPO advantage)

## Key Concepts Demonstrated

### DQL Specific
- **Experience Replay Buffer** - Breaking temporal correlations in data
- **Target Networks** - Stabilizing Q-value updates
- **Epsilon Decay** - Balancing exploration vs exploitation
- **Double DQL** - Reducing overestimation bias

### PPO Specific
- **Clipped Objective** - Preventing too large policy updates
- **Advantage Estimation** - Reducing variance in gradient estimates
- **Value Function Learning** - Parallel critic for advantage computation
- **Policy Entropy** - Encouraging exploration

## Hyperparameters

### DQL Common Settings
```python
learning_rate = 1e-3
buffer_size = 100000
batch_size = 64
gamma = 0.99
target_update_frequency = 1000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
```

### PPO Common Settings
```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
```

## Monitoring Training

To monitor training progress in real-time:

```bash
tensorboard --logdir=../logs/
```

Navigate to `http://localhost:6006` to view:
- Reward curves for both algorithms
- Loss values (Q-loss for DQL, policy/value loss for PPO)
- Exploration metrics (epsilon for DQL, entropy for PPO)
- Episode lengths and success rates

## Dependencies

Key packages required:
- `stable-baselines3` - Implementations of both algorithms
- `gymnasium` - RL environments
- `torch` - Neural network backend
- `tensorboard` - Training monitoring
- `matplotlib` - Results visualization
- `numpy` - Numerical operations

## Troubleshooting

### DQL Not Learning
- Check epsilon decay rate (may be exploring too long)
- Verify replay buffer is filling properly
- Ensure target network updates are occurring
- Try reducing learning rate

### PPO Not Learning
- Increase number of training steps
- Adjust clip range (try 0.1 or 0.3)
- Tune GAE lambda parameter
- Check that advantage normalization is enabled

### Unstable Training (Both)
- Reduce learning rate
- Increase batch size
- Check reward scaling
- Verify environment is resetting properly

### Slow Convergence
- For DQL: Increase batch size or learning rate
- For PPO: Increase n_steps or n_epochs
- Adjust network architecture (add layers/neurons)
- Check that GPU is being utilized

## Expected Outcomes

After completing this lab, you should:

- Understand the fundamental differences between value-based and policy-based RL
- Recognize when to use DQL vs PPO based on problem characteristics
- Be able to tune hyperparameters for both algorithms
- Interpret training curves and diagnose common issues
- Appreciate the trade-offs between sample efficiency and stability

## Performance Comparison

| Metric | DQL | PPO |
|--------|-----|-----|
| Sample Efficiency | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Training Stability | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Continuous Actions | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Discrete Actions | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Implementation Complexity | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Final Performance | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - Original DQN paper
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Original PPO paper
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [OpenAI Spinning Up](https://spinningup.openai.com/) - RL educational resource
