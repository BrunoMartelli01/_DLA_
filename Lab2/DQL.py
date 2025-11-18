import datetime

from gymnasium.vector import AsyncVectorEnv
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter


class QNetwork(nn.Module):
    """
Goal
To create a neural network that can take a stack of four preprocessed game frames as input and output a vector
of Q-values, one for each possible action in the game. The Q-value for a given action represents the predicted
total future reward the agent can expect to receive if it takes that action in the current state and continues
to play optimally thereafter.

Method
The network architecture is specifically designed to process visual information and is composed of two main parts:
a convolutional feature extractor and a fully-connected head.
    1. Convolutional Base: This part of the network processes the input image to identify important
                           spatial features.
        `nn.Conv2d`: Three sequential convolutional layers with decreasing kernel sizes and strides
                    (`8x8`, `4x4`, `3x3`). This hierarchical structure allows the network to first learn simple
                    features (like edges and corners) and then combine them into more complex ones. The input has
                    4 channels, representing the four stacked historical frames, which allows the network to infer
                    dynamic information like the direction and speed of objects.
       `F.relu`: ReLU activation is applied after each convolutional layer to introduce non-linearity, enabling
                 the network to learn more complex patterns.

    2.  Flattening: After the convolutional layers have extracted a rich set of 2D feature maps, the
                    `x.view(x.size(0), -1)` operation flattens this multi-dimensional data into a single, long vector.
                    This prepares the data to be processed by the dense layers.

    3.  Fully-Connected Head: This part of the network takes the flattened features and uses them to make the final Q-value
                              predictions.

            `nn.Linear(64 * 7 * 7, 512)`: A dense hidden layer that combines all the visual features to form a
                                          high-level representation of the game state. The input size (`64 * 7 * 7`) is
                                          derived from the varius convolutional layer: (4, 84, 84) -> (32, 20, 20) ->
                                          -> (64, 9, 9) -> (64, 7, 7).

            `nn.Linear(512, num_actions)`: The final output layer. It maps the 512-dimensional state representation
                                          to a vector with a size equal to the number of possible actions. Crucially,
                                          there is no final activation function (like Softmax). The output consists
                                          of raw, unbounded Q-values, as these are direct estimates of future rewards,
                                          not probabilities.

    4.  Forward Pass: The `forward` method defines the flow of data. A critical first step is normalizing the input
                      pixel values (`x / 255.0`) to a `[0, 1]` range, which is essential for stable training of deep
                      neural networks.

Result
*   An instance of this `QNetwork` is a callable object that embodies the Q-function.
*   When a state  is passed to it, it returns a `[batch_size, num_actions]` tensor.
*   Each row in the output tensor contains the estimated Q-values for every possible action from the corresponding
    input state.
    """

    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent:
    def __init__(self, env, buffer_capacity=10000, batch_size=32, gamma=0.99, lr=1e-4, stay=True):
        self.env = env
        self.num_actions = env.action_space.n
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stay = stay
        self.q_net = QNetwork(self.num_actions).to(self.device)
        self.q_net.train()
        self.target_net = QNetwork(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity, self.device)

        self.steps_done = 0

        self.epsilon = 1
        self.epsilon_decay = 0.9999925
        self.epsilon_min = 0.05

    def select_action(self, state_batch: torch.Tensor, explore=True):
        """
    Goal
    To select an action for each state in a batch, implementing the epsilon-greedy policy. This means the agent will
    "exploit" by choosing the action with the highest predicted Q-value most of the time, but will "explore" by
    choosing a completely random action with a small probability, epsilon.

    Method
    * With probability epsilon, the agent explores by selecting a completely random action from the environment.

    * Otherwise (with probability 1-epsilon), the agent exploits its knowledge. It uses the q_net to predict the
      Q-values for all possible actions and greedily chooses the action with the highest value using argmax.

    * During training, the epsilon value is decayed after each step, causing the agent to gradually shift from
      exploration to exploitation as it becomes more confident in its policy.

    Result
    * The function returns a tensor of actions, one for each state in the input batch. These actions can then be passed
      to the environment's step function.

            """
        if not isinstance(state_batch, torch.Tensor):
            state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)

        if explore and (np.random.rand() < self.epsilon) and self.q_net.training:
            actions = torch.tensor(
                [self.env.action_space.sample() for _ in range(state_batch.shape[0])],
                device="cpu", dtype=torch.int64
            )
        else:
            with torch.no_grad():
                q_values = self.q_net(state_batch)
                actions = q_values.argmax(dim=1).to("cpu")

        if self.q_net.training and explore:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return actions

    def eval_model(self, num_episodes=10):
        """
    Goal
        To obtain a stable and unbiased estimate of the agent's true performance. This is achieved by:
        1. Running the agent for multiple episodes (num_episodes) and averaging the results to smooth out randomness.
        2. Forcing the agent to act purely greedily to measure its learned policy's raw skill.
        3. Executing these evaluation episodes in parallel to significantly speed up the process.
    Method
        1. Vectorized Environment Factory: An AsyncVectorEnv is created, which spins up num_episodes separate game
            environments, each running in its own process. This allows the agent to play all games simultaneously.
            The environments are created using a lambda function that calls the make_env helper, which applies all
            necessary pre-processing wrappers (grayscale, frame stacking, etc.).
        2. Forcing a Greedy Policy: Before the evaluation starts, the model is put into evaluation mode
            (self.q_net.eval()) and the exploration rate is set to zero (self.epsilon = 0.0). This ensures the agent's
            performance is a measure of its learned knowledge.
        3. The Vectorized Evaluation Loop: The core of the function is the while not finished.all(): loop, which runs
            until all parallel episodes have terminated.
            *   Action Selection: The agent selects an action for all active environments in a single batch using
                    self.select_action(state, explore=False).
            *   Step: eval_env.step() executes these actions in all environments at once and returns the results
                    as batches.
            *   Tracking Progress: The code updates the total rewards and lengths for each episode. The Not operator (~)
                    is used on finished to ensure that rewards and lengths are only updated for environments
                    that have not yet finished.
            *   Termination Check: The finished tensor is updated at the end of each step. The loop continues until
                    every element in this tensor is True.
            *   Cleanup and State Restoration: After the loop finishes, the function is a good citizen. It restores
                    the original epsilon value and sets the network back to training mode (self.q_net.train()).
                    It also properly closes the vectorized environment to free up resources.
    Result
        The function returns four values that provide a comprehensive summary of the agent's performance:
        *   rewards.mean().item(): The average total reward across all evaluation episodes.
        *   lengths.float().mean().item(): The average length of the episodes.
        """
        stay = self.stay
        eval_env = AsyncVectorEnv([lambda: make_env(stay=stay) for _ in range(num_episodes)])
        rewards = torch.zeros(num_episodes, dtype=torch.float32, device=self.device)
        lengths = torch.zeros(num_episodes, dtype=torch.int32, device=self.device)
        finished = torch.zeros(num_episodes, dtype=torch.bool, device=self.device)

        self.q_net.eval()
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # greedy

        state, _ = eval_env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        while not finished.all():
            actions = self.select_action(state, explore=False)  # torch.Tensor (num_envs,)
            next_state, reward, done, truncated, _ = eval_env.step(actions.numpy())

            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done = torch.tensor(done, dtype=torch.bool, device=self.device)
            truncated = torch.tensor(truncated, dtype=torch.bool, device=self.device)

            rewards += reward * (~finished)
            lengths += (~finished)

            state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            finished |= (done | truncated)

        self.epsilon = old_epsilon
        self.q_net.train()
        eval_env.close()

        return rewards.mean().item(), lengths.float().mean().item()

    def update_model(self):
        """
        Goal
            To update the weights of the main Q-network (q_net) to minimize the difference between its predicted
            Q-values and a more accurate "target" Q-value. This process, repeated thousands of times, is what allows
            the agent to learn an effective policy.
        Method
            1. Experience Sampling: The process begins by sampling a random batch of past transitions (state,
                action, reward, next_state, done) from the replay_buffer. Sampling randomly is crucial as it breaks the
                temporal correlation between consecutive steps, leading to much more stable training.

            2. Calculate the Predicted Q-values: This is what the network currently thinks the value is.
                *   self.q_net(state): The q_net predicts the Q-values for all possible actions from the initial state.
                *   .gather(1, action.unsqueeze(1)): This is a key step that uses the action tensor to "gather" or
                        select only the Q-values corresponding to the actions that were actually taken in the sampled
                        experiences.

            3. Calculate the Target Q-values: This is what the network should think the value is,
                based on the outcome. This entire block is wrapped in with torch.no_grad() because it's used to create
                a static target, not for gradient computation.
                *   self.target_net(next_state).max(1)[0]: The separate, slow-updating target_net is used to estimate
                        the value of the next state. It does this by predicting the Q-values for the next_state and
                        taking the maximum one. Using a separate network is a critical technique for stabilizing DQN
                        training, preventing the network from chasing a constantly moving prediction.
                *   expected_q_values = reward + self.gamma * next_q_values * (1 - done): .The target value is the
                        immediate reward plus the discounted (gamma) value of the best possible future action.
                        The (1 - done) term zeros out the future value if the episode has ended, as there is no future
                        from a terminal state.

            4. Compute Loss and Update Network:
                *   loss = F.mse_loss(...): The loss is the Mean Squared Error between the network's prediction
                        (q_values) and the calculated target (expected_q_values).
                *   Standard PyTorch backpropagation (optimizer.zero_grad(), loss.backward(), optimizer.step()) is
                        then used to update the weights of the q_net to minimize this error.
        Result
            *   The primary result is that the weights of the q_net have been adjusted slightly, making its future
                predictions for similar states more accurate.
            *   The function also returns the scalar loss.item(), a useful metric for logging and monitoring.
                A consistently decreasing loss indicates that the agent is successfully learning to
                predict the Q-values.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = state.squeeze(-1)
        q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_state = next_state.squeeze(-1)
            next_q_values = self.target_net(next_state).max(1)[0]
            expected_q_values = reward + self.gamma * next_q_values * (1 - done)
        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes, early_stop_patience=200, max_len=50,sync=5000, save_path="best_dqn_car_racing.pth",
              writer=None,eval_every=50, eval_episodes=15):
        """

    Goal
        To train the Q-network from scratch, saving the best-performing model discovered during the process. The training
        will stop early if the agent's performance stops improving.
    Method
        The function is built around a main for loop that iterates over a set number of num_episodes. Within this loop,
        a sequence of operations manages the agent's lifecycle:
            1. Initialization: Before the loop, it sets up deques to store recent history of rewards and losses,
                initializes counters for steps and patience, and creates a tqdm progress bar (pbar) for a  command-line
                interface.
            2. The Inner (Per-Step) Loop: For each episode, a while True loop runs, representing the agent's interaction
                with the environment on a step-by-step basis.
                * Act: The agent selects an action using its epsilon-greedy policy (self.select_action).
                * Step: The action is sent to the environment, which returns the next_state, reward, and termination
                    signals (done, truncated).
                * Store: The complete transition (state, action, reward, next_state, done) is pushed into the
                    replay_buffer.
                * Learn: Periodically (controlled by learn_every), the self.update_model() function is called to sample
                    a batch from the replay buffer and perform one gradient descent step on the q_net.
                * Sync Target Network: Much less frequently (controlled by sync), the weights of the main q_net are
                    copied over to the target_net. This is a crucial stability technique in DQN.
        3. Post-Episode Logging: After an episode ends, key metrics like the total episode reward and average loss are
            logged to history and, if a writer is provided, sent to TensorBoard for real-time visualization. This
            allows for monitoring the agent's learning progress.

        4. Periodic Evaluation and Early Stopping: This is the core of the training's efficiency and robustness.
            * Evaluation: Periodically (controlled by eval_every), the self.eval_model() function is called.
                This pauses training, turns off exploration, and runs the agent for several episodes to get a stable,
                unbiased estimate of its current performance (eval_avg).
            * Saving the Best Model: The current evaluation score (eval_avg) is compared to the best_avg_reward seen
                so far. If the current model is better, the best_avg_reward is updated, the patience counter is reset,
                and the model's weights are saved to a file (torch.save).
    Result
        The primary results of this function are:
            * A Saved Model File: The most valuable result is the saved model file (best_dqn_car_racing.pth).
                This file contains the weights of the agent at its peak performance, as determined by the periodic
                evaluations, not necessarily the weights from the very last training step.
            * TensorBoard Logs: If a writer was provided, a full set of logs is created, allowing for detailed
                post-training analysis of reward curves, loss, epsilon decay, and more.
        """
        rewards_history = deque(maxlen=max_len)
        losses_history = deque(maxlen=max_len)
        n_steps = 0
        learn_every = 4
        best_avg_reward = -float("inf")
        patience_counter = 0
        eval_avg = 0
        pbar = trange(num_episodes, desc="Training", unit="episode")
        for episode in pbar:
            state, _ = self.env.reset()
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0) # (1,S,H,W)
            episode_reward = 0
            episode_losses = []

            while True:
                n_steps += 1
                action = self.select_action(state)[0].item()

                next_state, reward, done, truncated, _ = self.env.step(action)

                self.replay_buffer.push(state.squeeze(0), action, reward, next_state, done)

                episode_reward += reward
                state = torch.from_numpy(next_state).float().to(self.device).unsqueeze(0)  # (1,S,H,W)

                if n_steps % sync == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                if len(self.replay_buffer) >= self.batch_size and n_steps % learn_every == 0:
                    loss = self.update_model()
                    if loss is not None:
                        episode_losses.append(loss)
                        if writer:
                            writer.add_scalar('Training/Loss', loss, n_steps)

                if done or truncated:
                    break

            rewards_history.append(episode_reward)
            if episode_losses:
                losses_history.append(np.mean(episode_losses))

            avg_reward = np.mean(rewards_history) if rewards_history else 0
            avg_loss = np.mean(losses_history) if losses_history else 0
            if writer:
                writer.add_scalar('Rewards/Episode_Reward', episode_reward, episode)
                writer.add_scalar(f'Rewards/Average_Reward_{max_len}_Episodes', avg_reward, episode)
                writer.add_scalar('Parameters/Epsilon', self.epsilon, episode)

            if (episode) % eval_every == 0:
                eval_avg, eval_len_avg = self.eval_model(num_episodes=eval_episodes)
                if writer:
                    writer.add_scalar("Eval/Average_Reward", eval_avg, episode)
                    writer.add_scalar("Eval/Average_Episode_Length", eval_len_avg, episode)
                if eval_avg > best_avg_reward:
                    best_avg_reward = eval_avg
                    patience_counter = 0
                    torch.save(self.q_net.state_dict(), save_path)
                else:
                    patience_counter += 1

            pbar.set_postfix({
                "Ep.Reward": f"{episode_reward:.1f}",
                "Last Eval": f"{eval_avg:.1f}",
                "Best.Eval": f"{best_avg_reward:.1f}",
                "Avg.Loss": f"{avg_loss:.4f}",
                "Pat": patience_counter
            })

            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping! Best eval reward: {best_avg_reward:.1f}")
                break


class ReplayBuffer:
    """

    Goal
        To serve as the agent's memory, storing past experiences and providing randomized mini-batches to enable
        stable, off-policy training.
    Method
         * It uses a deque with a fixed capacity to efficiently store a large but finite number of recent experiences.
         * The push method pre-processes each experience by immediately converting it to a tensor on the target
            device (e.g., the GPU) for optimization.
         * The core sample method randomly selects a batch of these stored experiences, effectively breaking their
            sequential correlation.
    Result
        It provides the main training loop with decorrelated batches of experience, perfectly formatted and ready for
            the GPU.

    """
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        next_state = torch.from_numpy(next_state).float().to(self.device)

        # Unsqueeze state and next_state to add a batch dimension for storage
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        action = torch.tensor([action], dtype=torch.int64, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.tensor([done], dtype=torch.float32, device=self.device)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return torch.cat(state), torch.cat(action), torch.cat(reward), torch.cat(next_state), torch.cat(done)

    def __len__(self):
        return len(self.buffer)


class StayOnTrack(gym.Wrapper):
    """
    Wrapper per CarRacing-v3 che modifica la ricompensa per incentivare una guida sicura.
    - Rimuove la penalità di -0.1 per ogni frame.
    - Applica una forte penalità per l'uscita di strada (erba).
    - Offre un piccolo bonus per rimanere in pista.
    """

    def __init__(self, env, off_track_penalty=-10.0, on_track_reward=0.2):
        super().__init__(env)
        self.off_track_penalty = off_track_penalty
        self.on_track_reward = on_track_reward

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        new_reward = reward+0.1
        patch_erba = obs[75:80, 42:53]  # Area under the vehicle
        mean_rgb = np.mean(patch_erba, axis=(0, 1))
        if mean_rgb[1] > 135:  # Found it empiricaly
            new_reward += self.off_track_penalty
        else:
            new_reward += self.on_track_reward
        return obs, new_reward, done, truncated, info


class CropObservation(gym.ObservationWrapper):
    """
    Goal
        To improve the signal-to-noise ratio of the input image. By removing the dashboard, which contains constantly
        changing for rewards velocity and direction, the agent is not distracted by irrelevant information and can
        focus its learning capacity entirely on the track ahead.
    Method
        1. Redefining the Observation Space:  The __init__ method redefines the environment's observation_space to
            inform the agent that it will now receive square (84, 84) images.
        2. The Cropping Logic (observation method): This method is automatically called on every new frame from the
            environment. It applies a NumPy slice obs[:84, 6:90] which does two things:
                * The primary action is [:84, ...], which selects the top 84 rows of the 96-pixel-tall image.
                    This effectively cuts off the bottom 12 pixels, which contain the distracting dashboard display.
                * The secondary slice [..., 6:90] trims the side borders to create the final 84x84 shape.
    Result
        The agent receives a smaller, cleaner 84x84 frame that contains only visual information about the track and its
            surroundings. This leads to significantly more effective and stable learning, as the agent is not trying to
            find correlations in the dashboard.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84),  # After Grayscale
            dtype=np.uint8
        )

    def observation(self, obs):
        return obs[:84, 6:90]


OFF_TRACK_PENALTY = -3.0
ON_TRACK_REWARD = 0.15


def make_env(render=False, continuous=False, max_episode_steps=1500, stay=True):
    env = gym.make("CarRacing-v3", continuous=continuous, render_mode="human" if render else None,
                   max_episode_steps=max_episode_steps, )
    if stay:
        env = StayOnTrack(env, off_track_penalty=-3.0, on_track_reward=0.05)
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = CropObservation(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


if __name__ == '__main__':
    import os

    base_env = make_env(render=False, continuous=False, max_episode_steps=1500, stay=False)
    agent = DQNAgent(base_env, stay=False)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"./Lab2/runs/dqn_car_racing_{date}"
    writer = SummaryWriter(log_dir)
    best_model_path = f'./Lab2/dqn/best_dqn_car_racing_{date}.pth'
    agent.train(num_episodes=10000, early_stop_patience=10, max_len=50, sync=4, save_path=best_model_path,
                writer=writer)
    writer.close()
