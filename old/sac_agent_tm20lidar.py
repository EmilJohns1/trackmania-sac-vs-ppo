import pickle
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from tmrl import get_environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """
    The Actor network for the Soft Actor-Critic agent, producing actions based on observations.

    Args:
        obs_dim (int): Dimension of the observation space.
        act_dim (int): Dimension of the action space.
    """

    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.log_std.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(min=-20, max=2)
        std = torch.exp(log_std).clamp(min=1e-6)
        return mu, std


class Critic(nn.Module):
    """
    The Critic network for the Soft Actor-Critic agent, estimating Q-values based on state and action pairs.

    Args:
        obs_dim (int): Dimension of the observation space.
        act_dim (int): Dimension of the action space.
    """

    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q_value = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_value(x)


class SACAgent:
    """
    The Soft Actor-Critic agent responsible for learning policies and value functions.

    Args:
        obs_dim (int): Dimension of the observation space.
        act_dim (int): Dimension of the action space.
    """

    def __init__(self, obs_dim, act_dim, alpha=0.1):
        # Define networks
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic1 = Critic(obs_dim, act_dim).to(device)
        self.critic2 = Critic(obs_dim, act_dim).to(device)
        self.target_critic1 = Critic(obs_dim, act_dim).to(device)
        self.target_critic2 = Critic(obs_dim, act_dim).to(device)

        self.target_entropy = -act_dim * 0.5
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        self.alpha = self.log_alpha.exp()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=1e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=1e-4)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Initialize mean and std for normalization
        self.obs_mean = np.zeros(obs_dim)
        self.obs_var = np.ones(obs_dim)
        self.obs_count = 1e-5  # Small constant to prevent division by zero

        self.ewc_lambda = 1000.0  # Regularization strength for EWC
        self.fisher_matrix = {}
        self.prev_params = {}


    """Action sampling method for the SAC agent"""
    def sample_action(self, obs):
        mu, std = self.actor(obs)

        std = std.clamp(min=1e-6)
        dist = torch.distributions.Normal(mu, std)

        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)

        return action, log_prob.sum(1, keepdim=True)


    """Method to update the parameters of the SAC agent"""
    def update_parameters(self, replay_buffer, gamma=0.99, tau=0.005):
        obs, action, reward, next_obs, done = replay_buffer.sample(replay_buffer.batch_size)
        with torch.no_grad():
            next_action, log_prob = self.sample_action(next_obs)
            target_q1 = self.target_critic1(next_obs, next_action)
            target_q2 = self.target_critic2(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            q_target = reward + (1 - done) * gamma * target_q

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=0.5)

        new_action, log_prob = self.sample_action(obs)
        q1_new = self.critic1(obs, new_action)
        q2_new = self.critic2(obs, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().clamp(min=0.01)

        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return reward

    """Method to update the observation statistics for normalization"""
    def update_obs_stats(self, obs):
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += delta * delta2

    """Method to preprocess the observation data for the SAC agent"""
    def preprocess_obs(self, obs):
        # Handle the case where obs has an empty dictionary as the second element
        if (
            isinstance(obs, tuple)
            and len(obs) > 1
            and isinstance(obs[1], dict)
            and not obs[1]
        ):
            obs_data = obs[0]  # Use the first element directly if there's an empty dict
        else:
            obs_data = obs  # Otherwise, use the entire tuple as expected

        expected_obs_size = 85

        # Check if obs_data contains the expected components for normal processing
        if isinstance(obs_data, tuple) and len(obs_data) == 4:
            # Process each component of the observation data
            speed = np.asarray(obs_data[0]).flatten()
            lidar = np.asarray(obs_data[1]).reshape(-1)  # Flatten it to shape (76,)
            prev_action_1 = np.asarray(obs_data[2]).flatten()  # (3,)
            prev_action_2 = np.asarray(obs_data[3]).flatten()  # (3,)

            # Check for NaNs
            if (
                np.any(np.isnan(speed))
                or np.any(np.isnan(lidar))
                or np.any(np.isnan(prev_action_1))
                or np.any(np.isnan(prev_action_2))
            ):
                print("NaN detected in observation components")

            # Compute the current and next distances to the centerline
            current_centerline_distance = calculate_centerline_distance(obs)
            next_lidar = np.asarray(
                obs_data[1][1]
            )  # Assuming the next lidar scan is here
            next_centerline_distance = next_lidar[0] - next_lidar[18]

            # Concatenate all components into a single observation vector
            concatenated_obs = np.concatenate(
                [
                    speed,
                    lidar,
                    prev_action_1,
                    prev_action_2,
                    [current_centerline_distance, next_centerline_distance],
                ]
            )

            # Ensure the observation has the expected size by padding if necessary
            if concatenated_obs.shape[0] != expected_obs_size:
                print("Observation size mismatch. Padding or truncating to fit.")
                concatenated_obs = np.pad(
                    concatenated_obs,
                    (0, max(0, expected_obs_size - concatenated_obs.shape[0])),
                )[:expected_obs_size]
        else:
            # Handle the unexpected structure case by returning a zero-filled tensor
            print("Unexpected observation structure:", obs_data)
            concatenated_obs = np.zeros(expected_obs_size)

        # Update statistics with the new observation
        self.update_obs_stats(concatenated_obs)

        # Normalize using the current mean and std deviation
        normalized_obs = (concatenated_obs - self.obs_mean) / np.sqrt(
            self.obs_var / self.obs_count
        )

        return torch.tensor(normalized_obs, dtype=torch.float32).to(device)

    def compute_fisher_matrix(self, replay_buffer, num_samples=100):
        """
        Computes the Fisher Information Matrix (FIM) using the replay buffer.
        Args:
            replay_buffer: Replay buffer to sample data from.
            num_samples: Number of samples to use for computing FIM.
        """
        self.fisher_matrix = {}
        self.prev_params = {}

        # Ensure there are enough samples in the buffer for the specified number of samples
        if replay_buffer.size() >= num_samples:
            for i in range(num_samples):
                # Dynamically adjust batch size to be less than or equal to buffer size
                batch_size = min(replay_buffer.batch_size, replay_buffer.size())

                # Sample a batch from the buffer
                obs, action, _, _, _ = replay_buffer.sample(batch_size)  # Sample a batch of adjusted size
                mu, std = self.actor(obs)
                dist = torch.distributions.Normal(mu, std)
                log_probs = dist.log_prob(action).sum(dim=1)

                # Compute gradients for the log probabilities
                self.actor_optimizer.zero_grad()
                log_probs.mean().backward(retain_graph=True)

                for name, param in self.actor.named_parameters():
                    if param.grad is not None:
                        if name not in self.fisher_matrix:
                            self.fisher_matrix[name] = param.grad.data.clone().pow(2)
                        else:
                            self.fisher_matrix[name] += param.grad.data.clone().pow(2)

            # Normalize Fisher matrix and store parameters
            for name, param in self.actor.named_parameters():
                if name in self.fisher_matrix:
                    self.fisher_matrix[name] /= num_samples
                self.prev_params[name] = param.data.clone()


    def ewc_loss(self):
        """
        Computes the EWC loss to regularize updates.
        Returns:
            Regularization loss term for EWC.
        """
        loss = 0.0
        for name, param in self.actor.named_parameters():
            if name in self.fisher_matrix:
                fisher_value = self.fisher_matrix[name]
                prev_param_value = self.prev_params[name]
                loss += (fisher_value * (param - prev_param_value).pow(2)).sum()
        return self.ewc_lambda * loss

class ReplayBuffer:
    """
    Replay buffer for storing experiences and sampling mini-batches for training.

    Args:
        buffer_size (int): Maximum size of the replay buffer.
        batch_size (int): Size of mini-batches to sample.
    """

    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size


    """Stores a new experience tuple in the replay buffer"""
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))


    """Samples a mini-batch of experiences from the replay buffer"""
    def sample(self, batch_size):
        if self.batch_size < batch_size:
            batch_size = self.batch_size
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.float32).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device),
        )

    """Returns the current size of the replay buffer"""
    def size(self):
        return len(self.buffer)


"""Calculates the distance to the centerline from the lidar scan data"""
def calculate_centerline_distance(obs):
    lidar = np.asarray(obs[1]).reshape(-1)
    return lidar[0] - lidar[18]


"""Saves the graph data so it doesn't reset every time we rerun our agent"""
def save_graph_data(cumulative_rewards, fastest_lap_times, steps_record, last_step, filename="graphs/graph_data.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({
            'cumulative_rewards': cumulative_rewards,
            'fastest_lap_times': fastest_lap_times,
            'steps_record': steps_record,
            'last_step': last_step
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Graph data saved to {filename}.")


"""Loads the graph data"""
def load_graph_data(filename="graphs/graph_data.pkl"):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Graph data loaded from {filename}.")
        return (
            data['cumulative_rewards'],
            data['fastest_lap_times'],
            data['steps_record'],
            data.get('last_step', 0)  # Default to 0 if 'last_step' is not found
        )
    except FileNotFoundError:
        print(f"No graph data found at {filename}, starting fresh.")
        return [], [], [], 0


"""Function to plot and save graphs"""
def plot_and_save_graphs(steps, cumulative_rewards, fastest_lap_times, steps_record, filename_prefix="graphs/performance"):
    # Plot cumulative reward vs. steps
    plt.figure(figsize=(12, 6))

    # Cumulative Reward vs Steps
    plt.subplot(1, 2, 1)
    plt.plot(steps_record, cumulative_rewards, label="Cumulative Reward")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward vs Steps")
    plt.legend()

    # Fastest Lap Time vs Steps
    plt.subplot(1, 2, 2)
    plt.plot(steps_record, fastest_lap_times, label="Fastest Lap Time")
    plt.xlabel("Steps")
    plt.ylabel("Fastest Lap Time (s)")
    plt.title("Fastest Lap Time vs Steps")
    plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_step_{steps}.png")
    plt.close()
    print(f"Saved graphs at step {steps}.")


"""Saves the current values of the SAC agent)"""
def save_agent(agent, filename="agents/sac_agent_tm20lidar.pth"):
    torch.save(
        {
            "actor_state_dict": agent.actor.state_dict(),
            "critic1_state_dict": agent.critic1.state_dict(),
            "critic2_state_dict": agent.critic2.state_dict(),
            "target_critic1_state_dict": agent.target_critic1.state_dict(),
            "target_critic2_state_dict": agent.target_critic2.state_dict(),
            "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": agent.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": agent.critic2_optimizer.state_dict(),
            "log_alpha": agent.log_alpha,
            "alpha_optimizer_state_dict": agent.alpha_optimizer.state_dict(),
        },
        filename,
    )
    print("Agent saved to", filename)


"""Saves the replay buffer using pickle"""
def save_replay_buffer(replay_buffer, filename="agents/sac_replay_buffer_tm20lidar.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(replay_buffer.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Replay buffer saved to", filename)


"""Loads the saved SAC agent if it exists."""
def load_agent(agent, filename="agents/sac_agent_tm20lidar.pth"):
    checkpoint = torch.load(filename, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic1.load_state_dict(checkpoint["critic1_state_dict"])
    agent.critic2.load_state_dict(checkpoint["critic2_state_dict"])
    agent.target_critic1.load_state_dict(checkpoint["target_critic1_state_dict"])
    agent.target_critic2.load_state_dict(checkpoint["target_critic2_state_dict"])

    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    agent.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer_state_dict"])
    agent.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer_state_dict"])
    agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])

    agent.log_alpha = checkpoint["log_alpha"]
    print("Agent loaded from", filename)


"""Loads the saved replay buffer for the SAC agent using pickle"""
def load_replay_buffer(replay_buffer, filename="agents/sac_replay_buffer_tm20lidar.pkl"):
    with open(filename, 'rb') as f:
        replay_buffer.buffer = pickle.load(f)  # Use pickle.load for loading
    print("Replay buffer loaded from", filename)


env = get_environment()

agent = SACAgent(85, 3)

BUFFER_SIZE = int(5e5)
BATCH_SIZE = 512

replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

agent_checkpoint_file = "agents/sac_agent_tm20lidar.pth"
replay_buffer_file = "agents/sac_replay_buffer_tm20lidar.pkl"

try:
    load_agent(agent, agent_checkpoint_file)
    load_replay_buffer(replay_buffer, replay_buffer_file)
except FileNotFoundError:
    print("No saved agent or replay buffer found, starting fresh.")

# Start training loop
obs, info = env.reset()  # Initial environment reset
reward = 0
THRESHOLD = 1.8

cumulative_rewards, fastest_lap_times, steps_record, last_saved_step = load_graph_data()
steps = last_saved_step

cumulative_reward = 0
current_reward = 0
modified_reward = 0
episode_time = 0
fastest_lap_time = float("inf")
episode_start_time = time.time()


for step in range(10000000):
    processed_obs = agent.preprocess_obs(obs)

    act, log_prob = agent.sample_action(
        torch.tensor(processed_obs, dtype=torch.float32).unsqueeze(0).to(device)
    )
    action = (
        act.cpu().detach().numpy().flatten()
    )

    speed = obs[0]

    modified_reward = 0

    if speed < 10 and abs(action[1]) > 0 and abs(action[2]) < 0.6:
        action[1] = 0

    if speed < 10 and action[0] < 0.9 and abs(action[2]) < 0.6:
        action[0] = 1.0

    # Take a step in the environment and get updated reward
    obs_next, reward, terminated, truncated, info = env.step(action)

    # Accumulate the reward
    modified_reward += reward
    cumulative_reward += modified_reward

    done = terminated or truncated
    if done:
        # Calculate the total episode time for the lap
        episode_time = time.time() - episode_start_time
        episode_start_time = time.time()

        # Check for a new fastest lap time
        if (episode_time < fastest_lap_time) and modified_reward > 25:
            print("new fastest lap time")
            fastest_lap_time = episode_time

        # Record cumulative reward and fastest lap time
        cumulative_rewards.append(cumulative_reward)
        fastest_lap_times.append(fastest_lap_time)
        steps_record.append(step)

        # Reset cumulative reward for the next episode
        cumulative_reward = 0

        # Compute and store the Fisher matrix for EWC after each episode/task
        agent.compute_fisher_matrix(replay_buffer)

        # Reset environment
        obs, info = env.reset()
    else:
        # Update observation for the next step
        obs = obs_next

        # Store experience in replay buffer and update parameters
    replay_buffer.store(
        processed_obs, action, modified_reward, agent.preprocess_obs(obs_next), done
    )

    if replay_buffer.size() >= BATCH_SIZE:
        agent.update_parameters(replay_buffer)

    # Save agent and replay buffer periodically
    if step % 10000 == 0 and step != 0:
        save_agent(agent, agent_checkpoint_file)
        save_replay_buffer(replay_buffer, replay_buffer_file)

        #Save graph data
        steps = last_saved_step + step

        save_graph_data(cumulative_rewards, fastest_lap_times, steps_record, steps)

        # Plot and save graphs every 10,000 steps
        plot_and_save_graphs(steps, cumulative_rewards, fastest_lap_times, steps_record)

        time.sleep(1)