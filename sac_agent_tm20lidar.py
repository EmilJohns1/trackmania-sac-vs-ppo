import pickle
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

        # Initialize weights and biases
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
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
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

    def __init__(self, obs_dim, act_dim, alpha_start=0.2, alpha_decay=0.99):
        # Define networks
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic1 = Critic(obs_dim, act_dim).to(device)
        self.critic2 = Critic(obs_dim, act_dim).to(device)
        self.target_critic1 = Critic(obs_dim, act_dim).to(device)
        self.target_critic2 = Critic(obs_dim, act_dim).to(device)

        self.target_entropy = -act_dim
        self.log_alpha = torch.tensor(
            np.log(alpha_start), requires_grad=True, device=device
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = self.log_alpha.exp()
        self.alpha_decay = alpha_decay

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def sample_action(self, obs):
        mu, std = self.actor(obs)

        std = std.clamp(min=1e-6)
        dist = torch.distributions.Normal(mu, std)

        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)

        return action, log_prob.sum(1, keepdim=True)

    def update_parameters(self, replay_buffer, gamma=0.99, tau=0.005):
        obs, action, reward, next_obs, done = replay_buffer.sample()
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

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)

        new_action, log_prob = self.sample_action(obs)
        q1_new = self.critic1(obs, new_action)
        q2_new = self.critic2(obs, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        target_gas = 0.8
        gas_regularization = F.mse_loss(
            new_action[:, 0], torch.full_like(new_action[:, 0], target_gas)
        )
        actor_loss += 0.2 * gas_regularization

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.alpha_decay < 1.0:
            self.log_alpha.data = self.log_alpha.data * self.alpha_decay
            self.alpha = self.log_alpha.exp()

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


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.float32).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device),
        )

    def size(self):
        return len(self.buffer)


def preprocess_obs(obs):
    # Check if obs contains an empty dictionary as the second element
    if (
        isinstance(obs, tuple)
        and len(obs) > 1
        and isinstance(obs[1], dict)
        and not obs[1]
    ):
        # Handle the case where obs has an empty dict as the second element
        obs_data = obs[0]  # Use the first element directly if there's an empty dict
    else:
        obs_data = obs  # Otherwise, use the entire tuple as expected

    # Define the expected observation size
    expected_obs_size = 85

    # Check if obs_data contains the expected components for normal processing
    if isinstance(obs_data, tuple) and len(obs_data) == 4:
        # Process each component of the observation data
        speed = np.asarray(
            obs_data[0]
        ).flatten()  # Assuming this is a single-element array
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

        # Current distance to centerline (difference between lidar[0] and lidar[18])
        current_centerline_distance = lidar[0] - lidar[18]

        # Next distance to centerline (using the next lidar scan obs[1][1])
        next_lidar = np.asarray(
            obs_data[1][1]
        )  # Next lidar scan (shape should be (19,))
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

    # Convert to torch tensor and return
    return torch.tensor(concatenated_obs, dtype=torch.float32).to(device)


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


"""Saves the replay buffer"""


def save_replay_buffer(
    replay_buffer, filename="agents/sac_replay_buffer_tm20lidar.pkl"
):
    with open(filename, "wb") as f:
        pickle.dump(replay_buffer.buffer, f)
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


"""Loads the saved replay buffer for the SAC agent"""


def load_replay_buffer(
    replay_buffer, filename="agents/sac_replay_buffer_tm20lidar.pkl"
):
    with open(filename, "rb") as f:
        replay_buffer.buffer = pickle.load(f)
    print("Replay buffer loaded from", filename)


env = get_environment()

obs_dim = preprocess_obs(env.reset()).shape[0]
act_dim = env.action_space.shape[0]

print(obs_dim)
print(act_dim)

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256

replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

agent = SACAgent(obs_dim, act_dim)

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
previous_speed = 0

for step in range(10000000):  # Total number of training steps
    time.sleep(0.01)  # Small delay to match environment timing

    # Preprocess observation
    processed_obs = preprocess_obs(obs)

    # Convert processed observation to tensor for action sampling
    act, log_prob = agent.sample_action(
        torch.tensor(processed_obs, dtype=torch.float32).unsqueeze(0).to(device)
    )
    action = (
        act.cpu().detach().numpy().flatten()
    )  # Convert action to numpy for environment

    # Get the speed from the observation (obs[0])
    speed = obs[0]

    # 1. Penalize for braking when speed is below 10
    if speed < 15 and action[1] > 0:
        reward -= 1  # Penalize for braking at low speed
        action[1] = 0  # Make braking illegal by setting brake action to 0

    # 2. Penalize for not accelerating when speed is below 5 (action[0] should be 1)
    if speed < 15 and action[0] != 1:
        reward -= 1  # Penalize for not accelerating at low speed
        action[0] = 1  # Force acceleration (throttle) to 1

    # 3. Prevent steering when speed is below 5 (action[2] corresponds to steering)
    if speed < 15 and action[2] != 0:
        reward -= 1
        action[2] = 0  # Make steering illegal by setting action[2] to 0

    speed_diff = speed - previous_speed

    max_penalty = 5
    max_reward = 5
    if speed_diff > 0:
        reward += min(speed_diff, max_reward)
    else:
        reward -= min(-speed_diff, max_penalty)

    previous_speed = speed

    # Take a step in the environment
    obs_next, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated  # Check if episode is done

    # Store experience in replay buffer
    replay_buffer.store(processed_obs, action, reward, preprocess_obs(obs_next), done)

    # Start training if enough samples are available
    if replay_buffer.size() >= BATCH_SIZE:
        agent.update_parameters(replay_buffer)

    # Save agent and replay buffer periodically
    if step % 10000 == 0:  # Save every 10,000 steps
        save_agent(agent, agent_checkpoint_file)
        save_replay_buffer(replay_buffer, replay_buffer_file)

    # Reset environment if episode is done
    if done:
        obs, info = env.reset()
    else:
        obs = obs_next  # Update observation for next step
