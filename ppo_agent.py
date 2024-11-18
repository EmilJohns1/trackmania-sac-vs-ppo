import os
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tmrl import get_environment
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PPO_AGENT")

@dataclass
class PPOConfig:
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    observation_space: int = 83
    action_space: int = 3
    batch_size: int = 256
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    eps_clip: float = 0.2
    k_epochs: int = 10
    lam: float = 0.95
    entropy_factor: float = 0.05
    memory_size: int = 20000
    num_episodes: int = 1000
    max_steps: int = 2500
    log_interval: int = 10
    save_interval: int = 100
    model_dir: str = "agentsPPO"
    log_dir: str = "graphsPPO"
    avg_ray: float = 400
    critic_coef: float = 0.5
    grad_clip_val: float = 0.1
    norm_advantages: bool = True

class Actor(nn.Module):
    def __init__(self, observation_space: int, action_space: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        # Define dimensions
        lidar_dim = 72
        other_dim = observation_space - lidar_dim

        # Lidar Subnetwork
        self.lidar_net = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Other Sensors Subnetwork
        self.other_net = nn.Sequential(
            nn.Linear(other_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Combined Network
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.mean = nn.Linear(hidden_dim, action_space)
        self.log_std = nn.Linear(hidden_dim, action_space)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.lidar_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for layer in self.other_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for layer in self.combined_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.xavier_uniform_(self.log_std.weight)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, state: torch.Tensor):
        lidar_dim = 72
        lidar = state[:, :lidar_dim]
        other = state[:, lidar_dim:]
        lidar_features = self.lidar_net(lidar)
        other_features = self.other_net(other)
        combined = torch.cat((lidar_features, other_features), dim=1)
        x = self.combined_net(combined)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, observation_space: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        # Define dimensions based on your observation structure
        lidar_dim = 72
        other_dim = observation_space - lidar_dim

        # Lidar Subnetwork
        self.lidar_net = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Other Sensors Subnetwork
        self.other_net = nn.Sequential(
            nn.Linear(other_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Combined Network
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.value_head = nn.Linear(hidden_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.lidar_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for layer in self.other_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for layer in self.combined_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, state: torch.Tensor):
        lidar_dim = 72  # Adjust based on your observation structure
        lidar = state[:, :lidar_dim]
        other = state[:, lidar_dim:]
        lidar_features = self.lidar_net(lidar)
        other_features = self.other_net(other)
        combined = torch.cat((lidar_features, other_features), dim=1)
        x = self.combined_net(combined)
        value = self.value_head(x)
        return value

class PPOAgent:
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = config.device
        self.observation_space = config.observation_space
        self.action_space = config.action_space
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.k_epochs = config.k_epochs
        self.lam = config.lam
        self.entropy_factor = config.entropy_factor

        self.actor = Actor(self.observation_space, self.action_space).to(self.device)
        self.critic = Critic(self.observation_space).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.memory: Dict[str, List[Any]] = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }

        self.obs_mean = np.zeros(self.observation_space, dtype=np.float32)
        self.obs_var = np.ones(self.observation_space, dtype=np.float32)
        self.obs_count = 1.0

    def preprocess_obs(self, obs: Any) -> np.ndarray:
        try:
            # Handle different observation formats
            if len(obs) == 2:
                obs = obs[0]
            if isinstance(obs, tuple) and len(obs) == 4:
                concatenated = np.concatenate([np.asarray(part).flatten() for part in obs])
                padded = np.pad(
                    concatenated,
                    (0, max(0, self.config.observation_space - len(concatenated))),
                    'constant'
                )[:self.config.observation_space]
                return padded.astype(np.float32)
            else:
                logger.warning("Unexpected observation format. Setting to zeros.")
                return np.zeros(self.config.observation_space, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error in preprocess_obs: {e}")
            return np.zeros(self.config.observation_space, dtype=np.float32)

    def update_obs_stats(self, obs: np.ndarray):
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += delta * delta2

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var / self.obs_count) + 1e-8)

    def select_action(self, state: Any) -> (np.ndarray, float):
        obs = self.preprocess_obs(state)
        self.update_obs_stats(obs)
        norm_obs = self.normalize_obs(obs)
        state_tensor = torch.FloatTensor(norm_obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, std = self.actor(state_tensor)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1).item()
        action_tanh = torch.tanh(action).clamp(-0.999999, 0.999999)

        return action_tanh.cpu().numpy()[0], log_prob

    def store_transition(self, state: Any, action: np.ndarray, log_prob: float, reward: float, done: bool):
        obs = self.preprocess_obs(state)
        self.update_obs_stats(obs)
        norm_obs = self.normalize_obs(obs)
        self.memory['states'].append(norm_obs)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(norm_obs).unsqueeze(0).to(self.device)
            value = self.critic(state_tensor).cpu().item()
            self.memory['values'].append(value)

        logger.debug(f"Stored Transition | Reward: {reward} | Done: {done}")

    def compute_returns_and_advantages(self, rewards: List[float], dones: List[bool], values: List[float], next_value: float) -> (List[float], List[float]):
        returns, advantages = [], []
        gae = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * (values[step + 1] if step + 1 < len(values) else next_value) * mask - values[step]
            gae = delta + self.gamma * self.lam * mask * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
        return returns, advantages

    def update(self):
        """Update policy and value networks."""
        if not self.memory['states']:
            logger.warning("No transitions to update.")
            return

        states = np.array(self.memory['states'], dtype=np.float32)
        actions = np.array(self.memory['actions'], dtype=np.float32)
        old_log_probs = np.array(self.memory['log_probs'], dtype=np.float32)
        rewards = self.memory['rewards']
        dones = self.memory['dones']

        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            values = self.critic(states_tensor).squeeze().cpu().numpy()
            # Get value of the last state
            last_state = self.memory['states'][-1]
            last_state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            next_value = self.critic(last_state_tensor).cpu().item()
            values = list(values) + [next_value]

        returns, advantages = self.compute_returns_and_advantages(rewards, dones, values, next_value)

        advantages = np.array(advantages, dtype=np.float32)
        if self.config.norm_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        dataset = TensorDataset(
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(old_log_probs).to(self.device),
            returns,
            advantages
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        actor_losses = []
        critic_losses = []
        total_losses = []

        for epoch in range(self.k_epochs):
            for batch in loader:
                b_states, b_actions, b_old_log_probs, b_returns, b_advantages = batch

                mean, std = self.actor(b_states)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                ratios = torch.exp(log_probs - b_old_log_probs.squeeze())
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_factor * entropy.mean()
                critic_loss = F.mse_loss(self.critic(b_states).squeeze(), b_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_val)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_val)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                total_losses.append((actor_loss + self.config.critic_coef * critic_loss).item())

        # Reset memory after update
        self.reset_memory()

        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)
        avg_total_loss = np.mean(total_losses)

        logger.info(f"PPO Update | Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f} | Total Loss: {avg_total_loss:.4f}")

    def reset_memory(self):
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }

    def save(self, path: str):
        """Save model checkpoints."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        logger.info(f"Model loaded from {path}")

class PPOTrainer:
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = config.device

        # Create directories for models and logs
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.log_dir, 'plots'), exist_ok=True)

        # Initialize environment and agent
        self.env = get_environment()
        self.agent = PPOAgent(config)

        # Initialize metrics
        self.total_rewards: List[float] = []
        self.best_reward: float = -float('inf')

        self.episode_numbers: List[int] = []
        self.avg_rewards: List[float] = []

        # New metrics for lap times and steps
        self.lap_times: List[float] = []
        self.steps_record: List[int] = []
        self.cumulative_steps: int = 0  # Total steps across episodes

    def plot_and_save_graphs(self, steps: int, cumulative_rewards: List[float], lap_times: List[float], steps_record: List[int], filename_prefix: str = "graphsPPO/performance"):
        plt.figure(figsize=(12, 6))

        # Subplot 1: Cumulative Reward vs Steps
        plt.subplot(1, 2, 1)
        plt.plot(steps_record, cumulative_rewards, label="Cumulative Reward", color='blue')
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward vs Steps")
        plt.legend()
        plt.grid(True)

        # Subplot 2: Lap Time vs Steps
        plt.subplot(1, 2, 2)
        plt.plot(steps_record, lap_times, label="Lap Time", color='orange')
        plt.xlabel("Steps")
        plt.ylabel("Lap Time (s)")
        plt.title("Lap Time vs Steps")
        plt.legend()
        plt.grid(True)

        # Save the plots
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_step_{steps}.png")
        plt.close()
        logger.info(f"Saved training metrics plot to {filename_prefix}_step_{steps}.png")

    def run(self):
        logger.info("Starting PPO training...")
        for episode in range(1, self.config.num_episodes + 1):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            lap_time = 0.0  # Initialize lap_time for the episode

            for step in range(self.config.max_steps):
                action, log_prob = self.agent.select_action(state)

                clamped_action = np.clip(action, -1, 1)

                next_state, reward, terminated, truncated, info = self.env.step(clamped_action)
                done = terminated or truncated

                # Extract lap_time from info if available
                if 'lap_time' in info:
                    lap_time = info['lap_time']
                else:
                    # Handle cases where lap_time is not provided
                    lap_time = 0.0

                # Store transition
                self.agent.store_transition(state, action, log_prob, reward, done)
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.cumulative_steps += 1

                if done:
                    break

            # Update policy if enough transitions are collected
            if len(self.agent.memory['states']) >= self.config.memory_size:
                logger.info(f"Updating policy at episode {episode}")
                self.agent.update()

            # Logging rewards
            self.total_rewards.append(episode_reward)
            avg_reward = np.mean(self.total_rewards[-100:])

            # Append to episode_numbers and avg_rewards
            self.episode_numbers.append(episode)
            self.avg_rewards.append(avg_reward)

            # Append new metrics
            self.lap_times.append(lap_time)
            self.steps_record.append(self.cumulative_steps)

            logger.info(
                f"Episode {episode}/{self.config.num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Average Reward (last 100): {avg_reward:.2f} | "
                f"Lap Time: {lap_time:.2f} | Steps: {episode_steps}"
            )

            # Plot and save rewards at intervals
            if episode % self.config.log_interval == 0:
                self.plot_and_save_graphs(
                    steps=episode,
                    cumulative_rewards=self.total_rewards.copy(),
                    lap_times=self.lap_times.copy(),
                    steps_record=self.steps_record.copy(),
                    filename_prefix=os.path.join(self.config.log_dir, 'plots', 'training_metrics')
                )

                # Save best model if average reward improves
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    best_model_path = os.path.join(self.config.model_dir, 'best_model.pth')
                    self.agent.save(best_model_path)
                    logger.info(f"New best model saved to {best_model_path}")

            # Save model checkpoints at intervals
            if episode % self.config.save_interval == 0:
                model_path = os.path.join(self.config.model_dir, f"ppo_episode_{episode}.pt")
                self.agent.save(model_path)
                logger.info(f"Model checkpoint saved to {model_path}")

        # Final plotting if not captured by log_interval
        if self.config.num_episodes % self.config.log_interval != 0:
            last_avg_reward = np.mean(self.total_rewards[-100:])
            self.episode_numbers.append(self.config.num_episodes)
            self.avg_rewards.append(last_avg_reward)
            self.lap_times.append(lap_time)
            self.steps_record.append(self.cumulative_steps)
            self.plot_and_save_graphs(
                steps=self.config.num_episodes,
                cumulative_rewards=self.total_rewards.copy(),
                lap_times=self.lap_times.copy(),
                steps_record=self.steps_record.copy(),
                filename_prefix=os.path.join(self.config.log_dir, 'plots', 'training_metrics')
            )

        self.env.close()
        logger.info("PPO training completed.")

def train_ppo():
    config = PPOConfig()
    trainer = PPOTrainer(config)
    trainer.run()

if __name__ == "__main__":
    train_ppo()
