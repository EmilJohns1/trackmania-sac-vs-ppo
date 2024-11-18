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
    state_dim: int = 83
    action_dim: int = 3
    batch_size: int = 100
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    eps_clip: float = 0.2
    k_epochs: int = 10
    lam: float = 0.95
    entropy_factor: float = 0.05
    memory_size: int = 20000
    num_episodes: int = 1000
    max_steps: int = 1000
    log_interval: int = 10
    save_interval: int = 100
    model_dir: str = "agentsPPO"
    log_dir: str = "graphsPPO"
    avg_ray: float = 400
    critic_coef: float = 0.5
    grad_clip_val: float = 0.1
    norm_advantages: bool = True

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        # Define dimensions
        lidar_dim = 72
        other_dim = state_dim - lidar_dim

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

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
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
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        # Define dimensions based on your observation structure
        lidar_dim = 72
        other_dim = state_dim - lidar_dim

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
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.k_epochs = config.k_epochs
        self.lam = config.lam
        self.entropy_factor = config.entropy_factor

        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)

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

        self.obs_mean = np.zeros(self.state_dim, dtype=np.float32)
        self.obs_var = np.ones(self.state_dim, dtype=np.float32)
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
                    (0, max(0, self.config.state_dim - len(concatenated))),
                    'constant'
                )[:self.config.state_dim]
                return padded.astype(np.float32)
            else:
                logger.warning("Unexpected observation format. Setting to zeros.")
                return np.zeros(self.config.state_dim, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error in preprocess_obs: {e}")
            return np.zeros(self.config.state_dim, dtype=np.float32)

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

        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        self.env = get_environment()
        self.agent = PPOAgent(config)

        self.total_rewards: List[float] = []
        self.best_reward: float = -float('inf')

        self.episode_numbers: List[int] = []
        self.avg_rewards: List[float] = []

    def plot_and_save_rewards(self, episode: int, avg_reward: float):
        self.episode_numbers.append(episode)
        self.avg_rewards.append(avg_reward)

        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_numbers, self.avg_rewards, label='Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward vs Episode')
        plt.legend()
        plt.grid(True)

        plots_dir = os.path.join(self.config.log_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f'average_reward_episode_{episode}.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved reward plot to {plot_path}")

    def run(self):
        logger.info("Starting PPO training...")
        for episode in range(1, self.config.num_episodes + 1):
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.config.max_steps):
                action, log_prob = self.agent.select_action(state)

                clamped_action = np.clip(action, -1, 1)

                next_state, reward, terminated, truncated, _ = self.env.step(clamped_action)
                done = terminated or truncated

                self.agent.store_transition(state, action, log_prob, reward, done)
                state = next_state
                episode_reward += reward

                if done:
                    break

            # Update policy
            if len(self.agent.memory['states']) >= self.config.memory_size:
                logger.info(f"Updating policy at episode {episode}")
                self.agent.update()

            # Logging rewards
            self.total_rewards.append(episode_reward)
            avg_reward = np.mean(self.total_rewards[-100:])

            logger.info(
                f"Episode {episode}/{self.config.num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Average Reward (last 100): {avg_reward:.2f}"
            )

            # Plot and save rewards at intervals
            if episode % self.config.log_interval == 0:
                self.plot_and_save_rewards(episode, avg_reward)

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
            self.plot_and_save_rewards(self.config.num_episodes, last_avg_reward)

        self.env.close()
        logger.info("PPO training completed.")

def train_ppo():
    config = PPOConfig()
    trainer = PPOTrainer(config)
    trainer.run()

if __name__ == "__main__":
    train_ppo()
