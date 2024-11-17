import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from tmrl import get_environment  # Ensure this function is correctly defined in the tmrl module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration using dataclass
@dataclass
class PPOConfig:
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    state_dim: int = 85
    action_dim: int = 3
    batch_size: int = 100
    actor_lr: float = 0.001
    critic_lr: float = 0.0005
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

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.xavier_uniform_(self.log_std.weight)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, state: torch.Tensor):
        x = self.network(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor):
        return self.network(state)

# PPO Agent
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
        """Preprocess environment observations for compatibility."""
        try:
            if isinstance(obs, tuple) and len(obs) == 4:
                # Flatten and concatenate all parts of the observation
                concatenated = np.concatenate([np.asarray(part).flatten() for part in obs])
                # Pad with zeros if necessary
                padded = np.pad(
                    concatenated,
                    (0, max(0, self.config.state_dim - len(concatenated))),
                    'constant'
                )[:self.config.state_dim]
                return padded.astype(np.float32)
            else:
                # Handle unexpected formats by returning a zero array
                logger.warning("Unexpected observation format. Setting to zeros.")
                return np.zeros(self.config.state_dim, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error in preprocess_obs: {e}")
            return np.zeros(self.config.state_dim, dtype=np.float32)

    def update_obs_stats(self, obs: np.ndarray):
        """Update running statistics for observation normalization."""
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += delta * delta2

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations."""
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var / self.obs_count) + 1e-8)

    def select_action(self, state: Any) -> (np.ndarray, np.ndarray):
        """Select action based on current policy."""
        obs = self.preprocess_obs(state)
        self.update_obs_stats(obs)
        norm_obs = self.normalize_obs(obs)
        state_tensor = torch.FloatTensor(norm_obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, std = self.actor(state_tensor)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        action_tanh = torch.tanh(action).clamp(-0.999999, 0.999999)  # Prevent exact -1 or 1

        return action_tanh.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def store_transition(self, state: Any, action: np.ndarray, log_prob: np.ndarray, reward: float, done: bool):
        """Store transition in memory."""
        obs = self.preprocess_obs(state)
        self.update_obs_stats(obs)
        norm_obs = self.normalize_obs(obs)
        self.memory['states'].append(norm_obs)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)

    def compute_returns_and_advantages(self, rewards: List[float], dones: List[bool], values: List[float], next_value: float) -> (List[float], List[float]):
        """Compute returns and Generalized Advantage Estimation (GAE)."""
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

        # Convert memory to NumPy arrays for efficient tensor creation
        states = np.array(self.memory['states'], dtype=np.float32)
        actions = np.array(self.memory['actions'], dtype=np.float32)
        old_log_probs = np.array(self.memory['log_probs'], dtype=np.float32)
        rewards = self.memory['rewards']
        dones = self.memory['dones']

        # Compute value estimates
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            values = self.critic(states_tensor).squeeze().cpu().numpy()
            # Compute next value
            next_state = self.memory['states'][-1]
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_value = self.critic(next_state_tensor).item()
            values = list(values) + [next_value]

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(rewards, dones, values, next_value)

        # Normalize advantages
        advantages = np.array(advantages, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(old_log_probs).to(self.device),
            returns,
            advantages
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # PPO updates
        for epoch in range(self.k_epochs):
            for batch in loader:
                b_states, b_actions, b_old_log_probs, b_returns, b_advantages = batch

                # Forward pass
                mean, std = self.actor(b_states)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                # PPO objective
                ratios = torch.exp(log_probs - b_old_log_probs.squeeze())
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_factor * entropy.mean()
                critic_loss = F.mse_loss(self.critic(b_states).squeeze(), b_returns)

                # Backward pass and optimization
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        # Clear memory after update
        self.reset_memory()
        logger.debug("Agent updated successfully.")

    def reset_memory(self):
        """Clear memory buffers."""
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

# PPO Trainer
class PPOTrainer:
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = config.device
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        self.env = get_environment()
        self.agent = PPOAgent(config)

        self.total_rewards: List[float] = []
        self.best_reward: float = -float('inf')

        self.episode_numbers: List[int] = []
        self.avg_rewards: List[float] = []

    def plot_and_save_rewards(self, episode: int, avg_reward: float):
        """Plot and save average rewards."""
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
        """Run the training loop."""
        logger.info("Starting PPO training...")
        for episode in range(1, self.config.num_episodes + 1):
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.config.max_steps):
                # Select action
                action, log_prob = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store transition in memory
                self.agent.store_transition(state, action, log_prob, reward, done)
                state = next_state
                episode_reward += reward

                if done:
                    break

            # Update policy if memory size is exceeded
            if len(self.agent.memory['states']) >= self.config.memory_size:
                logger.info(f"Updating policy at episode {episode}")
                self.agent.update()

            # Track reward
            self.total_rewards.append(episode_reward)
            avg_reward = np.mean(self.total_rewards[-100:])  # Last 100 episodes
            self.writer.add_scalar("Reward/Total", episode_reward, episode)
            self.writer.add_scalar("Reward/Average (last 100)", avg_reward, episode)

            # Per-episode logging
            logger.info(
                f"Episode {episode}/{self.config.num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Average Reward (last 100): {avg_reward:.2f}"
            )

            # Plot and save rewards at log intervals
            if episode % self.config.log_interval == 0:
                self.plot_and_save_rewards(episode, avg_reward)

                # Save the best model
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    best_model_path = os.path.join(self.config.model_dir, 'best_model.pth')
                    self.agent.save(best_model_path)
                    logger.info(f"New best model saved to {best_model_path}")

            # Save model checkpoints at save intervals
            if episode % self.config.save_interval == 0:
                model_path = os.path.join(self.config.model_dir, f"ppo_episode_{episode}.pt")
                self.agent.save(model_path)
                logger.info(f"Model checkpoint saved to {model_path}")

        # Final plotting if needed
        if self.config.num_episodes % self.config.log_interval != 0:
            last_avg_reward = np.mean(self.total_rewards[-(self.config.num_episodes % self.config.log_interval):])
            self.plot_and_save_rewards(self.config.num_episodes, last_avg_reward)

        self.writer.close()
        self.env.close()
        logger.info("PPO training completed.")

# Training function
def train_ppo():
    """Initialize and run the PPO trainer."""
    config = PPOConfig()
    trainer = PPOTrainer(config)
    trainer.run()

if __name__ == "__main__":
    train_ppo()
