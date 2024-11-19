import os
import pickle
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tmrl import get_environment


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
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    OBSERVATION_SPACE: int = 83
    ACTION_SPACE: int = 3
    BATCH_SIZE: int = 512
    ACTOR_LR: float = 0.00003
    CRITIC_LR: float = 0.00005
    GAMMA: float = 0.99
    EPS_CLIP: float = 0.2
    K_EPOCHS: int = 10
    LAM: float = 0.95
    ENTROPY_FACTOR: float = 0.01
    MEMORY_SIZE: int = 20000
    NUM_EPISODES: int = 10000
    MAX_STEPS: int = 2500
    LOG_INTERVAL: int = 10
    SAVE_INTERVAL: int = 100
    CRITIC_COEF: float = 0.5
    GRAD_CLIP_VAL: float = 0.1
    NORM_ADVANTAGES: bool = True


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super(Actor, self).__init__()
        lidar_dim = 72
        other_dim = obs_dim - lidar_dim

        self.lidar_net = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.other_net = nn.Sequential(
            nn.Linear(other_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)
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

    def forward(self, state):
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
    def __init__(self, obs_dim, hidden_dim=256):
        super(Critic, self).__init__()
        lidar_dim = 72
        other_dim = obs_dim - lidar_dim

        self.lidar_net = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.other_net = nn.Sequential(
            nn.Linear(other_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01)
        )

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

    def forward(self, state):
        lidar_dim = 72
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
        self.device = config.DEVICE
        self.observation_space = config.OBSERVATION_SPACE
        self.action_space = config.ACTION_SPACE
        self.batch_size = config.BATCH_SIZE
        self.gamma = config.GAMMA
        self.eps_clip = config.EPS_CLIP
        self.k_epochs = config.K_EPOCHS
        self.lam = config.LAM
        self.entropy_factor = config.ENTROPY_FACTOR

        self.actor = Actor(self.observation_space, self.action_space).to(self.device)
        self.critic = Critic(self.observation_space).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

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

    def preprocess_obs(self, obs: Any):
        try:
            if len(obs) == 2:
                obs = obs[0]
            if isinstance(obs, tuple) and len(obs) == 4:
                concatenated = np.concatenate([np.asarray(part).flatten() for part in obs])
                padded = np.pad(
                    concatenated,
                    (0, max(0, self.config.OBSERVATION_SPACE - len(concatenated))),
                    'constant'
                )[:self.config.OBSERVATION_SPACE]
                return padded.astype(np.float32)
            else:
                logger.warning("Unexpected observation format. Setting to zeros.")
                return np.zeros(self.config.OBSERVATION_SPACE, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error in preprocess_obs: {e}")
            return np.zeros(self.config.OBSERVATION_SPACE, dtype=np.float32)

    def update_obs_stats(self, obs):
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += delta * delta2

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var / self.obs_count) + 1e-8)

    def select_action(self, state):
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

    def store_transition(self, state, action, log_prob, reward, done):
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

    def compute_returns_and_advantages(self, rewards, dones, values, next_value):
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
            last_state = self.memory['states'][-1]
            last_state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            next_value = self.critic(last_state_tensor).cpu().item()
            values = list(values) + [next_value]

        returns, advantages = self.compute_returns_and_advantages(rewards, dones, values, next_value)

        advantages = np.array(advantages, dtype=np.float32)
        if self.config.NORM_ADVANTAGES:
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
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.GRAD_CLIP_VAL)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.GRAD_CLIP_VAL)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                total_losses.append((actor_loss + self.config.CRITIC_COEF * critic_loss).item())

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

    def save(self, path="agents/ppo/saved_agent.pth"):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
            }, path)
            logger.info(f"PPO agent saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save PPO agent to {path}: {e}")

    def load(self, path="agents/ppo/saved_agent.pth"):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            logger.info(f"PPO agent loaded from {path}")
        except FileNotFoundError:
            logger.warning(f"No saved agent found at {path}. Starting with random weights.")
        except Exception as e:
            logger.error(f"Failed to load PPO agent from {path}: {e}")


class PPOTrainer:
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = config.DEVICE

        self.env = get_environment()
        self.agent = PPOAgent(config)

        self.total_rewards: List[float] = []
        self.best_reward: float = -float('inf')

        self.episode_numbers: List[int] = []
        self.avg_rewards: List[float] = []

        self.lap_times: List[float] = []
        self.steps_record: List[int] = []
        self.cumulative_steps: int = 0

        try:
            loaded_rewards, loaded_lap_times, loaded_steps_record, self.last_step = self.load_graph_data()
            self.total_rewards = loaded_rewards
            self.lap_times = loaded_lap_times
            self.steps_record = loaded_steps_record
            self.cumulative_steps = self.last_step

            min_length = min(len(self.total_rewards), len(self.lap_times), len(self.steps_record))
            if not (len(self.total_rewards) == len(self.lap_times) == len(self.steps_record)):
                logger.warning(
                    f"Truncating lists to minimum length {min_length} to ensure consistency."
                )
                self.total_rewards = self.total_rewards[:min_length]
                self.lap_times = self.lap_times[:min_length]
                self.steps_record = self.steps_record[:min_length]

            self.agent.load()
            logger.info("Successfully loaded graph data and agent.")
        except FileNotFoundError:
            logger.info("No saved agent or graph data found, starting fresh.")
        except Exception as e:
            logger.error(f"Error loading saved agent or graph data: {e}")
            logger.info("Starting with a fresh agent and graph data.")

    def plot_and_save_graphs(self, steps, cumulative_rewards, lap_times, steps_record,
                             filename_prefix="graphs/ppo/performance"):
        try:
            os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)

            if not (len(steps_record) == len(cumulative_rewards) == len(lap_times)):
                logger.error(
                    f"Data length mismatch: steps_record({len(steps_record)}), "
                    f"cumulative_rewards({len(cumulative_rewards)}), "
                    f"lap_times({len(lap_times)})"
                )
                return

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.plot(steps_record, cumulative_rewards, label="Cumulative Reward")
            plt.xlabel("Steps")
            plt.ylabel("Cumulative Reward")
            plt.title("Cumulative Reward vs Steps")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(steps_record, lap_times, label="Lap Time")
            plt.xlabel("Steps")
            plt.ylabel("Lap Time (s)")
            plt.title("Lap Time vs Steps")
            plt.legend()

            plt.tight_layout()
            save_path = f"{filename_prefix}_step_{steps}.png"
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved training metrics plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to plot and save graphs: {e}", exc_info=True)

    def save_graph_data(self, cumulative_rewards, lap_times, steps_record, last_step,
                        filename="graphs/ppo/graph_data.pkl"):
        try:
            min_length = min(len(cumulative_rewards), len(lap_times), len(steps_record))
            if not (len(cumulative_rewards) == len(lap_times) == len(steps_record)):
                logger.warning(
                    f"Truncating lists to minimum length {min_length} before saving."
                )
                cumulative_rewards = cumulative_rewards[:min_length]
                lap_times = lap_times[:min_length]
                steps_record = steps_record[:min_length]

            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, 'wb') as f:
                pickle.dump({
                    'cumulative_rewards': cumulative_rewards,
                    'lap_times': lap_times,
                    'steps_record': steps_record,
                    'last_step': last_step
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Graph data saved to {filename}.")
        except OSError as e:
            logger.error(f"Failed to save graph data: {e}")

    def load_graph_data(self, filename="graphs/ppo/graph_data.pkl"):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Graph data loaded from {filename}.")
            return (
                data.get('cumulative_rewards', []),
                data.get('lap_times', []),
                data.get('steps_record', []),
                data.get('last_step', 0)
            )
        except FileNotFoundError:
            logger.warning(f"No graph data found at {filename}, starting fresh.")
            return [], [], [], 0

    def run(self):
        logger.info("Starting PPO training...")
        for episode in range(1, self.config.NUM_EPISODES + 1):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            lap_time = 0

            for step in range(self.config.MAX_STEPS):
                action, log_prob = self.agent.select_action(state)

                clamped_action = np.clip(action, -1, 1)

                next_state, reward, terminated, truncated, info = self.env.step(clamped_action)
                done = terminated or truncated

                if 'lap_time' in info:
                    lap_time = info['lap_time']
                else:
                    lap_time = 0.0

                self.agent.store_transition(state, action, log_prob, reward, done)
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.cumulative_steps += 1

                if done:
                    break

            if len(self.agent.memory['states']) >= self.config.MEMORY_SIZE:
                logger.info(f"Updating policy at episode {episode}")
                self.agent.update()

            self.total_rewards.append(episode_reward)
            avg_reward = np.mean(self.total_rewards[-100:])

            self.episode_numbers.append(episode)
            self.avg_rewards.append(avg_reward)

            self.lap_times.append(lap_time)
            self.steps_record.append(self.cumulative_steps)

            logger.info(
                f"Episode {episode}/{self.config.NUM_EPISODES} | "
                f"Reward: {episode_reward:.2f} | "
                f"Average Reward (last 100): {avg_reward:.2f} | "
                f"Lap Time: {lap_time:.2f} | Steps: {episode_steps}"
            )

            logger.debug(
                f"After Episode {episode}: "
                f"total_rewards({len(self.total_rewards)}), "
                f"lap_times({len(self.lap_times)}), "
                f"steps_record({len(self.steps_record)})"
            )

            if episode % self.config.SAVE_INTERVAL == 0 and episode_steps != 0:
                self.agent.save()
                self.save_graph_data(self.total_rewards.copy(), self.lap_times.copy(), self.steps_record.copy(),
                                     self.cumulative_steps)
                steps = self.cumulative_steps
                self.plot_and_save_graphs(steps, self.total_rewards.copy(), self.lap_times.copy(),
                                          self.steps_record.copy())

        self.env.close()
        logger.info("PPO training completed.")


def train_ppo():
    config = PPOConfig()
    trainer = PPOTrainer(config)
    trainer.run()

if __name__ == "__main__":
    train_ppo()