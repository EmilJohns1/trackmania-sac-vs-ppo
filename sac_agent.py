import os
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from tmrl import get_environment


@dataclass
class SACConfig:
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    MAX_STEPS: int = 10000000
    OBSERVATION_SPACE: int = 85
    ACTION_SPACE: int = 3
    ALPHA: float = 0.01
    BUFFER_SIZE: int = int(5e5)
    BATCH_SIZE: int = 256
    GAMMA: float = 0.995
    TAU: float = 0.005
    LEARNING_RATE_ACTOR: float = float(1e-5)
    LEARNING_RATE_CRITIC: float = float(5e-5)
    LEARNING_RATE_ENTROPY: float = float(3e-4)
    SAVE_INTERVAL: int = 10000

def calculate_centerline_distance(obs):
    lidar = np.asarray(obs[1]).reshape(-1)
    return lidar[0] - lidar[18]


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

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
    def __init__(self, config: SACConfig):
        self.config = config
        self.device = config.DEVICE
        self.obs_dim = config.OBSERVATION_SPACE
        self.act_dim = config.ACTION_SPACE
        self.alpha = config.ALPHA
        self.learning_rate_actor = config.LEARNING_RATE_ACTOR
        self.learning_rate_critic = config.LEARNING_RATE_CRITIC
        self.learning_rate_entropy = config.LEARNING_RATE_ENTROPY

        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic1 = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.critic2 = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.target_critic1 = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.target_critic2 = Critic(self.obs_dim, self.act_dim).to(self.device)

        self.target_entropy = -0.5
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate_entropy)
        self.alpha = self.log_alpha.exp()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.learning_rate_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.learning_rate_critic)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.obs_mean = np.zeros(self.obs_dim)
        self.obs_var = np.ones(self.obs_dim)
        self.obs_count = 1e-5

        self.ewc_lambda = 1000.0
        self.fisher_matrix = {}
        self.prev_params = {}

    def sample_action(self, obs):
        mu, std = self.actor(obs)

        std = std.clamp(min=1e-6)
        dist = torch.distributions.Normal(mu, std)

        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)

        return action, log_prob.sum(1, keepdim=True)

    def update_parameters(self, replay_buffer):
        obs, action, reward, next_obs, done = replay_buffer.sample(replay_buffer.batch_size)
        gamma = self.config.GAMMA
        tau = self.config.TAU

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
        actor_loss += self.ewc_loss()

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

    def compute_fisher_matrix(self, replay_buffer, num_samples=100):
        """
        Computes the Fisher Information Matrix (FIM) using the replay buffer.
        Args:
            replay_buffer: Replay buffer to sample data from.
            num_samples: Number of samples to use for computing FIM.
        """
        self.fisher_matrix = {}
        self.prev_params = {}

        if replay_buffer.size() >= num_samples:
            for i in range(num_samples):
                batch_size = min(replay_buffer.batch_size, replay_buffer.size())

                obs, action, _, _, _ = replay_buffer.sample(batch_size)
                mu, std = self.actor(obs)
                dist = torch.distributions.Normal(mu, std)
                log_probs = dist.log_prob(action).sum(dim=1)

                self.actor_optimizer.zero_grad()
                log_probs.mean().backward(retain_graph=True)

                for name, param in self.actor.named_parameters():
                    if param.grad is not None:
                        if name not in self.fisher_matrix:
                            self.fisher_matrix[name] = param.grad.data.clone().pow(2)
                        else:
                            self.fisher_matrix[name] += param.grad.data.clone().pow(2)

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

    def update_obs_stats(self, obs):
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += delta * delta2

    def preprocess_obs(self, obs):
        if (
            isinstance(obs, tuple)
            and len(obs) > 1
            and isinstance(obs[1], dict)
            and not obs[1]
        ):
            obs_data = obs[0]
        else:
            obs_data = obs

        expected_obs_size = self.obs_dim

        if isinstance(obs_data, tuple) and len(obs_data) == 4:
            speed = np.asarray(obs_data[0]).flatten()
            lidar = np.asarray(obs_data[1]).reshape(-1)
            prev_action_1 = np.asarray(obs_data[2]).flatten()
            prev_action_2 = np.asarray(obs_data[3]).flatten()

            if (
                np.any(np.isnan(speed))
                or np.any(np.isnan(lidar))
                or np.any(np.isnan(prev_action_1))
                or np.any(np.isnan(prev_action_2))
            ):
                print("NaN detected in observation components")

            current_centerline_distance = calculate_centerline_distance(obs)
            next_lidar = np.asarray(
                obs_data[1][1]
            )
            next_centerline_distance = next_lidar[0] - next_lidar[18]

            concatenated_obs = np.concatenate(
                [
                    speed,
                    lidar,
                    prev_action_1,
                    prev_action_2,
                    [current_centerline_distance, next_centerline_distance],
                ]
            )

            if concatenated_obs.shape[0] != expected_obs_size:
                concatenated_obs = np.pad(
                    concatenated_obs,
                    (0, max(0, expected_obs_size - concatenated_obs.shape[0])),
                )[:expected_obs_size]
        else:
            concatenated_obs = np.zeros(expected_obs_size)

        self.update_obs_stats(concatenated_obs)

        normalized_obs = (concatenated_obs - self.obs_mean) / np.sqrt(
            self.obs_var / self.obs_count
        )

        return torch.tensor(normalized_obs, dtype=torch.float32).to(self.device)

    def save(self, path="agents/sac/saved_agent.pth"):
        """Saves the SAC agent (actor, critics, and optimizers)."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha': self.alpha.item(),
            'log_alpha': self.log_alpha.item(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'obs_mean': self.obs_mean,
            'obs_var': self.obs_var,
            'obs_count': self.obs_count,
            'fisher_matrix': self.fisher_matrix,
            'prev_params': self.prev_params
        }, path)
        print(f"SAC agent saved to {path}")

    def load(self, path="agents/sac/saved_agent.pth"):
        """Loads the SAC agent (actor, critics, and optimizers)."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.alpha = torch.tensor(checkpoint['alpha'], device=self.device)
        self.log_alpha = torch.tensor(np.log(checkpoint['alpha']), device=self.device, requires_grad=True)
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.obs_mean = checkpoint.get('obs_mean', self.obs_mean)
        self.obs_var = checkpoint.get('obs_var', self.obs_var)
        self.obs_count = checkpoint.get('obs_count', self.obs_count)
        self.fisher_matrix = checkpoint.get('fisher_matrix', {})
        self.prev_params = checkpoint.get('prev_params', {})
        print(f"SAC agent loaded from {path}")

class ReplayBuffer:
    def __init__(self, config: SACConfig):
        self.device = config.DEVICE
        self.buffer = deque(maxlen=config.BUFFER_SIZE)
        self.batch_size = config.BATCH_SIZE

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if self.batch_size < batch_size:
            batch_size = self.batch_size
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.float32).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.tensor(next_states, dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device),
        )

    def size(self):
        return len(self.buffer)

    def save(self, path="agents/sac/replay_buffer.pkl"):
        """Saves the replay buffer to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Replay buffer saved to {path}")

    def load(self, path="agents/sac/replay_buffer.pkl"):
        """Loads the replay buffer from a file."""
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)
        print(f"Replay buffer loaded from {path}")


class SACTrainer:
    def __init__(self, config: SACConfig):
        self.device = config.DEVICE
        self.max_steps = config.MAX_STEPS
        self.batch_size = config.BATCH_SIZE
        self.save_interval = config.SAVE_INTERVAL

        self.env = get_environment()
        self.agent = SACAgent(config)
        self.replay_buffer = ReplayBuffer(config)

        try:
            self.replay_buffer.load()
            self.cumulative_rewards, self.fastest_lap_times, self.steps_record, self.last_step = self.load_graph_data()
            self.agent.load()
        except FileNotFoundError:
            print("No saved agent or replay buffer found, starting fresh.")

    def apply_penalties(self, obs, action):
        speed = obs[0]

        if speed < 17.5 and action[1] > 0:
            action[1] = -1

        if speed < 22.5 and action[0] < 0.9:
            action[0] = 1.0

        return action

    def plot_and_save_graphs(self, steps, cumulative_rewards, lap_times, steps_record, filename_prefix="graphs/sac/performance"):
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
        plt.savefig(f"{filename_prefix}_step_{steps}.png")
        plt.close()
        print(f"Saved graphs at step {steps}.")

    def save_graph_data(self, cumulative_rewards, lap_times, steps_record, last_step, filename="graphs/sac/graph_data.pkl"):
        """Saves graph data to a file."""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, 'wb') as f:
                pickle.dump({
                    'cumulative_rewards': cumulative_rewards,
                    'lap_times': lap_times,
                    'steps_record': steps_record,
                    'last_step': last_step
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Graph data saved to {filename}.")
        except OSError as e:
            print(f"Failed to save graph data: {e}")

    def load_graph_data(self, filename="graphs/sac/graph_data.pkl"):
        """Loads graph data from a file."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"Graph data loaded from {filename}.")
            return (
                data['cumulative_rewards'],
                data['lap_times'],
                data['steps_record'],
                data.get('last_step', 0)
            )
        except FileNotFoundError:
            print(f"No graph data found at {filename}, starting fresh.")
            return [], [], [], 0

    def run(self):
        cumulative_reward = 0
        lap_time = 0
        episode_start_time = time.time()

        cumulative_rewards = []
        lap_times = []
        steps_record = []

        obs, info = self.env.reset()

        for step in range(self.max_steps):
            processed_obs = self.agent.preprocess_obs(obs)
            act, log_prob = self.agent.sample_action(
                torch.tensor(processed_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            action = act.cpu().detach().numpy().flatten()

            action = self.apply_penalties(obs, action)

            obs_next, reward, terminated, truncated, info = self.env.step(action)
            cumulative_reward += reward

            done = terminated or truncated
            if done:
                episode_time = time.time() - episode_start_time
                episode_start_time = time.time()

                if float(reward) > 25:
                    lap_time = episode_time

                cumulative_rewards.append(cumulative_reward)
                lap_times.append(lap_time)
                steps_record.append(step)

                cumulative_reward = 0

                self.agent.compute_fisher_matrix(self.replay_buffer)

                obs, info = self.env.reset()
            else:
                obs = obs_next

            self.replay_buffer.store(
                processed_obs, action, reward, self.agent.preprocess_obs(obs_next), done
            )

            if self.replay_buffer.size() >= self.batch_size:
                self.agent.update_parameters(self.replay_buffer)

            if step % self.save_interval == 0 and step != 0:
                self.agent.save()
                self.replay_buffer.save()
                self.save_graph_data(cumulative_rewards, lap_times, steps_record, step)
                steps = self.last_step + step
                self.plot_and_save_graphs(steps, cumulative_rewards, lap_times, steps_record)

                time.sleep(5)

def train_sac():
    config = SACConfig()
    trainer = SACTrainer(config)
    trainer.run()

if __name__ == "__main__":
    train_sac()
