from dataclasses import field, dataclass

import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

from tmrl import get_environment

@dataclass
class SACConfig:
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    MAX_STEPS: int = 10000000
    STEP_DELAY: float = 0.01
    OBSERVATION_SPACE: int = 85
    ACTION_SPACE: int = 3
    ALPHA: float = 0.1
    BUFFER_SIZE: int = int(5e5)
    BATCH_SIZE: int = 256
    GAMMA: int = 0.995
    TAU: int = 0.005

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

        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic1 = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.critic2 = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.target_critic1 = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.target_critic2 = Critic(self.obs_dim, self.act_dim).to(self.device)

        self.target_entropy = -self.act_dim * 0.5
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = self.log_alpha.exp()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

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
        obs, action, reward, next_obs, done = replay_buffer.sample()
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


class ReplayBuffer:
    def __init__(self, config: SACConfig):
        self.device = config.DEVICE
        self.buffer = deque(maxlen=config.BUFFER_SIZE)
        self.batch_size = config.BATCH_SIZE

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
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


class SACTrainer:
    def __init__(self, config: SACConfig):
        self.device = config.DEVICE
        self.max_steps = config.MAX_STEPS
        self.step_delay = config.STEP_DELAY
        self.batch_size = config.BATCH_SIZE

        self.env = get_environment()
        self.config = SACConfig()
        self.agent = SACAgent(config)
        self.replay_buffer = ReplayBuffer(config)

    def run(self):
        cumulative_reward = 0
        lap_time = 0
        episode_start_time = time.time()

        cumulative_rewards = []
        lap_times = []
        steps_record = []

        obs, info = self.env.reset()

        for step in range(self.max_steps):
            time.sleep(self.step_delay)

            processed_obs = self.agent.preprocess_obs(obs)
            act, log_prob = self.agent.sample_action(
                torch.tensor(processed_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            action = act.cpu().detach().numpy().flatten()

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

            if step % 10000 == 0 and step != 0:
                # Save agent and replay buffer periodically
                pass


def train_sac():
    config = SACConfig()
    trainer = SACTrainer(config)
    trainer.run()

if __name__ == "__main__":
    train_sac()