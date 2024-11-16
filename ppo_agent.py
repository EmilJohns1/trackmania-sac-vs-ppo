import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from torch.distributions import Normal

from tmrl import get_environment

# Actor is for continous actions instead of discrete like before
class Actor(nn.Module):
    def __init__(self, config, hidden_sizes=[64, 64]):
        super(Actor, self).__init__()
        layers = []
        input_dim = config["STATE_DIM"]
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        self.hidden = nn.Sequential(*layers)

        # Output layers for mean and log_std
        self.mean_head = nn.Linear(input_dim, config["ACTION_DIM"])
        self.log_std_head = nn.Linear(input_dim, config["ACTION_DIM"])

        # Initialize log_std parameters to small values to prevent too large initial std
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        hidden = self.hidden(state)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def select_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return action.detach().cpu().numpy(), log_prob.detach(), entropy.detach()


class Critic(nn.Module):
    def __init__(self, config, hidden_sizes=[64, 64]):
        super(Critic, self).__init__()
        layers = []
        input_dim = config["STATE_DIM"]
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.value = nn.Linear(input_dim, 1)

    def forward(self, state):
        hidden = self.hidden(state)
        value = self.value(hidden)
        return value

class PPOAgent:
    def __init__(self, config):
        super().__init__()
        self.device = config["DEVICE"]
        self.gamma = config["GAMMA"]
        self.eps_clip = config["EPS_CLIP"]
        self.k_epochs = config["K_EPOCHS"]
        self.lam = config["LAM"]
        self.entropy_factor = config["ENTROPY_FACTOR"]
        self.batch_size = config["BATCH_SIZE"]

        state_dim = config["STATE_DIM"]
        action_dim = config["ACTION_DIM"]

        self.actor = Actor(state_dim, action_dim, hidden_sizes=[64, 64]).to(self.device)
        self.critic = Critic(state_dim, hidden_sizes=[64, 64]).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=config["ACTOR_LR"])
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=config["CRITIC_LR"])

        self.memory = deque(maxlen=config["MEMORY_SIZE"])

    #Computes generalized advantage estimation
    def compute_gae(self, rewards, dones, values, next_values):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    # Update the actor and critic
    def update(self):
        # Not enough samples to update!
        if len(self.memory) < self.batch_size:
            return

        # Convert memory to lists
        states = torch.FloatTensor([m['state'] for m in self.memory]).to(self.device)
        actions = torch.FloatTensor([m['action'] for m in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([m['log_prob'] for m in self.memory]).to(self.device)
        rewards = [m['reward'] for m in self.memory]
        dones = [m['done'] for m in self.memory]

        # Compute value estimates
        with torch.no_grad():
            values = self.critic(states).squeeze().cpu().numpy()
            next_states = torch.FloatTensor([m['next_state'] for m in self.memory]).to(self.device)
            next_values = self.critic(next_states).squeeze().cpu().numpy()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, dones, values, next_values)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 0.00000001)

        # Shuffle the indices
        indices = np.arange(len(states))
        np.random.shuffle(indices)

        for _ in range(self.k_epochs):
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                minibatch_indices = indices[start:end]

                minibatch_states = states[minibatch_indices]
                minibatch_actions = actions[minibatch_indices]
                minibatch_old_log_probs = old_log_probs[minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]
                minibatch_returns = returns[minibatch_indices]

                # Get current policy outputs
                mean, std = self.actor(minibatch_states)
                dist = Normal(mean, std)
                z = dist.rsample()
                actions_tanh = torch.tanh(z)
                # Calculate log_prob with tanh correction
                log_prob = dist.log_prob(z) - torch.log(1 - actions_tanh.pow(2) + 0.0000001)
                log_prob = log_prob.sum(dim=-1)

                # Calculate entropy
                entropy = dist.entropy().sum(dim=-1)

                # Calculate ratio for clipping
                ratios = torch.exp(log_prob - minibatch_old_log_probs)

                # Surrogate loss
                surr1 = ratios * minibatch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * minibatch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_factor * entropy.mean()

                # Update Actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                # Critic loss
                value = self.critic(minibatch_states).squeeze()
                critic_loss = nn.MSELoss()(value, minibatch_returns)

                # Update Critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        # Clear memory after update
        self.memory.clear()


class StateNormalizer:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.mean = np.zeros(state_dim)
        self.var = np.ones(state_dim)
        self.count = 0

    def update(self, states):
        batch_mean = np.mean(states, axis=0)
        batch_var = np.var(states, axis=0)
        batch_count = len(states)

        # Update running mean and variance using Welford's method
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        new_var = (self.var * self.count + batch_var * batch_count +
                   delta ** 2 * self.count * batch_count / tot_count) / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, state):
        return (state - self.mean) / (np.sqrt(self.var) + 1e-8)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'var': self.var,
                'count': self.count
            }, f)
        print(f"StateNormalizer saved to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.mean = data['mean']
            self.var = data['var']
            self.count = data['count']
        print(f"StateNormalizer loaded from {path}")


def calculate_centerline_distance(obs):
    lidar = np.asarray(obs[1]).reshape(-1)
    return lidar[0] - lidar[18]


class PPOTrainer:
    def __init__(self, config):
        self.device = config["DEVICE"]
        self.num_episodes = config["NUM_EPISODES"]
        self.max_steps = config["MAX_STEPS"]
        self.log_interval = config["LOG_INTERVAL"]
        self.save_interval = config["SAVE_INTERVAL"]
        self.model_dir = config["MODEL_DIR"]
        self.log_dir = config["LOG_DIR"]

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.env = get_environment()
        self.agent = PPOAgent(config)

        self.normalizer = StateNormalizer(config["STATE_DIM"])

        self.total_rewards = []
        self.best_reward = -float('inf')

    def run(self):
        for episode in range(1, self.num_episodes + 1):
            state, info = self.env.reset()
            state = self.normalizer.normalize(state)
            episode_reward = 0
            done = False
            truncated = False
            states_batch = []

            for step in range(self.max_steps):
                action, log_prob, entropy = self.agent.actor.select_action(state)
                action = np.clip(action, -1.0, 1.0)

                next_state, reward, done, truncated, info = self.env.step(action)

                next_state_normalized = self.normalizer.normalize(next_state)

                self.agent.store_transition({
                    'state': state,
                    'action': action,
                    'log_prob': log_prob.item(),
                    'reward': reward,
                    'next_state': next_state_normalized,
                    'done': done or truncated,
                    'entropy': entropy.item(),
                })

                states_batch.append(state)
                state = next_state_normalized
                episode_reward += reward

                if done or truncated:
                    break

            self.normalizer.update(states_batch)
            self.agent.update()
            self.total_rewards.append(episode_reward)

        self.writer.close()
        self.env.close()


def train_ppo():
    config = {
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "STATE_DIM": 85,
        "ACTION_DIM": 3,
        "BATCH_SIZE": 200,
        "ACTOR_LR": 0.001,
        "CRITIC_LR": 0.0005,
        "GAMMA": 0.99,
        "EPS_CLIP": 0.2,
        "K_EPOCHS": 10,
        "LAM": 0.95,
        "ENTROPY_FACTOR": 0.02,
        "MEMORY_SIZE": 20000,

        "NUM_EPISODES": 1000,
        "MAX_STEPS": 1000,
        "LOG_INTERVAL": 10,
        "SAVE_INTERVAL": 100,

        "MODEL_DIR": "agentsPPO",
        "LOG_DIR": "graphsPPO",
    }

    trainer = PPOTrainer(config)
    trainer.run()


if __name__ == "__main__":
    train_ppo()