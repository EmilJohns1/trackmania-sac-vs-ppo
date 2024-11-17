import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from tmrl import get_environment  # Ensure this function is correctly defined in the tmrl module

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.mean, self.log_std]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.value]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value

# Define the PPO Agent
class PPOAgent:
    def __init__(self, config):
        self.device = config["DEVICE"]
        self.state_dim = config["STATE_DIM"]
        self.action_dim = config["ACTION_DIM"]
        self.batch_size = config["BATCH_SIZE"]
        self.actor_lr = config["ACTOR_LR"]
        self.critic_lr = config["CRITIC_LR"]
        self.gamma = config["GAMMA"]
        self.eps_clip = config["EPS_CLIP"]
        self.k_epochs = config["K_EPOCHS"]
        self.lam = config["LAM"]
        self.entropy_factor = config["ENTROPY_FACTOR"]
        self.memory_size = config["MEMORY_SIZE"]

        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }

        # Initialize observation statistics for normalization
        self.obs_mean = np.zeros(self.state_dim)
        self.obs_var = np.ones(self.state_dim)
        self.obs_count = 1.0  # Initialize obs_count to 1.0 for stable normalization

    def preprocess_obs(self, obs):
        """
        Preprocess the observation to convert it into a flat NumPy array.
        This method does NOT update the running statistics.
        """
        if isinstance(obs, tuple):
            if len(obs) > 1 and isinstance(obs[1], dict) and not obs[1]:
                obs_data = obs[0]
            else:
                obs_data = obs
        else:
            obs_data = obs

        expected_obs_size = self.state_dim

        if isinstance(obs_data, tuple) and len(obs_data) == 4:
            try:
                speed = np.asarray(obs_data[0]).flatten()
                lidar = np.asarray(obs_data[1]).flatten()
                prev_action_1 = np.asarray(obs_data[2]).flatten()
                prev_action_2 = np.asarray(obs_data[3]).flatten()

                # Concatenate all components into a single observation vector
                concatenated_obs = np.concatenate(
                    [
                        speed,
                        lidar,
                        prev_action_1,
                        prev_action_2
                    ]
                )

                # Ensure the observation has the expected size by padding if necessary
                if concatenated_obs.shape[0] != expected_obs_size:
                    concatenated_obs = np.pad(
                        concatenated_obs,
                        (0, max(0, expected_obs_size - len(concatenated_obs))),
                        'constant'
                    )[:expected_obs_size]

            except Exception as e:
                concatenated_obs = np.zeros(expected_obs_size)
        else:
            # Handle the unexpected structure case by returning a zero-filled array
            concatenated_obs = np.zeros(expected_obs_size)

        return concatenated_obs  # Return as NumPy array

    def update_obs_stats(self, obs):
        """
        Update running mean and variance for observation normalization.
        """
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += delta * delta2

    def select_action(self, state):
        # Preprocess the state without updating stats
        concatenated_obs = self.preprocess_obs(state)
        # Normalize using current stats
        normalized_obs = (concatenated_obs - self.obs_mean) / (np.sqrt(self.obs_var / self.obs_count) + 1e-8)
        state_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).to(self.device)  # [1, state_dim]

        with torch.no_grad():
            mean, std = self.actor(state_tensor)

        dist = Normal(mean, std)
        action = dist.sample()
        action_clipped = torch.tanh(action)
        action_clipped = torch.clamp(action_clipped, -0.999999, 0.999999)  # Prevent exact -1 or 1

        log_prob = dist.log_prob(action) - torch.log(1 - action_clipped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action_clipped.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def store_transition(self, state, action, log_prob, reward, done):
        # Preprocess the state and update stats
        concatenated_obs = self.preprocess_obs(state)
        # Update stats with the new observation
        self.update_obs_stats(concatenated_obs)
        # Normalize the observation with updated stats
        normalized_obs = (concatenated_obs - self.obs_mean) / (np.sqrt(self.obs_var / self.obs_count) + 1e-8)
        self.memory['states'].append(normalized_obs)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)

    def compute_returns_and_advantages(self, next_value):
        rewards = self.memory['rewards']
        dones = self.memory['dones']
        values = self.memory['values']

        returns = []
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * (values[step + 1] if step + 1 < len(values) else next_value) * mask - values[step]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        return returns, advantages

    def update(self):
        if len(self.memory['states']) == 0:
            print("No transitions to update.")
            return

        # Convert memory to tensors
        states = torch.FloatTensor(self.memory['states']).to(self.device)  # [N, state_dim]
        actions = torch.FloatTensor(self.memory['actions']).to(self.device)  # [N, action_dim]
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)  # [N, 1]
        dones = torch.FloatTensor(self.memory['dones']).to(self.device)  # [N]

        # Compute value estimates
        with torch.no_grad():
            values = self.critic(states).squeeze().cpu().numpy()  # [N]
            next_state = self.memory['states'][-1]
            next_state_processed = self.preprocess_obs(next_state)
            next_state_tensor = torch.FloatTensor(next_state_processed).unsqueeze(0).to(self.device)  # [1, state_dim]
            next_value = self.critic(next_state_tensor).item()
            self.memory['values'] = list(values) + [next_value]

        returns, advantages = self.compute_returns_and_advantages(next_value)
        returns = torch.FloatTensor(returns).to(self.device)  # [N]
        advantages = torch.FloatTensor(advantages).to(self.device)  # [N]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.k_epochs):
            for batch in loader:
                b_states, b_actions, b_old_log_probs, b_returns, b_advantages = batch

                # Get current log probs and entropy
                mean, std = self.actor(b_states)
                dist = Normal(mean, std)

                # Clamp actions to prevent atanh from receiving exactly -1 or 1
                b_actions_clipped = torch.clamp(b_actions, -0.999999, 0.999999)
                action_samples = torch.atanh(b_actions_clipped)

                # Compute log_probs
                log_probs = dist.log_prob(action_samples) - torch.log(1 - b_actions_clipped.pow(2) + 1e-6)
                log_probs = log_probs.sum(dim=-1, keepdim=True)

                # Compute entropy
                entropy = dist.entropy().sum(dim=-1, keepdim=True)

                # Get state values
                values = self.critic(b_states)

                # Calculate ratios
                ratios = torch.exp(log_probs - b_old_log_probs)

                # Surrogate loss
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_factor * entropy.mean()

                # Value loss
                critic_loss = F.mse_loss(values.squeeze(), b_returns)

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        # Clear memory after update
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }

    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"Model loaded from {path}")

# Define the PPO Trainer
class PPOTrainer:
    def __init__(self, config):
        self.device = config["DEVICE"]
        self.num_episodes = config["NUM_EPISODES"]
        self.max_steps = config["MAX_STEPS"]
        self.log_interval = config["LOG_INTERVAL"]
        self.save_interval = config["SAVE_INTERVAL"]
        self.model_dir = config["MODEL_DIR"]
        self.log_dir = config["LOG_DIR"]

        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Initialize environment and agent
        self.env = get_environment()  # Ensure this function is defined correctly
        self.agent = PPOAgent(config)

        self.total_rewards = []
        self.best_reward = -float('inf')

        # Initialize data structures for plotting
        self.episode_numbers = []
        self.avg_rewards = []
        self.cumulative_rewards = []  # Optional: If you want to plot cumulative rewards per episode

    def log(self, message):
        """
        Logs a formatted message to the console.
        """
        print(f"[PPOTrainer] {message}")

    def plot_and_save_rewards(self, episode, avg_reward):
        """
        Plots and saves the average reward over episodes.
        """
        self.episode_numbers.append(episode)
        self.avg_rewards.append(avg_reward)

        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_numbers, self.avg_rewards, label='Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward vs Episode')
        plt.legend()
        plt.grid(True)
        # Ensure the plots directory exists
        plots_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        # Save the plot
        plt.savefig(os.path.join(plots_dir, f'average_reward_episode_{episode}.png'))
        plt.close()
        self.log(f"Saved reward plot to plots/average_reward_episode_{episode}.png")

    def run(self):
        for episode in range(1, self.num_episodes + 1):
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.max_steps):
                # Select action using the agent
                action, log_prob = self.agent.select_action(state)
                action = np.clip(action, -1.0, 1.0)

                # Take action in the environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store transition in the agent's memory
                self.agent.store_transition(state, action, log_prob, reward, done)

                state = next_state
                episode_reward += reward

                if done:
                    break

            # Update agent after each episode
            self.agent.update()
            self.total_rewards.append(episode_reward)

            # Logging and Plotting
            if episode % self.log_interval == 0:
                avg_reward = np.mean(self.total_rewards[-self.log_interval:])
                self.writer.add_scalar('Average Reward', avg_reward, episode)
                self.log(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")

                # Plot and save the average reward
                self.plot_and_save_rewards(episode, avg_reward)

                # Save the best model
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    best_model_path = os.path.join(self.model_dir, 'best_model.pth')
                    self.agent.save(best_model_path)
                    self.log(f"Best model saved to {best_model_path}")

            # Save model at regular intervals
            if episode % self.save_interval == 0:
                model_path = os.path.join(self.model_dir, f'model_{episode}.pth')
                self.agent.save(model_path)
                self.log(f"Model saved to {model_path}")

        # After training is complete, plot the final rewards
        if self.num_episodes % self.log_interval != 0:
            last_avg_reward = np.mean(self.total_rewards[-(self.num_episodes % self.log_interval):])
            self.plot_and_save_rewards(self.num_episodes, last_avg_reward)

        # Close writer and environment
        self.writer.close()
        self.env.close()
        self.log("Training complete.")

# Define the training function
def train_ppo():
    config = {
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "STATE_DIM": 85,  # Ensure this matches the preprocessed state size
        "ACTION_DIM": 3,
        "BATCH_SIZE": 200,
        "ACTOR_LR": 0.001,
        "CRITIC_LR": 0.0005,
        "GAMMA": 0.99,
        "EPS_CLIP": 0.2,
        "K_EPOCHS": 10,
        "LAM": 0.95,
        "ENTROPY_FACTOR": 0.02,  # Increase this to make the AI explore more
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
