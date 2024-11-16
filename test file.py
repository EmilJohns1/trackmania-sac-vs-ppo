import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
import numpy as np
from tmrl import get_environment
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Tuple, Box  # Correct import from gymnasium.spaces
import os
import re

# Constants for Reward Function
TARGET_SPEED = 50.0  # Desired speed in the game units (adjust as needed)
SPEED_THRESHOLD_LOW = 1.0  # Below this speed, encourage acceleration
SPEED_THRESHOLD_HIGH = 100.0  # Above this speed, encourage braking
LIDAR_THRESHOLD = 200  # Distance in LIDAR units to penalize proximity to walls
STEERING_CHANGE_THRESHOLD = 0.5  # Maximum allowed change in steering angle between steps
REWARD_COMPLETION = 1000.0  # Reward for completing the track
PENALTY_CRASH = -50.0  # Penalty for crashing into walls
PENALTY_REVERSE = -10.0  # Penalty for moving backwards
PENALTY_STANDING_STILL = -100  # Small penalty for not moving forward
PENALTY_UNNECESSARY_BRAKING = -10  # Penalize braking when not necessary
PENALTY_INSUFFICIENT_ACCELERATION = -1.0  # Penalize not accelerating when needed
PENALTY_STEERING_CHANGE = -0.5  # Penalize large steering changes


def find_latest_model(directory, prefix="ppo_trackmania_episode_"):
    """
    Finds the latest model file in the specified directory based on the episode number.
    """
    files = os.listdir(directory)
    model_files = [f for f in files if f.startswith(prefix) and f.endswith(".pth")]

    # Extract episode numbers and sort
    episode_numbers = [
        int(re.search(rf"{prefix}(\d+)\.pth", f).group(1)) for f in model_files if re.search(rf"{prefix}(\d+)\.pth", f)
    ]
    if not episode_numbers:
        return None

    latest_episode = max(episode_numbers)
    return os.path.join(directory, f"{prefix}{latest_episode}.pth")


# Helper function to flatten observations
def flatten_observation(obs, space):
    """
    Flattens the observation based on its space.
    """
    if isinstance(space, Tuple):
        # Recursively flatten each component of the Tuple
        return np.concatenate([flatten_observation(o, s) for o, s in zip(obs, space.spaces)])
    elif isinstance(space, Box):
        return obs.flatten()
    else:
        raise NotImplementedError(f"Flattening not implemented for space type {type(space)}")

# Continuous Actor Class
class ContinuousActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super(ContinuousActor, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        self.hidden = nn.Sequential(*layers)

        # Output layers for mean and log_std
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)

        # Log standard deviation clamping parameters for numerical stability
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
        """
        Selects an action based on the current policy.
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        z = dist.rsample()  # Reparameterization trick
        action = torch.tanh(z)  # Squash to [-1, 1]

        # To compute log_prob, account for the tanh transformation
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return action.detach().cpu().numpy(), log_prob.detach(), entropy.detach()

# Critic Class
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_sizes=[256, 256]):
        super(Critic, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, state):
        x = self.hidden(state)
        value = self.value_head(x)
        return value

# State Normalizer Class
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

        # Update running mean and variance
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        new_var = (self.var * self.count + batch_var * batch_count +
                   delta ** 2 * self.count * batch_count / tot_count) / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, states):
        return (states - self.mean) / (np.sqrt(self.var) + 1e-8)

# PPOAgent Class
class PPOAgent:
    def __init__(self, state_dim, action_dim,
                 actor_lr=0.0003, critic_lr=0.001,
                 gamma=0.99, eps_clip=0.3,
                 value_epochs=4, policy_epochs=4,
                 minibatch_size=64, lam=0.95, entropy_factor=0.01,
                 device='cpu'):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_epochs = value_epochs
        self.policy_epochs = policy_epochs
        self.minibatch_size = minibatch_size
        self.lam = lam  # GAE lambda
        self.entropy_factor = entropy_factor
        self.device = torch.device(device)

        self.actor = ContinuousActor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = []

    def select_action(self, state):
        # Convert state to PyTorch Tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Shape: [1, state_dim]

        # Get action from the actor
        action, log_prob, entropy = self.actor.select_action(state)

        # Remove batch dimension and return
        return action[0], log_prob, entropy

    def store_transition(self, transition):
        self.memory.append(transition)

    def compute_advantages(self, rewards, dones, values, next_values):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update(self):
        if len(self.memory) == 0:
            return  # Nothing to update

        # Extract data from memory
        states = torch.FloatTensor([m['state'] for m in self.memory]).to(self.device)
        actions = torch.FloatTensor([m['action'] for m in self.memory]).to(self.device)
        log_probs_old = torch.FloatTensor([m['log_prob'] for m in self.memory]).to(self.device)
        rewards = [m['reward'] for m in self.memory]
        dones = [m['done'] for m in self.memory]

        with torch.no_grad():
            values = self.critic(states).squeeze().cpu().numpy()
            next_states = torch.FloatTensor([m['next_state'] for m in self.memory]).to(self.device)
            next_values = self.critic(next_states).squeeze().cpu().numpy()

        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, dones, values, next_values)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Shuffle the indices for minibatch processing
        num_samples = states.size(0)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Split indices into minibatches
        for _ in range(self.value_epochs):
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = indices[start:end]

                # Critic update
                minibatch_states = states[minibatch_indices]
                minibatch_returns = returns[minibatch_indices]
                state_values = self.critic(minibatch_states).squeeze()
                critic_loss = nn.MSELoss()(state_values, minibatch_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        for _ in range(self.policy_epochs):
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = indices[start:end]

                minibatch_states = states[minibatch_indices]
                minibatch_actions = actions[minibatch_indices]
                minibatch_log_probs_old = log_probs_old[minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]

                # Forward pass through Actor to get current log_probs
                mean, std = self.actor(minibatch_states)
                dist = Normal(mean, std)
                z = torch.atanh(torch.clamp(minibatch_actions, -0.999999, 0.999999))
                current_log_probs = dist.log_prob(z) - torch.log(1 - minibatch_actions.pow(2) + 1e-7)
                current_log_probs = current_log_probs.sum(dim=-1)

                # PPO ratio
                ratios = torch.exp(current_log_probs - minibatch_log_probs_old)

                # Surrogate loss
                surr1 = ratios * minibatch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * minibatch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_factor * dist.entropy().sum(dim=-1).mean()

                # Actor update
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

        # Clear memory after update
        self.memory = []

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

# Training Loop with Enhanced Reward Function
def train():
    env = get_environment()

    print("Observation Space:", env.observation_space)
    print("Observation Space Shape:", getattr(env.observation_space, 'shape', 'No shape attribute'))

    # Determine state_dim based on the observation space
    if isinstance(env.observation_space, Tuple):
        state_dim = sum([int(np.prod(space.shape)) for space in env.observation_space.spaces])
    elif isinstance(env.observation_space, Box):
        state_dim = int(np.prod(env.observation_space.shape))
    else:
        raise NotImplementedError(
            f"State dimension determination not implemented for space type {type(env.observation_space)}")

    action_dim = env.action_space.shape[0]  # For continuous actions

    print(f"Computed state_dim: {state_dim}, action_dim: {action_dim}")

    # Initialize PPO Agent with specified parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=0.0003,  # Reduced learning rate for stability
        critic_lr=0.001,
        gamma=0.99,
        eps_clip=0.2,
        value_epochs=4,
        policy_epochs=4,
        minibatch_size=64,
        lam=0.95,
        entropy_factor=0.01,
        device=device
    )

    model_directory = "."  # Replace with your model directory if different
    latest_model_path = find_latest_model(model_directory)

    if latest_model_path:
        agent.load_model(latest_model_path)
        print(f"Loaded the latest model: {latest_model_path}")
    else:
        print("No saved model found. Starting fresh training.")

    # Initialize State Normalizer
    normalizer = StateNormalizer(state_dim)

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/ppo_trackmania')

    # Training parameters
    num_episodes = 1000
    max_steps = 10000000  # Max steps per episode

    # Initialize previous steering for smoothness penalty
    previous_steering = 0.0

    for episode in range(1, num_episodes + 1):
        raw_state, info = env.reset()
        state = flatten_observation(raw_state, env.observation_space)
        state = normalizer.normalize(state)
        done = False
        episode_reward = 0
        states_batch = []
        steering_changes = []

        for step in range(max_steps):
            # Select action
            action, log_prob, entropy = agent.select_action(state)

            # Extract relevant parts of the observation
            speed = raw_state[0][0]
            lidar = raw_state[1][0]
            previous_actions = raw_state[2:]
            front_lidar = lidar[:5]
            # Apply action to environment
            raw_next_state, env_reward, done, truncated, info = env.step(action)
            next_state = flatten_observation(raw_next_state, env.observation_space)
            next_state = normalizer.normalize(next_state)

            # Initialize reward components
            reward_total = 0.0
            print(f"Step: {step}, Done: {done}, Truncated: {truncated}, Info: {info}")
            print(f"Action: {action}, Speed: {speed}, Lidar: {lidar[:5]}, Reward: {reward_total}")

            if speed < SPEED_THRESHOLD_LOW and action[1] > 0.2:
                reward_total += PENALTY_UNNECESSARY_BRAKING * 5  # Strongly penalize braking at low speed
                print("Unnecessary braking at low speed: Penalizing heavily.")


            if speed < 0.005:
                reward_total += PENALTY_STANDING_STILL * 10  # Strongly penalize standing still
                print("Agent is standing still: Penalizing heavily.")

            if action[0] > 0.5:  # Throttle above a reasonable level
                reward_total += 10.0  # Strongly reward higher throttle
                print("Throttle detected: Rewarding strongly.")

            if np.any(front_lidar < LIDAR_THRESHOLD):
                reward_total += PENALTY_CRASH
                print("Collision detected: Penalizing heavily.")

                # Conflicting Actions
            if action[0] > 0.5 and action[1] > 0.2:  # Throttle and brake simultaneously
                reward_total -= 20.0
                print("Conflicting actions detected: Penalizing.")

            # Accumulate total reward
            episode_reward += reward_total

            # Store transition with the calculated reward
            agent.store_transition({
                'state': state,
                'action': action,
                'log_prob': log_prob.cpu().numpy(),
                'reward': reward_total,
                'next_state': next_state,
                'done': done or truncated,
                'entropy': entropy.cpu().numpy(),
            })

            states_batch.append(state)
            state = next_state

            if done or truncated:
                break

        # Update State Normalizer with collected states
        normalizer.update(states_batch)

        # Update PPO agent after each episode
        agent.update()

        # Logging
        if episode % 10 == 0:
            average_steering_change = np.mean(steering_changes) if steering_changes else 0.0
            print(
                f"Episode {episode} \t Reward: {episode_reward:.2f} \t Avg Steering Change: {average_steering_change:.2f}")
            writer.add_scalar('Reward', episode_reward, episode)
            writer.add_scalar('Avg Steering Change', average_steering_change, episode)

        # Optional: Save model periodically
        if episode % 100 == 0:
            agent.save_model(f'ppo_trackmania_episode_{episode}.pth')
            print(f"Model saved at episode {episode}")
            writer.add_scalar('Model Saved', episode, episode)

    # Close TensorBoard writer and environment
    writer.close()
    env.close()

if __name__ == "__main__":
    train()
