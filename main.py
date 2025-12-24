#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced World Model DQN with Trajectory Planning for UAV OFDMA Optimization.
Main training script without visualization (plots removed).

Key improvements:
1. Longer rollout (5-10 steps) for better trajectory planning
2. Model-based action selection using planning
3. Trajectory-level value estimation
4. Better exploitation of learned world model
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import copy
import types
import time

from uav_weather_env import WeatherAwareUAVEnv

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)

        states = torch.FloatTensor([e.state for e in experiences]).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(device)
        dones = torch.FloatTensor([int(e.done) for e in experiences]).unsqueeze(1).to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)


# Enhanced Transition Model with MLP encoder for state only
class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, encoder_dim=16):
        super(TransitionModel, self).__init__()

        # MLP encoder for state
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, encoder_dim),
            nn.ReLU(),
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU()
        )

        # Combined features: encoded state + one-hot action
        combined_dim = encoder_dim + action_dim

        # Enhanced MLP for next state prediction
        self.state_predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, state_dim)
        )

        # Enhanced MLP for reward prediction
        self.reward_predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state, action):
        # Convert action to one-hot encoding
        batch_size = state.size(0)
        action_one_hot = torch.zeros(batch_size, 9).to(device)  # 9 actions
        action_one_hot.scatter_(1, action, 1)

        # Encode state only
        state_encoded = self.state_encoder(state)

        # Concatenate encoded state with one-hot action
        combined_features = torch.cat([state_encoded, action_one_hot], dim=1)

        # Predict next state and reward
        next_state_pred = self.state_predictor(combined_features)
        reward_pred = self.reward_predictor(combined_features)

        return next_state_pred, reward_pred


# DQN Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# Enhanced World Model DQN Agent with Trajectory Planning
class TrajectoryPlanningAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            lr=1e-3,
            buffer_capacity=10000,
            batch_size=64,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.995,
            target_update_freq=10,
            model_lr=1e-3,
            model_update_freq=4,
            use_model_ratio=0.7,
            rollout_steps=5,
            planning_horizon=5,
            num_planning_samples=10,
            use_planning=True
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.model_update_freq = model_update_freq
        self.use_model_ratio = use_model_ratio
        self.steps_done = 0

        self.rollout_steps = rollout_steps
        self.planning_horizon = planning_horizon
        self.num_planning_samples = num_planning_samples
        self.use_planning = use_planning

        # Initialize Q-networks
        self.q_network = DQN(state_dim, action_dim).to(device)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Initialize World Model
        self.transition_model = TransitionModel(state_dim, action_dim).to(device)
        self.model_optimizer = optim.Adam(self.transition_model.parameters(), lr=model_lr)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Metrics tracking
        self.model_losses = []
        self.q_losses = []
        self.planning_stats = []

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy with optional planning."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            if self.use_planning and len(self.replay_buffer) > self.batch_size:
                return self._plan_action(state)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = self.q_network(state_tensor)
                    return q_values.argmax(dim=1).item()

    def _plan_action(self, state):
        """Model-based planning: simulate trajectories and choose best action."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            best_action = 0
            best_value = float('-inf')

            for action in range(self.action_dim):
                trajectory_value = 0.0
                current_state = state_tensor

                for step in range(self.planning_horizon):
                    if step == 0:
                        current_action = action
                    else:
                        q_values = self.q_network(current_state)
                        current_action = q_values.argmax(dim=1).item()

                    action_tensor = torch.LongTensor([current_action]).unsqueeze(1).to(device)
                    next_state_pred, reward_pred = self.transition_model(current_state, action_tensor)

                    trajectory_value += (self.gamma ** step) * reward_pred.item()
                    current_state = next_state_pred

                terminal_q = self.q_network(current_state).max().item()
                trajectory_value += (self.gamma ** self.planning_horizon) * terminal_q

                if trajectory_value > best_value:
                    best_value = trajectory_value
                    best_action = action

            return best_action

    def train_world_model(self):
        """Train the world model to predict next state and reward"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        next_state_pred, reward_pred = self.transition_model(states, actions)

        state_loss = F.mse_loss(next_state_pred, next_states)
        reward_loss = F.mse_loss(reward_pred, rewards)
        total_model_loss = state_loss + reward_loss

        self.model_optimizer.zero_grad()
        total_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.transition_model.parameters(), max_norm=1.0)
        self.model_optimizer.step()

        return total_model_loss.item()

    def generate_model_data(self, num_samples):
        """Generate synthetic experiences using the world model with multi-step rollout"""
        if len(self.replay_buffer) < num_samples:
            return [], [], [], [], []

        experiences = random.sample(self.replay_buffer.buffer, k=num_samples)
        states = [e.state for e in experiences]

        model_states = []
        model_actions = []
        model_rewards = []
        model_next_states = []
        model_dones = []

        with torch.no_grad():
            for initial_state in states:
                current_state = initial_state

                for step in range(self.rollout_steps):
                    if random.random() < 0.3:
                        action = random.randrange(self.action_dim)
                    else:
                        state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)
                        q_values = self.q_network(state_tensor)
                        action = q_values.argmax(dim=1).item()

                    state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)
                    action_tensor = torch.LongTensor([action]).unsqueeze(1).to(device)
                    next_state_pred, reward_pred = self.transition_model(state_tensor, action_tensor)

                    model_states.append(current_state.copy())
                    model_actions.append(action)
                    model_rewards.append(reward_pred.item())
                    model_next_states.append(next_state_pred.squeeze().cpu().numpy())
                    model_dones.append(False)

                    current_state = next_state_pred.squeeze().cpu().numpy()

        return model_states, model_actions, model_rewards, model_next_states, model_dones

    def train_q_network(self, use_model_data=False):
        """Train Q-network using real data and optionally model-generated data"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        if use_model_data and random.random() < self.use_model_ratio:
            states, actions, rewards, next_states, dones = self.generate_model_data(self.batch_size)

            if len(states) == 0:
                return None

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor([int(d) for d in dones]).unsqueeze(1).to(device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        q_loss = F.mse_loss(q_values, target_q_values)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        return q_loss.item()

    def train(self):
        """Combined training of world model and Q-network"""
        model_loss = None
        if self.steps_done % self.model_update_freq == 0:
            model_loss = self.train_world_model()
            if model_loss is not None:
                self.model_losses.append(model_loss)

        q_loss = self.train_q_network(use_model_data=True)
        if q_loss is not None:
            self.q_losses.append(q_loss)

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return {"q_loss": q_loss, "model_loss": model_loss}

    def save(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(self.q_network.state_dict(), os.path.join(folder_path, "q_network.pth"))
        torch.save(self.target_q_network.state_dict(), os.path.join(folder_path, "target_q_network.pth"))
        torch.save(self.transition_model.state_dict(), os.path.join(folder_path, "transition_model.pth"))

        params = {
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "model_losses": self.model_losses,
            "q_losses": self.q_losses
        }
        torch.save(params, os.path.join(folder_path, "params.pth"))

    def load(self, folder_path):
        self.q_network.load_state_dict(torch.load(os.path.join(folder_path, "q_network.pth")))
        self.target_q_network.load_state_dict(torch.load(os.path.join(folder_path, "target_q_network.pth")))
        self.transition_model.load_state_dict(torch.load(os.path.join(folder_path, "transition_model.pth")))

        params = torch.load(os.path.join(folder_path, "params.pth"))
        self.epsilon = params["epsilon"]
        self.steps_done = params["steps_done"]
        self.model_losses = params.get("model_losses", [])
        self.q_losses = params.get("q_losses", [])


# Environment wrapper
class UAVEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.action_dim = 9
        self.num_users = env.num_users
        self.state_dim = 2 + 2 * self.num_users + self.num_users

    def reset(self):
        obs, info = self.env.reset()
        uav_position = self.env.uav_position
        user_positions = self.env.user_positions
        channel_qualities = obs[2:]

        state = np.concatenate([
            uav_position,
            user_positions.flatten(),
            channel_qualities
        ])
        return state

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        uav_position = self.env.uav_position
        user_positions = self.env.user_positions
        channel_qualities = obs[2:]

        next_state = np.concatenate([
            uav_position,
            user_positions.flatten(),
            channel_qualities
        ])

        done = terminated or truncated
        return next_state, reward, done, info


# Training function (without plotting)
def train_trajectory_planner(env_wrapper, agent, num_episodes=200, max_steps=100, save_every=50,
                              data_collection_cutoff_episode=None, project_name='trajectory_planning'):
    """
    Train Trajectory Planning Agent with optional data collection cutoff.
    No visualization - only saves metrics to CSV and models to disk.

    Args:
        project_name: Name of the project, all results will be saved in this folder
    """
    rewards_history = []
    avg_rewards_history = []
    model_losses = []
    q_losses = []
    epsilons = []

    # Create directories with project name
    project_dir = os.path.join(os.getcwd(), project_name)
    os.makedirs(project_dir, exist_ok=True)
    model_dir = os.path.join(project_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Create metrics log file
    metrics_file = os.path.join(project_dir, 'training_metrics.csv')
    with open(metrics_file, 'w') as f:
        f.write('episode,reward,avg_reward,epsilon,q_loss,model_loss,buffer_size,elapsed_time\n')

    start_time = time.time()

    buffer_sizes = []
    data_collection_active = True

    if data_collection_cutoff_episode is not None:
        print(f"Data collection will stop after episode {data_collection_cutoff_episode}")

    for episode in range(1, num_episodes + 1):
        state = env_wrapper.reset()
        episode_reward = 0
        episode_losses = {"q_loss": [], "model_loss": []}

        # Check if we should stop data collection
        if data_collection_cutoff_episode is not None and episode > data_collection_cutoff_episode:
            if data_collection_active:
                print(f"\n*** Data collection stopped at episode {episode} ***")
                print(f"Buffer size frozen at: {len(agent.replay_buffer)} experiences")
                data_collection_active = False

        # Episode loop
        for step in range(1, max_steps + 1):
            action = agent.select_action(state)
            next_state, reward, done, _ = env_wrapper.step(action)

            if data_collection_active:
                agent.replay_buffer.add(state, action, reward, next_state, done)

            losses = agent.train()

            if losses["q_loss"] is not None:
                episode_losses["q_loss"].append(losses["q_loss"])
            if losses["model_loss"] is not None:
                episode_losses["model_loss"].append(losses["model_loss"])

            state = next_state
            episode_reward += reward

            if done:
                break

        # Record metrics
        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-min(20, len(rewards_history)):])
        avg_rewards_history.append(avg_reward)
        buffer_sizes.append(len(agent.replay_buffer))

        if episode_losses["q_loss"]:
            q_losses.append(np.mean(episode_losses["q_loss"]))
        else:
            q_losses.append(0)

        if episode_losses["model_loss"]:
            model_losses.append(np.mean(episode_losses["model_loss"]))
        else:
            model_losses.append(0)

        epsilons.append(agent.epsilon)

        # Write metrics to file
        elapsed_time = time.time() - start_time
        with open(metrics_file, 'a') as f:
            f.write(f'{episode},{episode_reward:.6f},{avg_reward:.6f},{agent.epsilon:.6f},'
                    f'{q_losses[-1]:.6f},{model_losses[-1]:.6f},{len(agent.replay_buffer)},{elapsed_time:.2f}\n')

        # Print progress
        if episode % 1 == 0:
            data_status = "ACTIVE" if data_collection_active else "STOPPED"
            planning_status = "ENABLED" if agent.use_planning else "DISABLED"
            print(f"Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}, "
                  f"Q Loss: {q_losses[-1]:.4f}, Model Loss: {model_losses[-1]:.4f}, "
                  f"Buffer: {len(agent.replay_buffer)}, Data: {data_status}, "
                  f"Planning: {planning_status}, Time: {elapsed_time:.2f}s")

        # Save model
        if episode % save_every == 0:
            agent.save(os.path.join(model_dir, f'episode_{episode}'))
            print(f"Checkpoint saved at episode {episode}")

    # Save final model
    agent.save(os.path.join(model_dir, 'final'))
    print(f"Final model saved")

    return rewards_history, avg_rewards_history, q_losses, model_losses, epsilons, buffer_sizes


# Evaluation function (without plotting)
def evaluate_trajectory_planner(env_wrapper, agent, num_episodes=5, project_name='trajectory_planning'):
    """Evaluate agent without visualization."""
    rewards = []
    steps = []

    project_dir = os.path.join(os.getcwd(), project_name)
    eval_dir = os.path.join(project_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    for episode in range(1, num_episodes + 1):
        state = env_wrapper.reset()
        episode_reward = 0
        step_count = 0

        done = False
        while not done and step_count < 100:
            action = agent.select_action(state, training=False)
            next_state, reward, done, _ = env_wrapper.step(action)

            state = next_state
            episode_reward += reward
            step_count += 1

        rewards.append(episode_reward)
        steps.append(step_count)

        print(f"Evaluation Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f}, Steps: {step_count}")

    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    print(f"Evaluation complete - Average Reward: {avg_reward:.2f}, Average Steps: {avg_steps:.2f}")

    # Save evaluation metrics to file
    eval_metrics_file = os.path.join(eval_dir, 'evaluation_metrics.csv')
    with open(eval_metrics_file, 'w') as f:
        f.write('episode,reward,steps\n')
        for i, (r, s) in enumerate(zip(rewards, steps), 1):
            f.write(f'{i},{r:.6f},{s}\n')

    return rewards, steps


# Main function
def main():
    # Environment parameters
    grid_size = 64
    num_users = 10
    max_steps = 100
    coverage_threshold = 10e6
    grid_cell_size = 4
    uav_altitude = 100
    ofdma_subcarriers = 64
    fixed_seed = 42

    print("=" * 70)
    print("Trajectory Planning Agent for UAV OFDMA Optimization")
    print("Training Mode: No Visualization (Faster)")
    print("=" * 70)

    # Create environment
    env = WeatherAwareUAVEnv(
        grid_size=grid_size,
        num_users=num_users,
        max_steps=max_steps,
        user_distribution="random",
        weather_evolution="gaussian_drift",
        weather_intensity=0.7,
        weather_drift_speed=(0.3, 0.2),
        weather_noise_sigma=0.2,
        coverage_threshold=coverage_threshold,
        grid_cell_size=grid_cell_size,
        uav_altitude=uav_altitude,
        ofdma_subcarriers=ofdma_subcarriers,
        uav_initial_position=(grid_size // 2, grid_size // 2)
    )

    # Fix environment for reproducibility
    _, _ = env.reset(seed=fixed_seed)
    fixed_user_positions = env.user_positions.copy()
    fixed_weather_center = env.weather_center.copy()
    fixed_weather_map = env.weather_map.copy()
    fixed_weather_drift_speed = env.weather_drift_speed

    def fixed_initialize_users(self):
        self.user_positions = fixed_user_positions.copy()

    def fixed_initialize_weather(self):
        self.weather_center = fixed_weather_center.copy()
        self.weather_map = fixed_weather_map.copy()
        self.weather_drift_speed = fixed_weather_drift_speed

    def fixed_update_weather(self):
        dx, dy = self.weather_drift_speed
        self.weather_center[0] += dx
        self.weather_center[1] += dy

        if self.weather_center[0] < 0 or self.weather_center[0] >= self.grid_size:
            dx = -dx
            self.weather_center[0] = np.clip(self.weather_center[0], 0, self.grid_size - 1)

        if self.weather_center[1] < 0 or self.weather_center[1] >= self.grid_size:
            dy = -dy
            self.weather_center[1] = np.clip(self.weather_center[1], 0, self.grid_size - 1)

        self.weather_drift_speed = (dx, dy)

        center_x, center_y = self.weather_center
        sigma = self.grid_size / 5.0
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))

        new_weather_map = self.weather_intensity * np.exp(
            -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2)
        )

        from scipy.ndimage import gaussian_filter
        new_weather_map = gaussian_filter(new_weather_map, sigma=1.0)
        self.weather_map = np.clip(new_weather_map, 0, 1)

    env._initialize_users = types.MethodType(fixed_initialize_users, env)
    env._initialize_weather = types.MethodType(fixed_initialize_weather, env)
    env._update_weather = types.MethodType(fixed_update_weather, env)

    # Create environment wrapper
    env_wrapper = UAVEnvWrapper(env)

    # Create agent
    state_dim = env_wrapper.state_dim
    action_dim = env_wrapper.action_dim

    agent = TrajectoryPlanningAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        buffer_capacity=16000,
        batch_size=64,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        target_update_freq=10,
        model_lr=1e-3,
        model_update_freq=4,
        use_model_ratio=0.7,
        rollout_steps=5,
        planning_horizon=5,
        num_planning_samples=10,
        use_planning=False
    )

    # Training parameters
    num_episodes = 1000
    save_every = 100
    data_collection_cutoff_episode = 4000

    # Project name for organizing results
    project_name = 'main_training_run'

    print(f"\nProject Name: {project_name}")
    print(f"Training for {num_episodes} episodes")
    print(f"Rollout steps: {agent.rollout_steps}")
    print(f"Planning horizon: {agent.planning_horizon}")
    print(f"Model-based planning: {'ENABLED' if agent.use_planning else 'DISABLED'}")
    print(f"Model/Real data ratio: {agent.use_model_ratio:.1%}/{1 - agent.use_model_ratio:.1%}")

    if data_collection_cutoff_episode is not None:
        print(f"Data collection will stop after episode: {data_collection_cutoff_episode}")

    # Train agent
    print("\nTraining Trajectory Planning Agent...")
    start_time = time.time()

    rewards_history, avg_rewards_history, q_losses, model_losses, epsilons, buffer_sizes = train_trajectory_planner(
        env_wrapper=env_wrapper,
        agent=agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
        save_every=save_every,
        data_collection_cutoff_episode=data_collection_cutoff_episode,
        project_name=project_name
    )

    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # Evaluate agent
    print("\nEvaluating Trajectory Planning Agent...")
    rewards, steps = evaluate_trajectory_planner(
        env_wrapper=env_wrapper,
        agent=agent,
        num_episodes=5,
        project_name=project_name
    )

    project_dir = os.path.join(os.getcwd(), project_name)
    print("\nTraining and evaluation completed!")
    print(f"All results saved in: {project_dir}")
    print(f"  - Models: {os.path.join(project_dir, 'models')}")
    print(f"  - Training metrics: {os.path.join(project_dir, 'training_metrics.csv')}")
    print(f"  - Evaluation metrics: {os.path.join(project_dir, 'evaluation/evaluation_metrics.csv')}")

    # Print final results
    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)
    print(f"Final average reward (last 20 episodes): {avg_rewards_history[-1]:.2f}")
    print(f"Final Q-network loss: {q_losses[-1]:.4f}")
    print(f"Final world model loss: {model_losses[-1]:.4f}")
    print(f"Final buffer size: {buffer_sizes[-1]} experiences")
    print(f"Evaluation average reward: {np.mean(rewards):.2f}")

    if data_collection_cutoff_episode is not None:
        print(f"\nData Collection Analysis:")
        print(f"Data collection stopped at episode: {data_collection_cutoff_episode}")
        print(f"Buffer size at cutoff: {buffer_sizes[min(data_collection_cutoff_episode - 1, len(buffer_sizes) - 1)]} experiences")

        if len(avg_rewards_history) > data_collection_cutoff_episode:
            pre_cutoff_avg = np.mean(
                avg_rewards_history[max(0, data_collection_cutoff_episode - 100):data_collection_cutoff_episode])
            post_cutoff_avg = np.mean(avg_rewards_history[data_collection_cutoff_episode:])
            print(f"Average reward before cutoff (last 100 episodes): {pre_cutoff_avg:.2f}")
            print(f"Average reward after cutoff: {post_cutoff_avg:.2f}")
            print(f"Performance change: {post_cutoff_avg - pre_cutoff_avg:+.2f}")


if __name__ == "__main__":
    main()
