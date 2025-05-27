#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple World Model DQN implementation for UAV OFDMA data rate optimization.
Features a lightweight transition model that only predicts one-step ahead,
avoiding the complexity and instability of multi-step planning.

Added feature: Stop collecting new data after specified episode while continuing training.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import copy
import types
import time
from tqdm import tqdm

from uav_weather_env import WeatherAwareUAVEnv

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experience replay buffer (same as DQN)
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


# Simple Transition Model - only predicts one step ahead
class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(TransitionModel, self).__init__()

        # Input: state + action (one-hot encoded)
        input_dim = state_dim + action_dim

        # Simple MLP for next state prediction
        self.state_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Simple MLP for reward prediction
        self.reward_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state, action):
        # Convert action to one-hot encoding
        batch_size = state.size(0)
        action_one_hot = torch.zeros(batch_size, 9).to(device)  # 9 actions
        action_one_hot.scatter_(1, action, 1)

        # Concatenate state and action
        input_tensor = torch.cat([state, action_one_hot], dim=1)

        # Predict next state and reward
        next_state_pred = self.state_predictor(input_tensor)
        reward_pred = self.reward_predictor(input_tensor)

        return next_state_pred, reward_pred


# DQN Network (same as original)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# World Model DQN Agent
class WorldModelDQNAgent:
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
            use_model_ratio=0.5,  # 50% of training uses model-generated data
            rollout_steps=1
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

        # Initialize Q-networks (same as DQN)
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

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy (same as DQN)"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.q_network(state)
                return q_values.argmax(dim=1).item()

    def train_world_model(self):
        """Train the world model to predict next state and reward"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from real experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Predict next state and reward
        next_state_pred, reward_pred = self.transition_model(states, actions)

        # Compute losses
        state_loss = F.mse_loss(next_state_pred, next_states)
        reward_loss = F.mse_loss(reward_pred, rewards)
        total_model_loss = state_loss + reward_loss

        # Update world model
        self.model_optimizer.zero_grad()
        total_model_loss.backward()
        self.model_optimizer.step()

        return total_model_loss.item()

    # def generate_model_data(self, num_samples):
    #     """Generate synthetic experiences using the world model"""
    #     if len(self.replay_buffer) < num_samples:
    #         return [], [], [], [], []
    #
    #     # Sample random states from replay buffer
    #     experiences = random.sample(self.replay_buffer.buffer, k=num_samples)
    #     states = [e.state for e in experiences]
    #
    #     model_states = []
    #     model_actions = []
    #     model_rewards = []
    #     model_next_states = []
    #     model_dones = []
    #
    #     with torch.no_grad():
    #         for state in states:
    #             # Random action for exploration in model
    #             # action = random.randrange(self.action_dim)
    #
    #             # 不要完全随机，用epsilon-greedy
    #             if random.random() < 0.3:  # 30%随机探索
    #                 action = random.randrange(self.action_dim)
    #             else:  # 70%根据当前策略选择
    #                 with torch.no_grad():
    #                     state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    #                     q_values = self.q_network(state_tensor)
    #                     action = q_values.argmax(dim=1).item()
    #
    #             # Use world model to predict next state and reward
    #             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    #             action_tensor = torch.LongTensor([action]).unsqueeze(1).to(device)
    #
    #             next_state_pred, reward_pred = self.transition_model(state_tensor, action_tensor)
    #
    #             model_states.append(state)
    #             model_actions.append(action)
    #             model_rewards.append(reward_pred.item())
    #             model_next_states.append(next_state_pred.squeeze().cpu().numpy())
    #             model_dones.append(False)  # Assume no termination for simplicity
    #
    #     return model_states, model_actions, model_rewards, model_next_states, model_dones

    def generate_model_data(self, num_samples):
        """Generate synthetic experiences using the world model with multi-step rollout"""
        if len(self.replay_buffer) < num_samples:
            return [], [], [], [], []

        # Sample random states from replay buffer
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

                # 进行多步展开
                for step in range(self.rollout_steps):
                    # 选择动作 (epsilon-greedy)
                    if random.random() < 0.8:
                        action = random.randrange(self.action_dim)
                    else:  # 70%根据当前策略选择
                        state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)
                        q_values = self.q_network(state_tensor)
                        action = q_values.argmax(dim=1).item()

                    # 使用world model预测下一状态和奖励
                    state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)
                    action_tensor = torch.LongTensor([action]).unsqueeze(1).to(device)
                    next_state_pred, reward_pred = self.transition_model(state_tensor, action_tensor)

                    # 存储这一步的转换
                    model_states.append(current_state.copy())
                    model_actions.append(action)
                    model_rewards.append(reward_pred.item())
                    model_next_states.append(next_state_pred.squeeze().cpu().numpy())
                    model_dones.append(False)  # 假设不终止

                    # 更新当前状态为预测的下一状态，用于下一步预测
                    current_state = next_state_pred.squeeze().cpu().numpy()

        return model_states, model_actions, model_rewards, model_next_states, model_dones

    def train_q_network(self, use_model_data=False):
        """Train Q-network using real data and optionally model-generated data"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        if use_model_data and random.random() < self.use_model_ratio:
            # Use model-generated data
            states, actions, rewards, next_states, dones = self.generate_model_data(self.batch_size)

            if len(states) == 0:
                return None

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor([int(d) for d in dones]).unsqueeze(1).to(device)
        else:
            # Use real data from replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute current Q values
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        q_loss = F.mse_loss(q_values, target_q_values)

        # Optimize Q-network
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return q_loss.item()

    def train(self):
        """Combined training of world model and Q-network"""
        # Train world model
        model_loss = None
        if self.steps_done % self.model_update_freq == 0:
            model_loss = self.train_world_model()
            if model_loss is not None:
                self.model_losses.append(model_loss)

        # Train Q-network (mix of real and model data)
        q_loss = self.train_q_network(use_model_data=True)
        if q_loss is not None:
            self.q_losses.append(q_loss)

        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return {"q_loss": q_loss, "model_loss": model_loss}

    def save(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(self.q_network.state_dict(), os.path.join(folder_path, "q_network.pth"))
        torch.save(self.target_q_network.state_dict(), os.path.join(folder_path, "target_q_network.pth"))
        torch.save(self.transition_model.state_dict(), os.path.join(folder_path, "transition_model.pth"))

        # Save training parameters
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

        # Load training parameters
        params = torch.load(os.path.join(folder_path, "params.pth"))
        self.epsilon = params["epsilon"]
        self.steps_done = params["steps_done"]
        self.model_losses = params.get("model_losses", [])
        self.q_losses = params.get("q_losses", [])


# Environment wrapper (same as original)
class UAVEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.action_dim = 9
        self.num_users = env.num_users
        self.state_dim = 2 + 2 * self.num_users + self.num_users
        self.current_render_dir = None

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

    def render(self):
        if self.current_render_dir is not None:
            original_dir = self.env.visualization_dir
            self.env.visualization_dir = self.current_render_dir
            self.env.render()
            self.env.visualization_dir = original_dir
        else:
            self.env.render()

    def set_render_dir(self, directory):
        self.current_render_dir = directory
        if directory is not None and not os.path.exists(directory):
            os.makedirs(directory)


# Training function for World Model DQN with data collection cutoff
def train_world_model_dqn(env_wrapper, agent, num_episodes=200, max_steps=100, render_every=20, save_every=50,
                          data_collection_cutoff_episode=None):
    """
    Train World Model DQN with optional data collection cutoff.

    Args:
        data_collection_cutoff_episode: Episode number after which to stop adding new data to buffer.
                                      If None, data collection continues throughout training.
    """
    rewards_history = []
    avg_rewards_history = []
    model_losses = []
    q_losses = []
    epsilons = []

    # Create directories
    render_dir = os.path.join(os.getcwd(), 'world_model_renders')
    os.makedirs(render_dir, exist_ok=True)
    model_dir = os.path.join(os.getcwd(), 'world_model_models')
    os.makedirs(model_dir, exist_ok=True)

    start_time = time.time()

    # Track buffer size for monitoring
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

        # Set up render directory
        if episode % render_every == 0:
            episode_render_dir = os.path.join(render_dir, f'episode_{episode}')
            os.makedirs(episode_render_dir, exist_ok=True)
            env_wrapper.set_render_dir(episode_render_dir)
        else:
            env_wrapper.set_render_dir(None)

        # Episode loop
        for step in range(1, max_steps + 1):
            # Select action
            action = agent.select_action(state)

            # Take step in environment (always for getting current episode reward)
            next_state, reward, done, _ = env_wrapper.step(action)

            # Store in replay buffer ONLY if data collection is active
            if data_collection_active:
                agent.replay_buffer.add(state, action, reward, next_state, done)

            # Train agent (both world model and Q-network) - this continues regardless
            losses = agent.train()

            if losses["q_loss"] is not None:
                episode_losses["q_loss"].append(losses["q_loss"])
            if losses["model_loss"] is not None:
                episode_losses["model_loss"].append(losses["model_loss"])

            # Render if needed
            if episode % render_every == 0:
                env_wrapper.render()

            # Update state and reward
            state = next_state
            episode_reward += reward

            if done:
                break

        # Record metrics
        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-min(20, len(rewards_history)):])
        avg_rewards_history.append(avg_reward)
        buffer_sizes.append(len(agent.replay_buffer))

        # Record losses
        if episode_losses["q_loss"]:
            q_losses.append(np.mean(episode_losses["q_loss"]))
        else:
            q_losses.append(0)

        if episode_losses["model_loss"]:
            model_losses.append(np.mean(episode_losses["model_loss"]))
        else:
            model_losses.append(0)

        epsilons.append(agent.epsilon)

        # Print progress with data collection status
        if episode % 1 == 0:
            elapsed_time = time.time() - start_time
            data_status = "ACTIVE" if data_collection_active else "STOPPED"
            print(f"Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}, "
                  f"Q Loss: {q_losses[-1]:.4f}, Model Loss: {model_losses[-1]:.4f}, "
                  f"Buffer: {len(agent.replay_buffer)}, Data: {data_status}, "
                  f"Time: {elapsed_time:.2f}s")

        # Save model and plot progress
        if episode % save_every == 0:
            agent.save(os.path.join(model_dir, f'episode_{episode}'))

            # Plot progress with model loss and buffer size
            plt.figure(figsize=(20, 15))

            plt.subplot(3, 3, 1)
            plt.plot(rewards_history)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')

            plt.subplot(3, 3, 2)
            plt.plot(avg_rewards_history)
            plt.title('Average Rewards (20 episodes)')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')

            plt.subplot(3, 3, 3)
            plt.plot(q_losses, label='Q Loss')
            plt.plot(model_losses, label='Model Loss')
            plt.title('Losses per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(3, 3, 4)
            plt.plot(epsilons)
            plt.title('Exploration Rate (Epsilon)')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')

            plt.subplot(3, 3, 5)
            plt.plot(model_losses)
            plt.title('World Model Loss')
            plt.xlabel('Episode')
            plt.ylabel('Model Loss')

            plt.subplot(3, 3, 6)
            plt.plot(buffer_sizes)
            plt.title('Replay Buffer Size')
            plt.xlabel('Episode')
            plt.ylabel('Buffer Size')
            if data_collection_cutoff_episode is not None:
                plt.axvline(x=data_collection_cutoff_episode, color='r', linestyle='--',
                            label=f'Data Cutoff: {data_collection_cutoff_episode}')
                plt.legend()

            plt.subplot(3, 3, 7)
            # Compare with a hypothetical DQN baseline (for illustration)
            plt.plot(avg_rewards_history, label='World Model DQN')
            plt.title('Performance Comparison')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.legend()

            plt.subplot(3, 3, 8)
            # Show data collection phases
            if data_collection_cutoff_episode is not None:
                phase1 = avg_rewards_history[:min(data_collection_cutoff_episode, len(avg_rewards_history))]
                phase2 = avg_rewards_history[data_collection_cutoff_episode:] if data_collection_cutoff_episode < len(
                    avg_rewards_history) else []

                plt.plot(range(1, len(phase1) + 1), phase1, 'b-', label='Data Collection Phase')
                if len(phase2) > 0:
                    plt.plot(range(data_collection_cutoff_episode + 1, len(avg_rewards_history) + 1),
                             phase2, 'r-', label='Model-Only Training Phase')
                plt.axvline(x=data_collection_cutoff_episode, color='k', linestyle='--', alpha=0.7)
                plt.title('Training Phases')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                plt.legend()
            else:
                plt.plot(avg_rewards_history)
                plt.title('Continuous Data Collection')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')

            plt.subplot(3, 3, 9)
            # Loss comparison between phases
            if data_collection_cutoff_episode is not None:
                phase1_losses = q_losses[:min(data_collection_cutoff_episode, len(q_losses))]
                phase2_losses = q_losses[data_collection_cutoff_episode:] if data_collection_cutoff_episode < len(
                    q_losses) else []

                plt.plot(range(1, len(phase1_losses) + 1), phase1_losses, 'b-', label='Data Collection Phase')
                if len(phase2_losses) > 0:
                    plt.plot(range(data_collection_cutoff_episode + 1, len(q_losses) + 1),
                             phase2_losses, 'r-', label='Model-Only Training Phase')
                plt.axvline(x=data_collection_cutoff_episode, color='k', linestyle='--', alpha=0.7)
                plt.title('Q-Loss by Training Phase')
                plt.xlabel('Episode')
                plt.ylabel('Q Loss')
                plt.legend()
            else:
                plt.plot(q_losses)
                plt.title('Q-Loss (Continuous)')
                plt.xlabel('Episode')
                plt.ylabel('Q Loss')

            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f'progress_{episode}.png'))
            plt.close()

    # Save final model
    agent.save(os.path.join(model_dir, 'final'))

    return rewards_history, avg_rewards_history, q_losses, model_losses, epsilons, buffer_sizes


# Evaluation function (same as DQN)
def evaluate_world_model_dqn(env_wrapper, agent, num_episodes=5, render=True):
    rewards = []
    steps = []

    eval_dir = os.path.join(os.getcwd(), 'world_model_eval')
    if render and not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    for episode in range(1, num_episodes + 1):
        state = env_wrapper.reset()
        episode_reward = 0
        step_count = 0

        if render:
            episode_render_dir = os.path.join(eval_dir, f'eval_episode_{episode}')
            os.makedirs(episode_render_dir, exist_ok=True)
            env_wrapper.set_render_dir(episode_render_dir)

        done = False
        while not done and step_count < 100:
            action = agent.select_action(state, training=False)
            next_state, reward, done, _ = env_wrapper.step(action)

            if render:
                env_wrapper.render()

            state = next_state
            episode_reward += reward
            step_count += 1

        rewards.append(episode_reward)
        steps.append(step_count)
        print(f"Evaluation Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f}, Steps: {step_count}")

    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    print(f"Evaluation complete - Average Reward: {avg_reward:.2f}, Average Steps: {avg_steps:.2f}")

    # Plot evaluation results
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, num_episodes + 1), rewards)
    plt.axhline(y=avg_reward, color='r', linestyle='--', label=f'Avg: {avg_reward:.2f}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('World Model DQN Evaluation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(eval_dir, 'evaluation_results.png'))
    plt.close()

    return rewards, steps


# Main function
def main():
    # Environment parameters (same as DQN)
    grid_size = 64
    num_users = 10
    max_steps = 100
    coverage_threshold = 10e6
    grid_cell_size = 4
    uav_altitude = 100
    ofdma_subcarriers = 64
    fixed_seed = 42

    print("=" * 70)
    print("World Model DQN for UAV OFDMA Data Rate Optimization")
    print("WITH DATA COLLECTION CUTOFF FEATURE")
    print("=" * 70)
    print("Key Features:")
    print("- One-step transition model (avoids error accumulation)")
    print("- Mixed training: real + model-generated data")
    print("- Lightweight MLP architecture")
    print("- Deterministic environment setup")
    print("- NEW: Stop data collection after specified episode")
    print("=" * 70)

    # Create environment (same setup as DQN)
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

    # Same environment fixes as DQN version
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

    # Create World Model DQN agent
    state_dim = env_wrapper.state_dim
    action_dim = env_wrapper.action_dim

    agent = WorldModelDQNAgent(
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
        model_lr=1e-3,  # Learning rate for world model
        model_update_freq=4,  # Update world model every 4 steps
        use_model_ratio=1.0,  # 80% model data, 20% real data
        rollout_steps = 1
    )
    test_only = False
    checkpoint_path = ''
    if test_only:
        # Load trained model
        print(f"Loading checkpoint from: {checkpoint_path}")
        agent.load(checkpoint_path)

        # Run world model test
        print("\n" + "=" * 50)
        print("TESTING WORLD MODEL")
        print("=" * 50)
        test_results = test_world_model(agent, env_wrapper, num_steps=100)

        # Run evaluation
        print("\n" + "=" * 50)
        print("EVALUATING AGENT PERFORMANCE")
        print("=" * 50)


        return



    # Training parameters
    num_episodes = 10000
    render_every = 10000
    save_every = 100

    # NEW PARAMETER: Set when to stop collecting new data
    # Set to None to disable cutoff, or specify episode number
    data_collection_cutoff_episode = 4000  # Stop collecting new data after episode 5000

    print(f"Training for {num_episodes} episodes")
    print(f"World model update frequency: every {agent.model_update_freq} steps")
    print(f"Model/Real data ratio: {agent.use_model_ratio:.1%}/{1 - agent.use_model_ratio:.1%}")

    if data_collection_cutoff_episode is not None:
        print(f"Data collection will stop after episode: {data_collection_cutoff_episode}")
        print(
            f"Episodes {data_collection_cutoff_episode + 1}-{num_episodes} will train purely on existing buffer + model data")
    else:
        print("Data collection will continue throughout training")

    # Train agent
    print("\nTraining World Model DQN agent...")
    start_time = time.time()

    rewards_history, avg_rewards_history, q_losses, model_losses, epsilons, buffer_sizes = train_world_model_dqn(
        env_wrapper=env_wrapper,
        agent=agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
        render_every=render_every,
        save_every=save_every,
        data_collection_cutoff_episode=data_collection_cutoff_episode  # NEW PARAMETER
    )

    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # Evaluate agent
    print("\nEvaluating World Model DQN agent...")
    rewards, steps = evaluate_world_model_dqn(
        env_wrapper=env_wrapper,
        agent=agent,
        num_episodes=5,
        render=True
    )

    print("\nTraining and evaluation completed!")
    print(f"Models saved in: {os.path.join(os.getcwd(), 'world_model_models')}")
    print(f"Renders saved in: {os.path.join(os.getcwd(), 'world_model_renders')}")
    print(f"Evaluation results saved in: {os.path.join(os.getcwd(), 'world_model_eval')}")

    # Print final comparison metrics
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
        print(f"Buffer size at cutoff: {buffer_sizes[data_collection_cutoff_episode - 1]} experiences")

        # Compare performance before and after cutoff
        if len(avg_rewards_history) > data_collection_cutoff_episode:
            pre_cutoff_avg = np.mean(
                avg_rewards_history[max(0, data_collection_cutoff_episode - 100):data_collection_cutoff_episode])
            post_cutoff_avg = np.mean(avg_rewards_history[data_collection_cutoff_episode:])
            print(f"Average reward before cutoff (last 100 episodes): {pre_cutoff_avg:.2f}")
            print(f"Average reward after cutoff: {post_cutoff_avg:.2f}")
            print(f"Performance change: {post_cutoff_avg - pre_cutoff_avg:+.2f}")


def test_world_model(agent, env_wrapper, num_steps=100, seed=42):
    """测试训练好的world model，比较真实和预测的reward/state误差"""
    # 设置随机种子 - 使用与原代码相同的方式
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 设置评估模式
    agent.q_network.eval()
    agent.transition_model.eval()

    # 存储结果
    real_rewards = []
    pred_rewards = []
    reward_errors = []
    state_errors = []

    # 重置环境
    state = env_wrapper.reset()
    real_state = torch.FloatTensor(state).unsqueeze(0).to(device)
    pred_state = real_state.clone()

    print(f"\nStarting World Model test for {num_steps} steps...")
    print("-" * 50)

    for step in range(num_steps):
        # 使用训练好的agent选择动作（小epsilon保证一定探索）
        epsilon = 0.8
        if np.random.random() < epsilon:
            action = np.random.randint(env_wrapper.action_dim)
        else:
            with torch.no_grad():
                q_values = agent.q_network(real_state)
                action = q_values.argmax().item()

        # 转换动作为tensor格式（与transition_model期望的格式一致）
        action_tensor = torch.LongTensor([action]).unsqueeze(1).to(device)

        # 真实环境步进
        next_state, real_reward, done, _ = env_wrapper.step(action)
        real_next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        # World model预测
        with torch.no_grad():
            pred_next_state, pred_reward_tensor = agent.transition_model(pred_state, action_tensor)
            pred_reward = pred_reward_tensor.item()

        # 计算误差
        reward_error = abs(real_reward - pred_reward)
        state_error = torch.norm(real_next_state - pred_next_state).item()

        # 存储数据
        real_rewards.append(real_reward)
        pred_rewards.append(pred_reward)
        reward_errors.append(reward_error)
        state_errors.append(state_error)

        if step % 1 == 0:
            print(f"Step {step:3d}: Real Reward={real_reward:8.4f}, Pred Reward={pred_reward:8.4f}, "
                  f"Reward Error={reward_error:8.4f}, State Error={state_error:8.4f}")

        # 更新状态
        real_state = real_next_state
        pred_state = pred_next_state  # 使用预测状态进行下一步预测（测试累积误差）

        if done:
            print(f"Episode ended at step {step + 1}")
            break

    # 恢复训练模式
    agent.q_network.train()
    agent.transition_model.train()

    # 计算统计信息
    avg_reward_error = np.mean(reward_errors)
    avg_state_error = np.mean(state_errors)
    max_reward_error = np.max(reward_errors)
    max_state_error = np.max(state_errors)
    std_reward_error = np.std(reward_errors)
    std_state_error = np.std(state_errors)

    print("-" * 50)
    print("World Model Test Results:")
    print(f"Average reward error: {avg_reward_error:.6f} ± {std_reward_error:.6f}")
    print(f"Maximum reward error: {max_reward_error:.6f}")
    print(f"sum reward error :{np.sum(real_rewards)}")
    print(f"Average state error:  {avg_state_error:.6f} ± {std_state_error:.6f}")
    print(f"Maximum state error:  {max_state_error:.6f}")
    print(f"Test steps: {len(real_rewards)}")

    # 绘制结果
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('World Model Test Results - UAV OFDMA Environment', fontsize=16)

    # Reward error curve
    axes[0, 0].plot(reward_errors, 'r-', alpha=0.7, linewidth=1)
    axes[0, 0].axhline(y=avg_reward_error, color='r', linestyle='--', alpha=0.8,
                       label=f'Average error: {avg_reward_error:.4f}')
    axes[0, 0].set_title('Reward Prediction Error')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Absolute Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # State error curve
    axes[0, 1].plot(state_errors, 'b-', alpha=0.7, linewidth=1)
    axes[0, 1].axhline(y=avg_state_error, color='b', linestyle='--', alpha=0.8,
                       label=f'Average error: {avg_state_error:.4f}')
    axes[0, 1].set_title('State Prediction Error')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('L2 Norm Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Real vs predicted rewards
    steps = range(len(real_rewards))
    axes[1, 0].plot(steps, real_rewards, 'g-', label='Real Reward', alpha=0.8, linewidth=2)
    axes[1, 0].plot(steps, pred_rewards, 'r--', label='Predicted Reward', alpha=0.8, linewidth=2)
    axes[1, 0].set_title('Real vs Predicted Rewards')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Reward Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Cumulative errors
    cumulative_reward_error = np.cumsum(reward_errors)
    cumulative_state_error = np.cumsum(state_errors)
    axes[1, 1].plot(steps, cumulative_reward_error, 'r-', label='Cumulative Reward Error', alpha=0.8, linewidth=2)
    ax2 = axes[1, 1].twinx()
    ax2.plot(steps, cumulative_state_error, 'b-', label='Cumulative State Error', alpha=0.8, linewidth=2)
    axes[1, 1].set_title('Cumulative Error Trends')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Cumulative Reward Error', color='r')
    ax2.set_ylabel('Cumulative State Error', color='b')
    axes[1, 1].tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='b')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    os.makedirs('world_model_test', exist_ok=True)
    plt.savefig('world_model_test/world_model_test_results.png', dpi=300, bbox_inches='tight')
    print(f"\nTest results chart saved to: world_model_test/world_model_test_results.png")

    # 也显示图表（如果在交互环境中）
    plt.show()

    return {
        'real_rewards': real_rewards,
        'pred_rewards': pred_rewards,
        'reward_errors': reward_errors,
        'state_errors': state_errors,
        'avg_reward_error': avg_reward_error,
        'avg_state_error': avg_state_error,
        'max_reward_error': max_reward_error,
        'max_state_error': max_state_error,
        'std_reward_error': std_reward_error,
        'std_state_error': std_state_error
    }

if __name__ == "__main__":
    main()
