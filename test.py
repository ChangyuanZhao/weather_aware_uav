#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test trained World Model DQN checkpoint

Features:
1. Load checkpoint and test episode rewards
2. Evaluate world model trajectory prediction accuracy
3. Calculate statistical metrics: R², rho (Pearson), MAPE
4. Visualize real vs predicted comparisons
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import types

# Import modules
sys.path.append('/home/changyuan/dreamer/weather_aware')
from world_model_new import TrajectoryPlanningAgent, UAVEnvWrapper
from uav_weather_env import WeatherAwareUAVEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """
    Calculate MAPE (Mean Absolute Percentage Error)

    MAPE = (1/n) * Σ|y_true - y_pred| / (|y_true| + epsilon) * 100%

    Use epsilon to avoid division by zero
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    return mape


def test_episode_rewards(agent, env_wrapper, num_episodes=10, max_steps=100, use_planning=True):
    """
    Test 1: Evaluate agent's episode cumulative rewards

    Args:
        agent: Trained agent
        env_wrapper: Environment wrapper
        num_episodes: Number of test episodes
        max_steps: Maximum steps per episode
        use_planning: Whether to use model-based planning

    Returns:
        episode_rewards: List of cumulative rewards per episode
        inference_times: List of inference times per episode (in seconds)
    """
    import time

    print("\n" + "=" * 70)
    print("Test 1: Episode Reward Evaluation")
    print("=" * 70)

    # Temporarily set planning state
    original_planning = agent.use_planning
    agent.use_planning = use_planning

    # Set to evaluation mode
    agent.q_network.eval()
    agent.transition_model.eval()

    episode_rewards = []
    episode_steps = []
    episode_inference_times = []  # Track total inference time per episode
    step_inference_times = []     # Track inference time per step

    for episode in range(1, num_episodes + 1):
        state = env_wrapper.reset()
        episode_reward = 0.0
        step_count = 0
        episode_start_time = time.time()

        done = False
        while not done and step_count < max_steps:
            # Measure inference time for action selection
            inference_start = time.time()
            action = agent.select_action(state, training=False)
            inference_time = time.time() - inference_start
            step_inference_times.append(inference_time)

            next_state, reward, done, info = env_wrapper.step(action)

            episode_reward += reward
            state = next_state
            step_count += 1

        episode_total_time = time.time() - episode_start_time
        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)
        episode_inference_times.append(episode_total_time)

        print(f"  Episode {episode}/{num_episodes}: "
              f"Reward = {episode_reward:8.2f}, Steps = {step_count}, "
              f"Time = {episode_total_time:.3f}s")

    # Statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_steps = np.mean(episode_steps)
    avg_episode_time = np.mean(episode_inference_times)
    avg_step_time = np.mean(step_inference_times)
    total_inference_time = np.sum(episode_inference_times)

    print("\n" + "-" * 70)
    print(f"Average Cumulative Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min/Max Reward: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Planning Status: {'ENABLED' if use_planning else 'DISABLED'}")
    print(f"\nInference Time Statistics:")
    print(f"  Total Inference Time: {total_inference_time:.3f}s")
    print(f"  Average Time per Episode: {avg_episode_time:.3f}s")
    print(f"  Average Time per Step: {avg_step_time*1000:.2f}ms")
    print(f"  Min/Max Step Time: {np.min(step_inference_times)*1000:.2f}ms / {np.max(step_inference_times)*1000:.2f}ms")
    print(f"  Inference FPS: {1.0/avg_step_time:.2f} steps/second")
    print("=" * 70)

    # Restore original settings
    agent.use_planning = original_planning
    agent.q_network.train()
    agent.transition_model.train()

    return episode_rewards, episode_inference_times, step_inference_times


def test_trajectory_prediction(agent, env_wrapper, num_steps=100, seed=42, save_dir='test_results'):
    """
    Test 2: Evaluate world model single-step prediction accuracy

    Strategy:
    - At each step, use world model to predict reward for (current_state, action)
    - Execute action in real environment and get real reward
    - Compare predicted vs real reward at each step
    - Use REAL next state for next prediction (no cumulative error)

    Args:
        save_dir: Directory to save results

    Returns:
        results: Dictionary containing all prediction data and statistical metrics
    """
    print("\n" + "=" * 70)
    print("Test 2: World Model Trajectory Prediction Accuracy Evaluation")
    print("=" * 70)

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Set to evaluation mode
    agent.q_network.eval()
    agent.transition_model.eval()

    # Storage
    real_rewards = []
    pred_rewards = []
    real_positions = []  # UAV positions
    pred_positions = []

    # Reset environment
    state = env_wrapper.reset()
    real_state = torch.FloatTensor(state).unsqueeze(0).to(device)
    pred_state = real_state.clone()  # Predicted state starts from real initial state

    print(f"\nStarting trajectory prediction test for {num_steps} steps...")
    print("-" * 70)

    for step in range(num_steps):
        # === Action selection (using trained policy) ===
        action = agent.select_action(state, training=False)
        action_tensor = torch.LongTensor([action]).unsqueeze(1).to(device)

        # === World model prediction (BEFORE executing real action) ===
        # Predict reward for current state + selected action
        with torch.no_grad():
            pred_next_state, pred_reward_tensor = agent.transition_model(real_state, action_tensor)
            pred_reward = pred_reward_tensor.item()

        # === Real environment execution ===
        next_state, real_reward, done, _ = env_wrapper.step(action)
        real_next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        # Extract UAV positions (first 2 dimensions of state)
        real_pos = next_state[:2]
        pred_pos = pred_next_state[0, :2].cpu().numpy()
        real_positions.append(real_pos)
        pred_positions.append(pred_pos)

        # === Store data ===
        real_rewards.append(real_reward)
        pred_rewards.append(pred_reward)

        # === Print progress ===
        if step % 10 == 0 or step < 5:
            reward_error = abs(real_reward - pred_reward)
            state_error = torch.norm(real_next_state - pred_next_state).item()
            print(f"  Step {step:3d}: Action={action}, Real R={real_reward:7.4f}, "
                  f"Pred R={pred_reward:7.4f}, "
                  f"R_err={reward_error:7.4f}, "
                  f"S_err={state_error:7.4f}")

        # === Update state (use REAL next state for next prediction) ===
        state = next_state
        real_state = real_next_state

        if done:
            print(f"\n  Episode ended at step {step + 1}")
            break

    # === Calculate statistical metrics ===
    print("\n" + "=" * 70)
    print("Statistical Metrics Calculation")
    print("=" * 70)

    real_rewards = np.array(real_rewards)
    pred_rewards = np.array(pred_rewards)
    real_positions = np.array(real_positions)
    pred_positions = np.array(pred_positions)

    # 1. R² (R-squared, coefficient of determination)
    r2 = r2_score(real_rewards, pred_rewards)
    print(f"\nR² (Coefficient of Determination): {r2:.6f}")
    print(f"   Interpretation: R² closer to 1 indicates more accurate model predictions")
    print(f"   Rating: ", end="")
    if r2 > 0.9:
        print("Excellent (>0.9)")
    elif r2 > 0.7:
        print("Good (0.7-0.9)")
    elif r2 > 0.5:
        print("Fair (0.5-0.7)")
    else:
        print("Poor (<0.5)")

    # 2. Pearson correlation coefficient rho
    rho, p_value = pearsonr(real_rewards, pred_rewards)
    print(f"\nrho (Pearson Correlation Coefficient): {rho:.6f}")
    print(f"   p-value: {p_value:.6e} (significant if p<0.05)")
    print(f"   Interpretation: rho closer to 1 indicates stronger linear correlation")
    print(f"   Rating: ", end="")
    if abs(rho) > 0.9:
        print("Strong correlation (|rho|>0.9)")
    elif abs(rho) > 0.7:
        print("Moderate correlation (0.7-0.9)")
    else:
        print("Weak correlation (<0.7)")

    # 3. MAPE (Mean Absolute Percentage Error)
    mape = calculate_mape(real_rewards, pred_rewards)
    print(f"\nMAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print(f"   Interpretation: Lower MAPE indicates more accurate predictions")
    print(f"   Rating: ", end="")
    if mape < 10:
        print("Excellent (<10%)")
    elif mape < 20:
        print("Good (10-20%)")
    elif mape < 30:
        print("Acceptable (20-30%)")
    else:
        print("Needs improvement (>30%)")

    # 4. Basic error statistics
    reward_errors = np.abs(real_rewards - pred_rewards)
    position_errors = np.linalg.norm(real_positions - pred_positions, axis=1)

    print(f"\nAdditional Statistics:")
    print(f"   Reward MAE (Mean Absolute Error): {np.mean(reward_errors):.6f}")
    print(f"   Reward RMSE (Root Mean Square Error): {np.sqrt(np.mean(reward_errors**2)):.6f}")
    print(f"   Cumulative Reward - Real: {np.sum(real_rewards):.2f}")
    print(f"   Cumulative Reward - Predicted: {np.sum(pred_rewards):.2f}")
    print(f"   Cumulative Reward Error: {abs(np.sum(real_rewards) - np.sum(pred_rewards)):.2f}")
    print(f"   Average Position Error: {np.mean(position_errors):.6f} grid cells")
    print(f"   Maximum Position Error: {np.max(position_errors):.6f} grid cells")

    print("=" * 70)

    # === Save detailed step-by-step data to txt file ===
    print("\nSaving detailed prediction data to file...")
    os.makedirs(save_dir, exist_ok=True)
    data_file = f'{save_dir}/prediction_details.txt'

    with open(data_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("World Model Prediction Accuracy - Detailed Step-by-Step Data\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Total Steps: {len(real_rewards)}\n")
        f.write(f"Test Seed: {seed}\n\n")

        # Header
        f.write("-" * 100 + "\n")
        f.write(f"{'Step':>5} | {'Real Reward':>12} | {'Pred Reward':>12} | {'Reward Error':>12} | "
                f"{'Real Pos X':>10} | {'Real Pos Y':>10} | {'Pred Pos X':>10} | {'Pred Pos Y':>10} | {'Pos Error':>10}\n")
        f.write("-" * 100 + "\n")

        # Data rows
        for i in range(len(real_rewards)):
            f.write(f"{i:5d} | {real_rewards[i]:12.6f} | {pred_rewards[i]:12.6f} | {reward_errors[i]:12.6f} | "
                    f"{real_positions[i][0]:10.4f} | {real_positions[i][1]:10.4f} | "
                    f"{pred_positions[i][0]:10.4f} | {pred_positions[i][1]:10.4f} | {position_errors[i]:10.6f}\n")

        f.write("-" * 100 + "\n\n")

        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 100 + "\n")
        f.write(f"R² (Coefficient of Determination):     {r2:.6f}\n")
        f.write(f"Pearson Correlation (ρ):                {rho:.6f}\n")
        f.write(f"P-value:                                 {p_value:.6e}\n")
        f.write(f"MAPE (Mean Absolute Percentage Error):  {mape:.4f}%\n")
        f.write(f"MAE (Mean Absolute Error):               {np.mean(reward_errors):.6f}\n")
        f.write(f"RMSE (Root Mean Square Error):           {np.sqrt(np.mean(reward_errors**2)):.6f}\n")
        f.write(f"Cumulative Real Reward:                  {np.sum(real_rewards):.6f}\n")
        f.write(f"Cumulative Predicted Reward:             {np.sum(pred_rewards):.6f}\n")
        f.write(f"Cumulative Reward Error:                 {abs(np.sum(real_rewards) - np.sum(pred_rewards)):.6f}\n")
        f.write(f"Average Position Error:                  {np.mean(position_errors):.6f} grid cells\n")
        f.write(f"Maximum Position Error:                  {np.max(position_errors):.6f} grid cells\n")
        f.write("=" * 100 + "\n")

    print(f"Detailed data saved to: {data_file}")

    # Also save raw data in CSV format for easy processing
    csv_file = f'{save_dir}/prediction_data.csv'
    with open(csv_file, 'w') as f:
        f.write("step,real_reward,pred_reward,reward_error,real_pos_x,real_pos_y,pred_pos_x,pred_pos_y,pos_error\n")
        for i in range(len(real_rewards)):
            f.write(f"{i},{real_rewards[i]:.6f},{pred_rewards[i]:.6f},{reward_errors[i]:.6f},"
                    f"{real_positions[i][0]:.4f},{real_positions[i][1]:.4f},"
                    f"{pred_positions[i][0]:.4f},{pred_positions[i][1]:.4f},{position_errors[i]:.6f}\n")

    print(f"CSV data saved to: {csv_file}")

    # Restore training mode
    agent.q_network.train()
    agent.transition_model.train()

    return {
        'real_rewards': real_rewards,
        'pred_rewards': pred_rewards,
        'real_positions': real_positions,
        'pred_positions': pred_positions,
        'reward_errors': reward_errors,
        'position_errors': position_errors,
        'r2': r2,
        'rho': rho,
        'p_value': p_value,
        'mape': mape,
        'mae': np.mean(reward_errors),
        'rmse': np.sqrt(np.mean(reward_errors**2))
    }


def visualize_results(episode_rewards, trajectory_results, save_dir='test_results',
                      step_inference_times=None, use_planning=False):
    """
    Visualize test results

    Generate 9 plots:
    1. Episode rewards bar chart
    2. Real vs predicted rewards comparison
    3. Reward error curve
    4. Scatter plot + fitted line (real vs predicted)
    5. UAV trajectory comparison
    6. Position error curve
    7. Cumulative errors
    8. Error distribution histogram
    9. Key metrics summary (including inference time)
    """
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 15))

    # === Plot 1: Episode Rewards Bar Chart ===
    ax1 = plt.subplot(3, 3, 1)
    episodes = range(1, len(episode_rewards) + 1)
    bars = ax1.bar(episodes, episode_rewards, color='steelblue', alpha=0.8)
    avg_reward = np.mean(episode_rewards)
    ax1.axhline(y=avg_reward, color='red', linestyle='--', linewidth=2,
                label=f'Average: {avg_reward:.2f}')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Cumulative Reward', fontsize=11)
    ax1.set_title('Episode Cumulative Rewards', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # === Plot 2: Real vs Predicted Rewards Time Series ===
    ax2 = plt.subplot(3, 3, 2)
    steps = range(len(trajectory_results['real_rewards']))
    ax2.plot(steps, trajectory_results['real_rewards'], 'g-',
             label='Real Reward', linewidth=2, alpha=0.8)
    ax2.plot(steps, trajectory_results['pred_rewards'], 'r--',
             label='Predicted Reward', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Steps', fontsize=11)
    ax2.set_ylabel('Reward', fontsize=11)
    ax2.set_title('Real vs Predicted Rewards', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # === Plot 3: Reward Prediction Error Curve ===
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(steps, trajectory_results['reward_errors'], 'r-', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=trajectory_results['mae'], color='darkred', linestyle='--',
                linewidth=2, label=f"MAE: {trajectory_results['mae']:.4f}")
    ax3.fill_between(steps, 0, trajectory_results['reward_errors'],
                     alpha=0.3, color='red')
    ax3.set_xlabel('Steps', fontsize=11)
    ax3.set_ylabel('Absolute Error', fontsize=11)
    ax3.set_title('Reward Prediction Error', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # === Plot 4: Scatter Plot + Fitted Line (Real vs Predicted) ===
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(trajectory_results['real_rewards'],
                trajectory_results['pred_rewards'],
                alpha=0.6, s=30, c='blue', edgecolors='black', linewidth=0.5)

    # Plot perfect prediction line (y=x)
    min_val = min(np.min(trajectory_results['real_rewards']),
                  np.min(trajectory_results['pred_rewards']))
    max_val = max(np.max(trajectory_results['real_rewards']),
                  np.max(trajectory_results['pred_rewards']))
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--',
             linewidth=2, label='Perfect Prediction (y=x)')

    # Linear fit
    z = np.polyfit(trajectory_results['real_rewards'],
                   trajectory_results['pred_rewards'], 1)
    p = np.poly1d(z)
    ax4.plot(trajectory_results['real_rewards'],
             p(trajectory_results['real_rewards']),
             'r-', linewidth=2, alpha=0.8, label=f'Fitted Line (y={z[0]:.2f}x+{z[1]:.2f})')

    ax4.set_xlabel('Real Reward', fontsize=11)
    ax4.set_ylabel('Predicted Reward', fontsize=11)
    ax4.set_title(f'Scatter Plot (R²={trajectory_results["r2"]:.4f}, '
                  f'rho={trajectory_results["rho"]:.4f})',
                  fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # === Plot 5: UAV Trajectory Comparison ===
    ax5 = plt.subplot(3, 3, 5)
    real_pos = trajectory_results['real_positions']
    pred_pos = trajectory_results['pred_positions']

    ax5.plot(real_pos[:, 0], real_pos[:, 1], 'g-o',
             label='Real Trajectory', linewidth=2, markersize=3, alpha=0.7)
    ax5.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--s',
             label='Predicted Trajectory', linewidth=2, markersize=3, alpha=0.7)

    # Mark start and end points
    ax5.plot(real_pos[0, 0], real_pos[0, 1], 'go', markersize=15,
             label='Start', zorder=5)
    ax5.plot(real_pos[-1, 0], real_pos[-1, 1], 'g^', markersize=15,
             label='Real End', zorder=5)
    ax5.plot(pred_pos[-1, 0], pred_pos[-1, 1], 'rs', markersize=15,
             label='Predicted End', zorder=5)

    ax5.set_xlabel('X Coordinate', fontsize=11)
    ax5.set_ylabel('Y Coordinate', fontsize=11)
    ax5.set_title('UAV Trajectory Comparison', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')

    # === Plot 6: Position Prediction Error ===
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(steps, trajectory_results['position_errors'],
             'purple', linewidth=2, alpha=0.8)
    ax6.axhline(y=np.mean(trajectory_results['position_errors']),
                color='darkviolet', linestyle='--', linewidth=2,
                label=f"Average: {np.mean(trajectory_results['position_errors']):.4f}")
    ax6.fill_between(steps, 0, trajectory_results['position_errors'],
                     alpha=0.3, color='purple')
    ax6.set_xlabel('Steps', fontsize=11)
    ax6.set_ylabel('Euclidean Distance Error (grid cells)', fontsize=11)
    ax6.set_title('UAV Position Prediction Error', fontsize=13, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # === Plot 7: Cumulative Errors ===
    ax7 = plt.subplot(3, 3, 7)
    cumulative_reward_error = np.cumsum(trajectory_results['reward_errors'])
    cumulative_position_error = np.cumsum(trajectory_results['position_errors'])

    ax7_twin = ax7.twinx()
    line1 = ax7.plot(steps, cumulative_reward_error, 'r-',
                     linewidth=2, label='Cumulative Reward Error', alpha=0.8)
    line2 = ax7_twin.plot(steps, cumulative_position_error, 'b-',
                          linewidth=2, label='Cumulative Position Error', alpha=0.8)

    ax7.set_xlabel('Steps', fontsize=11)
    ax7.set_ylabel('Cumulative Reward Error', fontsize=11, color='r')
    ax7_twin.set_ylabel('Cumulative Position Error', fontsize=11, color='b')
    ax7.tick_params(axis='y', labelcolor='r')
    ax7_twin.tick_params(axis='y', labelcolor='b')
    ax7.set_title('Cumulative Error Trends', fontsize=13, fontweight='bold')

    # Merge legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax7.legend(lines, labels, loc='upper left')
    ax7.grid(True, alpha=0.3)

    # === Plot 8: Error Distribution Histogram ===
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(trajectory_results['reward_errors'], bins=30,
             color='coral', alpha=0.7, edgecolor='black')
    ax8.axvline(x=trajectory_results['mae'], color='red',
                linestyle='--', linewidth=2, label=f'MAE: {trajectory_results["mae"]:.4f}')
    ax8.set_xlabel('Reward Prediction Error', fontsize=11)
    ax8.set_ylabel('Frequency', fontsize=11)
    ax8.set_title('Reward Error Distribution', fontsize=13, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # === Plot 9: Key Metrics Summary ===
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # Compute inference time statistics if available
    if step_inference_times is not None and len(step_inference_times) > 0:
        avg_step_time = np.mean(step_inference_times)
        inference_fps = 1.0 / avg_step_time
        inference_text = f"""
    Inference Performance:
      Planning: {'ENABLED' if use_planning else 'DISABLED'}
      Avg Time per Step: {avg_step_time*1000:.2f}ms
      Inference FPS: {inference_fps:.2f} steps/sec
      Min/Max Step: {np.min(step_inference_times)*1000:.2f}/{np.max(step_inference_times)*1000:.2f}ms
    """
    else:
        inference_text = ""

    summary_text = f"""
    Key Metrics Summary
    {'='*40}

    Episode Reward Evaluation:
      Average Cumulative Reward: {np.mean(episode_rewards):.2f}
      Standard Deviation: {np.std(episode_rewards):.2f}

    Trajectory Prediction Accuracy:
      R² (Coefficient of Determination): {trajectory_results['r2']:.6f}
      rho (Pearson Correlation): {trajectory_results['rho']:.6f}
      MAPE: {trajectory_results['mape']:.2f}%

    Reward Errors:
      MAE: {trajectory_results['mae']:.6f}
      RMSE: {trajectory_results['rmse']:.6f}
      Cumulative Error: {np.sum(trajectory_results['reward_errors']):.2f}

    Position Errors:
      Average: {np.mean(trajectory_results['position_errors']):.6f}
      Maximum: {np.max(trajectory_results['position_errors']):.6f}
{inference_text}
    Test Steps: {len(trajectory_results['real_rewards'])}
    """

    ax9.text(0.1, 0.5, summary_text, fontsize=10,
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, 'test_results_comprehensive.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization results saved to: {save_path}")

    plt.show()


def main():
    """Main test workflow"""

    print("\n" + "=" * 70)
    print("World Model DQN Checkpoint Test Program")
    print("=" * 70)

    # ============================================================
    # Configuration Section - Modify here to test your checkpoint
    # ============================================================

    # 1. Project name and checkpoint
    project_name = 'trajectory_planning_u10_t'  # Match training project name
    checkpoint_episode = 'episode_300'  # Which checkpoint to test
    # Or use: checkpoint_episode = 'final'

    checkpoint_path = f'{project_name}/models/{checkpoint_episode}'

    # 2. Test parameters
    num_test_episodes = 10      # Number of test episodes
    num_trajectory_steps = 100  # Trajectory prediction test steps
    use_planning = True         # Whether to use planning (recommended True)

    # 3. Environment parameters (should match training)
    grid_size = 64
    num_users = 10
    max_steps = 100
    fixed_seed = 42

    # ============================================================

    print("\nConfiguration:")
    print(f"  Project Name: {project_name}")
    print(f"  Checkpoint Path: {checkpoint_path}")
    print(f"  Test Episodes: {num_test_episodes}")
    print(f"  Trajectory Prediction Steps: {num_trajectory_steps}")
    print(f"  Use Planning: {use_planning}")

    # === Check if checkpoint exists ===
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint path not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        models_dir = f'{project_name}/models'
        if os.path.exists(models_dir):
            checkpoints = [d for d in os.listdir(models_dir)
                          if os.path.isdir(os.path.join(models_dir, d))]
            for ckpt in sorted(checkpoints):
                print(f"  - {models_dir}/{ckpt}")
        else:
            print(f"  Model directory not found: {models_dir}")
            print("\nAvailable projects:")
            if os.path.exists(os.getcwd()):
                projects = [d for d in os.listdir(os.getcwd())
                           if os.path.isdir(d) and 'trajectory_planning' in d.lower()]
                for proj in sorted(projects):
                    print(f"  - {proj}")
        return

    print(f"Checkpoint found: {checkpoint_path}")

    # === Create environment ===
    print("\nCreating test environment...")
    env = WeatherAwareUAVEnv(
        grid_size=grid_size,
        num_users=num_users,
        max_steps=max_steps,
        user_distribution="random",
        weather_evolution="gaussian_drift",
        weather_intensity=0.7,
        weather_drift_speed=(0.3, 0.2),
        weather_noise_sigma=0.2,
        coverage_threshold=10e6,
        grid_cell_size=4,
        uav_altitude=100,
        ofdma_subcarriers=64,
        uav_initial_position=(grid_size // 2, grid_size // 2)
    )

    # Fix environment (consistent with training)
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

    env_wrapper = UAVEnvWrapper(env)

    # === Create Agent and load checkpoint ===
    print("Creating Agent and loading checkpoint...")
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
        rollout_steps=5,  # Match training settings
        planning_horizon=5,  # Match training settings
        num_planning_samples=10,
        use_planning=use_planning
    )

    # Load checkpoint
    agent.load(checkpoint_path)
    print(f"Successfully loaded checkpoint")

    # === Fill replay buffer to enable planning ===
    if use_planning:
        print("\nFilling replay buffer to enable planning...")
        print(f"Target buffer size: {agent.batch_size + 1} (minimum for planning)")

        fill_episodes = 0
        while len(agent.replay_buffer) <= agent.batch_size:
            state = env_wrapper.reset()
            for step in range(max_steps):
                # Use Q-network to select action (planning not available yet)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = agent.q_network(state_tensor)
                    action = q_values.argmax(dim=1).item()

                next_state, reward, done, _ = env_wrapper.step(action)
                agent.replay_buffer.add(state, action, reward, next_state, done)

                state = next_state
                if done or len(agent.replay_buffer) > agent.batch_size:
                    break

            fill_episodes += 1

        print(f"Buffer filled with {len(agent.replay_buffer)} experiences from {fill_episodes} episodes")
        print(f"Planning is now ENABLED for testing\n")

    # === Test 1: Episode Reward Evaluation ===
    episode_rewards, episode_inference_times, step_inference_times = test_episode_rewards(
        agent=agent,
        env_wrapper=env_wrapper,
        num_episodes=num_test_episodes,
        max_steps=max_steps,
        use_planning=use_planning
    )

    # === Test 2: Trajectory Prediction Accuracy Evaluation ===
    test_results_dir = f'{project_name}/test_results'
    trajectory_results = test_trajectory_prediction(
        agent=agent,
        env_wrapper=env_wrapper,
        num_steps=num_trajectory_steps,
        seed=fixed_seed,
        save_dir=test_results_dir
    )

    # === Visualize results ===
    print("\nGenerating visualization plots...")
    visualize_results(episode_rewards, trajectory_results, save_dir=test_results_dir,
                     step_inference_times=step_inference_times, use_planning=use_planning)

    # === Save numerical results ===
    results_file = f'{test_results_dir}/metrics.txt'
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("World Model DQN Test Results\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Time: {np.datetime64('now')}\n")
        f.write(f"Planning: {'ENABLED' if use_planning else 'DISABLED'}\n\n")

        f.write("Episode Reward Evaluation:\n")
        f.write(f"  Average Cumulative Reward: {np.mean(episode_rewards):.2f}\n")
        f.write(f"  Standard Deviation: {np.std(episode_rewards):.2f}\n")
        f.write(f"  Min/Max: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}\n\n")

        f.write("Inference Time Statistics:\n")
        f.write(f"  Total Inference Time: {np.sum(episode_inference_times):.3f}s\n")
        f.write(f"  Average Time per Episode: {np.mean(episode_inference_times):.3f}s\n")
        f.write(f"  Average Time per Step: {np.mean(step_inference_times)*1000:.2f}ms\n")
        f.write(f"  Min/Max Step Time: {np.min(step_inference_times)*1000:.2f}ms / {np.max(step_inference_times)*1000:.2f}ms\n")
        f.write(f"  Inference FPS: {1.0/np.mean(step_inference_times):.2f} steps/second\n\n")

        f.write("Trajectory Prediction Accuracy:\n")
        f.write(f"  R² (Coefficient of Determination): {trajectory_results['r2']:.6f}\n")
        f.write(f"  rho (Pearson Correlation Coefficient): {trajectory_results['rho']:.6f}\n")
        f.write(f"  p-value: {trajectory_results['p_value']:.6e}\n")
        f.write(f"  MAPE (Mean Absolute Percentage Error): {trajectory_results['mape']:.2f}%\n")
        f.write(f"  MAE (Mean Absolute Error): {trajectory_results['mae']:.6f}\n")
        f.write(f"  RMSE (Root Mean Square Error): {trajectory_results['rmse']:.6f}\n\n")

        f.write("Cumulative Rewards:\n")
        f.write(f"  Real: {np.sum(trajectory_results['real_rewards']):.2f}\n")
        f.write(f"  Predicted: {np.sum(trajectory_results['pred_rewards']):.2f}\n")
        f.write(f"  Error: {abs(np.sum(trajectory_results['real_rewards']) - np.sum(trajectory_results['pred_rewards'])):.2f}\n\n")

        f.write("Position Prediction:\n")
        f.write(f"  Average Error: {np.mean(trajectory_results['position_errors']):.6f} grid cells\n")
        f.write(f"  Maximum Error: {np.max(trajectory_results['position_errors']):.6f} grid cells\n")

    print(f"\nNumerical results saved to: {results_file}")

    # === Save inference time data to CSV ===
    inference_csv_file = f'{test_results_dir}/inference_times.csv'
    with open(inference_csv_file, 'w') as f:
        f.write('step,inference_time_ms\n')
        for i, t in enumerate(step_inference_times):
            f.write(f'{i},{t*1000:.4f}\n')
    print(f"Inference time data saved to: {inference_csv_file}")

    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)
    print("\nGenerated Files:")
    print(f"  1. Visualization Plots: {test_results_dir}/test_results_comprehensive.png")
    print(f"  2. Numerical Metrics: {test_results_dir}/metrics.txt")
    print(f"  3. Detailed Data: {test_results_dir}/prediction_details.txt")
    print(f"  4. Prediction CSV: {test_results_dir}/prediction_data.csv")
    print(f"  5. Inference Time CSV: {test_results_dir}/inference_times.csv")


if __name__ == "__main__":
    main()
