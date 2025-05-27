import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter


class WeatherAwareUAVEnv(gym.Env):
    """
    Weather-Aware UAV Communication Coverage Environment

    A UAV acts as a mobile aerial base station to provide communication coverage
    for ground users in a weather-changing environment. The goal is to maximize
    average data rate in an OFDMA scenario.

    Each grid cell represents a 4×4 meter area in the real world, making the
    total coverage area 256×256 meters. The UAV flies at a constant altitude of 100 meters.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 grid_size=64,
                 num_users=10,
                 max_steps=100,
                 user_distribution="random",  # "random" or "grid"
                 weather_evolution="gaussian_drift",  # Weather evolution mode
                 weather_intensity=0.7,  # Weather intensity
                 weather_drift_speed=(0.5, 0.5),  # Weather movement speed (x, y)
                 weather_noise_sigma=0.05,  # Weather disturbance strength
                 uav_initial_position="random",  # "random" or (x, y)
                 include_energy_penalty=False,  # Whether to include energy penalty
                 energy_penalty_lambda=0.1,  # Energy penalty coefficient
                 coverage_threshold=10e6,  # Coverage threshold in bps (10 Mbps default)
                 grid_cell_size=4,  # Each grid cell is 4×4 meters
                 uav_altitude=100,  # UAV flies at a constant altitude of 100 meters
                 ofdma_subcarriers=64,  # Number of OFDMA subcarriers
                 render_mode=None):

        # Environment parameters
        self.grid_size = grid_size  # Grid size
        self.num_users = num_users  # Number of users
        self.max_steps = max_steps  # Maximum steps
        self.user_distribution = user_distribution
        self.weather_evolution = weather_evolution
        self.weather_intensity = weather_intensity
        self.weather_drift_speed = weather_drift_speed
        self.weather_noise_sigma = weather_noise_sigma
        self.uav_initial_position = uav_initial_position
        self.include_energy_penalty = include_energy_penalty
        self.energy_penalty_lambda = energy_penalty_lambda
        self.grid_cell_size = grid_cell_size  # Size of each grid cell in meters
        self.uav_altitude = uav_altitude  # UAV altitude in meters
        self.ofdma_subcarriers = ofdma_subcarriers  # Number of OFDMA subcarriers

        # Communication parameters
        self.frequency = 28e9  # 28 GHz
        self.bandwidth = 100e6  # 100 MHz
        self.subcarrier_bandwidth = self.bandwidth / self.ofdma_subcarriers  # Bandwidth per subcarrier
        self.tx_power_dBm = 30  # 30 dBm
        self.tx_power = 10 ** (self.tx_power_dBm / 10) / 1000  # Convert to Watts
        self.subcarrier_power = self.tx_power / self.ofdma_subcarriers  # Power per subcarrier
        self.noise_power_dBm = -94  # -94 dBm
        self.noise_power = 10 ** (self.noise_power_dBm / 10) / 1000  # Convert to Watts
        self.path_loss_exponent = 2.5  # Path loss exponent
        self.base_path_loss = 60  # Base path loss (dB)
        self.weather_sensitivity = 10  # Weather sensitivity factor
        self.coverage_threshold = coverage_threshold  # Threshold for visualization only

        # Action space: (dx, dy) ∈ {-1, 0, 1}²
        self.action_space = spaces.Discrete(9)

        # Observation space: UAV position + channel quality indicators for all users
        self.observation_space = spaces.Box(
            low=np.array([0, 0] + [0] * self.num_users),
            high=np.array([grid_size, grid_size] + [float('inf')] * self.num_users),
            dtype=np.float32
        )

        # Rendering settings
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Initialize environment
        self.reset()

    def _get_obs(self):
        """Get current observation"""
        # UAV position + channel quality for all users
        channel_gains = self._compute_channel_gains()

        # Channel quality in dB is more reasonable
        channel_gains_db = 10 * np.log10(channel_gains)

        return np.concatenate([self.uav_position, channel_gains_db])

    def _get_info(self):
        """Get additional information"""
        # Calculate channel gains and data rates for all users
        channel_gains = self._compute_channel_gains()
        data_rates = self._compute_data_rates(channel_gains)

        # Calculate number of covered users (for visualization only)
        covered_users = np.sum(data_rates > self.coverage_threshold)

        # Calculate average data rate
        avg_data_rate = np.mean(data_rates)

        # Calculate total data rate
        total_data_rate = np.sum(data_rates)

        return {
            "uav_position": self.uav_position.copy(),
            "user_positions": self.user_positions.copy(),
            "weather_map": self.weather_map.copy(),
            "channel_gains": channel_gains,
            "data_rates": data_rates,
            "covered_users": covered_users,  # Keep for visualization
            "avg_data_rate": avg_data_rate,
            "total_data_rate": total_data_rate,
        }

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)

        # Initialize time step
        self.current_step = 0

        # Initialize UAV position
        if self.uav_initial_position == "random":
            self.uav_position = np.array([
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size)
            ], dtype=np.float32)
        else:
            self.uav_position = np.array(self.uav_initial_position, dtype=np.float32)

        # Initialize user positions
        self._initialize_users()

        # Initialize weather map
        self._initialize_weather()

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Create visualization directory
        self.visualization_dir = os.path.join(os.getcwd(), 'uav_weather_visualizations')
        os.makedirs(self.visualization_dir, exist_ok=True)

        return observation, info

    def step(self, action):
        """Execute one step action"""
        # Parse action
        dx, dy = self._action_to_movement(action)

        # Record previous position
        previous_position = self.uav_position.copy()

        # Update UAV position
        self.uav_position[0] = np.clip(self.uav_position[0] + dx, 0, self.grid_size - 1)
        self.uav_position[1] = np.clip(self.uav_position[1] + dy, 0, self.grid_size - 1)

        # Calculate movement distance
        movement = np.linalg.norm(self.uav_position - previous_position)

        # Update weather map
        self._update_weather()

        # Calculate current channel gains and data rates
        channel_gains = self._compute_channel_gains()
        data_rates = self._compute_data_rates(channel_gains)

        # Calculate reward: average data rate across all users (scaled for stability)
        reward = np.mean(data_rates) / 1e6  # Scale to Mbps for stability

        # Optional: Add energy penalty
        if self.include_energy_penalty:
            energy_penalty = self.energy_penalty_lambda * movement ** 2
            reward -= energy_penalty

        # Update current step
        self.current_step += 1

        # Check if done
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render environment"""
        plt.figure(figsize=(12, 10))

        # 1. Plot weather map
        plt.subplot(2, 2, 1)
        weather_map = self.weather_map.copy()
        plt.imshow(weather_map, cmap='Blues', origin='lower', vmin=0, vmax=1)
        plt.title('Weather Map (Rainfall Intensity)')
        plt.colorbar(label='Intensity')

        # Mark UAV position on weather map
        plt.plot(self.uav_position[0], self.uav_position[1], 'ro', markersize=8, label='UAV')

        # Mark user positions on weather map
        for i, user_pos in enumerate(self.user_positions):
            plt.plot(user_pos[0], user_pos[1], 'go', markersize=5)
            plt.text(user_pos[0] + 0.5, user_pos[1] + 0.5, f"{i}", fontsize=8)

        plt.legend()

        # 2. Plot data rate map (similar to coverage map)
        plt.subplot(2, 2, 2)
        data_rate_map = np.zeros((self.grid_size, self.grid_size))

        # Calculate data rate at each point on the grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Calculate 3D distance from UAV to this point (including altitude)
                horizontal_distance = np.sqrt(((x - self.uav_position[0]) * self.grid_cell_size) ** 2 +
                                              ((y - self.uav_position[1]) * self.grid_cell_size) ** 2)
                d = np.sqrt(horizontal_distance ** 2 + self.uav_altitude ** 2)  # 3D distance

                # Get weather effect at this point
                weather_effect = self.weather_map[int(y), int(x)]

                # Calculate path loss
                path_loss_db = self.base_path_loss + 10 * self.path_loss_exponent * np.log10(
                    d) + self.weather_sensitivity * weather_effect

                # Calculate channel gain
                channel_gain = 10 ** (-path_loss_db / 10)

                # Calculate data rate (assume single user gets all subcarriers at this point)
                snr = self.tx_power * channel_gain / self.noise_power
                data_rate = self.bandwidth * np.log2(1 + snr)

                data_rate_map[y, x] = data_rate

        # Plot data rate map
        data_rate_cmap = LinearSegmentedColormap.from_list('data_rate_cmap', ['white', 'green'])
        plt.imshow(data_rate_map, cmap=data_rate_cmap, origin='lower', vmin=0,
                   vmax=np.max(data_rate_map) if np.max(data_rate_map) > 0 else 1)
        plt.title('Data Rate Map')
        plt.colorbar(label='Data rate (bps)')

        # Mark UAV position
        plt.plot(self.uav_position[0], self.uav_position[1], 'ro', markersize=8, label='UAV')

        # Mark user positions
        channel_gains = self._compute_channel_gains()
        data_rates = self._compute_data_rates(channel_gains)
        covered_users = (data_rates > self.coverage_threshold)  # Just for visualization

        for i, (user_pos, is_covered) in enumerate(zip(self.user_positions, covered_users)):
            color = 'g' if is_covered else 'r'
            plt.plot(user_pos[0], user_pos[1], f'{color}o', markersize=5)
            plt.text(user_pos[0] + 0.5, user_pos[1] + 0.5, f"{i}", fontsize=8)

        plt.legend()

        # 3. Plot user data rates
        plt.subplot(2, 2, 3)
        bars = plt.bar(range(self.num_users), data_rates / 1e6)  # Convert to Mbps

        # Set color based on coverage (for visualization only)
        for i, (bar, is_covered) in enumerate(zip(bars, covered_users)):
            bar.set_color('green' if is_covered else 'red')

        plt.axhline(y=self.coverage_threshold / 1e6, color='r', linestyle='--',
                    label=f'Threshold ({self.coverage_threshold / 1e6} Mbps)')
        plt.xlabel('User Index')
        plt.ylabel('Data Rate (Mbps)')
        plt.title(f'User Data Rates (Avg: {np.mean(data_rates) / 1e6:.2f} Mbps)')
        plt.legend()

        # 4. Plot cumulative statistics
        plt.subplot(2, 2, 4)

        # If no cumulative stats exist, create them
        if not hasattr(self, 'history_rewards'):
            self.history_rewards = []
            self.history_avg_data_rates = []
            self.history_total_data_rates = []

        # Calculate current metrics and add to history
        if len(self.history_rewards) < self.current_step:
            # Average data rate reward
            reward = np.mean(data_rates) / 1e6  # Scaled to Mbps

            # Optional: Add energy penalty
            if self.include_energy_penalty and self.current_step > 0:
                # Simplified handling, use fixed penalty
                energy_penalty = self.energy_penalty_lambda * 1.0
                reward -= energy_penalty

            self.history_rewards.append(reward)
            self.history_avg_data_rates.append(np.mean(data_rates) / 1e6)  # Convert to Mbps
            self.history_total_data_rates.append(np.sum(data_rates) / 1e6)  # Convert to Mbps

        # Plot history metrics
        plt.plot(self.history_rewards, 'b-', label='Reward')
        plt.plot(self.history_avg_data_rates, 'g-', label='Avg Data Rate (Mbps)')
        plt.plot(self.history_total_data_rates, 'r-', label='Total Capacity (Mbps)')
        plt.xlabel('Time Step')
        plt.title(f'Performance Metrics (Step {self.current_step}/{self.max_steps})')
        plt.legend()

        plt.tight_layout()

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"uav_step_{self.current_step:03d}.png"
        plt.savefig(os.path.join(self.visualization_dir, filename))
        plt.close()

    def close(self):
        """Close environment"""
        if self.window is not None:
            plt.close()

    def _initialize_users(self):
        """Initialize user positions"""
        if self.user_distribution == "grid":
            # Grid distribution
            grid_points = np.linspace(0, self.grid_size - 1, int(np.ceil(np.sqrt(self.num_users))))
            x, y = np.meshgrid(grid_points, grid_points)
            positions = np.column_stack([x.ravel(), y.ravel()])

            # If generated points are more than needed users, randomly select some
            if len(positions) > self.num_users:
                indices = self.np_random.choice(len(positions), self.num_users, replace=False)
                positions = positions[indices]

            self.user_positions = positions
        else:
            # Random distribution
            self.user_positions = np.vstack([
                self.np_random.uniform(0, self.grid_size, self.num_users),
                self.np_random.uniform(0, self.grid_size, self.num_users)
            ]).T

    def _initialize_weather(self):
        """Initialize weather map"""
        if self.weather_evolution == "gaussian_drift":
            # Create an empty weather map
            self.weather_map = np.zeros((self.grid_size, self.grid_size))

            # Create Gaussian hotspot
            center_x = self.np_random.uniform(0, self.grid_size)
            center_y = self.np_random.uniform(0, self.grid_size)

            self.weather_center = np.array([center_x, center_y])

            # Set Gaussian kernel size
            sigma = self.grid_size / 5.0

            # Generate coordinates for each grid point
            x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))

            # Calculate Gaussian hotspot
            self.weather_map = self.weather_intensity * np.exp(
                -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2)
            )

            # Smooth weather map
            self.weather_map = gaussian_filter(self.weather_map, sigma=1.0)
        else:
            # Default simple random weather
            self.weather_map = self.np_random.uniform(0, self.weather_intensity, (self.grid_size, self.grid_size))

    def _update_weather(self):
        """Update weather map"""
        if self.weather_evolution == "gaussian_drift":
            # Update weather center
            dx, dy = self.weather_drift_speed

            # Add randomness
            dx += self.np_random.normal(0, 0.2)
            dy += self.np_random.normal(0, 0.2)

            self.weather_center[0] += dx
            self.weather_center[1] += dy

            # Boundary handling (optional, let weather center bounce at boundary)
            if self.weather_center[0] < 0 or self.weather_center[0] >= self.grid_size:
                dx = -dx
                self.weather_center[0] = np.clip(self.weather_center[0], 0, self.grid_size - 1)

            if self.weather_center[1] < 0 or self.weather_center[1] >= self.grid_size:
                dy = -dy
                self.weather_center[1] = np.clip(self.weather_center[1], 0, self.grid_size - 1)

            # Update drift speed
            self.weather_drift_speed = (dx, dy)

            # Regenerate Gaussian hotspot
            center_x, center_y = self.weather_center
            sigma = self.grid_size / 5.0

            # Generate coordinates for each grid point
            x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))

            # Calculate Gaussian hotspot
            new_weather_map = self.weather_intensity * np.exp(
                -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2)
            )

            # Add spatial disturbance
            noise = self.np_random.normal(0, self.weather_noise_sigma, (self.grid_size, self.grid_size))
            new_weather_map += noise

            # Smooth weather map
            new_weather_map = gaussian_filter(new_weather_map, sigma=1.0)

            # Limit weather intensity to [0,1] range
            self.weather_map = np.clip(new_weather_map, 0, 1)
        else:
            # Default simple random disturbance
            noise = self.np_random.normal(0, self.weather_noise_sigma, (self.grid_size, self.grid_size))
            self.weather_map += noise
            self.weather_map = np.clip(self.weather_map, 0, 1)

    def _compute_channel_gains(self):
        """Calculate channel gains for all users"""
        channel_gains = np.zeros(self.num_users)

        for i, user_pos in enumerate(self.user_positions):
            # Calculate 3D distance from UAV to user (including altitude)
            horizontal_distance = np.sqrt(((self.uav_position[0] - user_pos[0]) * self.grid_cell_size) ** 2 +
                                          ((self.uav_position[1] - user_pos[1]) * self.grid_cell_size) ** 2)
            d = np.sqrt(horizontal_distance ** 2 + self.uav_altitude ** 2)  # Total 3D distance

            # Get weather effect at UAV position
            weather_effect = self._get_weather_at_position(self.uav_position)

            # Calculate path loss (dB)
            path_loss_db = self.base_path_loss + 10 * self.path_loss_exponent * np.log10(
                d) + self.weather_sensitivity * weather_effect

            # Convert to channel gain
            channel_gains[i] = 10 ** (-path_loss_db / 10)

        return channel_gains

    # def _compute_data_rates(self, channel_gains):
    #     """Calculate data rates for all users in OFDMA scenario"""
    #
    #     # Normalize channel gains to get resource allocation proportions
    #     if np.sum(channel_gains) > 0:
    #         allocation_proportions = channel_gains / np.sum(channel_gains)
    #     else:
    #         # Equal allocation if all channel gains are zero (should never happen)
    #         allocation_proportions = np.ones(self.num_users) / self.num_users
    #
    #     # Allocate subcarriers proportional to channel quality
    #     subcarrier_allocations = np.round(allocation_proportions * self.ofdma_subcarriers)
    #
    #     # Ensure total allocation doesn't exceed available subcarriers
    #     while np.sum(subcarrier_allocations) > self.ofdma_subcarriers:
    #         idx = np.argmax(subcarrier_allocations)
    #         subcarrier_allocations[idx] -= 1
    #
    #     # Ensure each user gets at least one subcarrier if possible
    #     zeros_idx = np.where(subcarrier_allocations == 0)[0]
    #     if len(zeros_idx) > 0 and np.sum(subcarrier_allocations) < self.ofdma_subcarriers:
    #         remaining = self.ofdma_subcarriers - np.sum(subcarrier_allocations)
    #         assign_idx = zeros_idx[:min(len(zeros_idx), int(remaining))]
    #         subcarrier_allocations[assign_idx] = 1
    #
    #     # Calculate data rate for each user
    #     data_rates = np.zeros(self.num_users)
    #     for i in range(self.num_users):
    #         if subcarrier_allocations[i] > 0:
    #             # Power allocated to this user
    #             allocated_power = self.subcarrier_power * subcarrier_allocations[i]
    #
    #             # Bandwidth allocated to this user
    #             allocated_bandwidth = self.subcarrier_bandwidth * subcarrier_allocations[i]
    #
    #             # SNR for this user
    #             snr = allocated_power * channel_gains[i] / self.noise_power
    #
    #             # Data rate (bps) using Shannon capacity formula
    #             data_rates[i] = allocated_bandwidth * np.log2(1 + snr)
    #
    #     return data_rates

    def _compute_data_rates(self, channel_gains):
        """Calculate data rates for all users in OFDMA scenario with uniform resource allocation"""
        # Uniform allocation of subcarriers regardless of channel gains
        uniform_allocation = np.ones(self.num_users) / self.num_users

        # Allocate subcarriers equally among users
        subcarrier_allocations = np.floor(uniform_allocation * self.ofdma_subcarriers)

        # Distribute remaining subcarriers (if any)
        remaining = self.ofdma_subcarriers - np.sum(subcarrier_allocations)
        if remaining > 0:
            # Assign remaining subcarriers one by one to users in order
            idx = 0
            while remaining > 0:
                subcarrier_allocations[idx % self.num_users] += 1
                idx += 1
                remaining -= 1

        # Calculate data rate for each user
        data_rates = np.zeros(self.num_users)
        for i in range(self.num_users):
            if subcarrier_allocations[i] > 0:
                # Power allocated to this user
                allocated_power = self.subcarrier_power * subcarrier_allocations[i]

                # Bandwidth allocated to this user
                allocated_bandwidth = self.subcarrier_bandwidth * subcarrier_allocations[i]

                # SNR for this user
                snr = allocated_power * channel_gains[i] / self.noise_power

                # Data rate (bps) using Shannon capacity formula
                data_rates[i] = allocated_bandwidth * np.log2(1 + snr)

        return data_rates


    def _get_weather_at_position(self, position):
        """Get weather condition at specified position"""
        x, y = position

        # Ensure indices are integers and within valid range
        x = int(np.clip(x, 0, self.grid_size - 1))
        y = int(np.clip(y, 0, self.grid_size - 1))

        return self.weather_map[y, x]

    def _action_to_movement(self, action):
        """Convert discrete action to movement vector"""
        # Action space: 0-8 corresponding to 9 possible movement directions
        # 0: No movement (0,0)
        # 1: Up (0,1)
        # 2: Up-right (1,1)
        # 3: Right (1,0)
        # 4: Down-right (1,-1)
        # 5: Down (0,-1)
        # 6: Down-left (-1,-1)
        # 7: Left (-1,0)
        # 8: Up-left (-1,1)

        movements = [
            (0, 0),  # No movement
            (0, 1),  # Up
            (1, 1),  # Up-right
            (1, 0),  # Right
            (1, -1),  # Down-right
            (0, -1),  # Down
            (-1, -1),  # Down-left
            (-1, 0),  # Left
            (-1, 1)  # Up-left
        ]

        return movements[action]