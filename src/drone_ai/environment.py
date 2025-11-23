"""
Drone AI Environment Module.

Gymnasium-compatible environment for drone delivery training with:
- Multi-delivery mission support
- Battery/energy management
- Priority-based delivery scheduling
- Mission planning integration
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

from .mission_planner import MissionPlanner, DeliveryRequest


class DroneDeliveryEnv(gym.Env):
    """
    Drone delivery environment with mission planning capabilities.

    Observation Space (31 dimensions):
        - Position (3): x, y, z coordinates
        - Velocity (3): vx, vy, vz
        - Orientation (3): roll, pitch, yaw
        - Angular velocity (3): wx, wy, wz
        - Target relative position (3): dx, dy, dz to current target
        - Target distance (1): scalar distance to target
        - Route phase (4): one-hot encoded [outbound, dropping, return, reload]
        - Delivery progress (3): deliveries_completed, deliveries_successful, route_score
        - Mission state (4): battery_level, pending_count/10, nearest_dist/100, max_priority/3
        - Reserved (4): future expansion

    Action Space (4 dimensions):
        - Motor thrust for 4 rotors (0.0 to 1.0 each)

    Phases:
        - outbound: Flying to delivery dropzone
        - dropping: Executing package drop
        - return: Returning to base (for reload or mission complete)
        - reload: At base, loading next package
    """

    # Environment constants
    GRAVITY = 9.81
    MASS = 1.0  # kg
    DT = 0.02   # 20ms timestep (50 Hz)

    # World bounds
    WORLD_SIZE = 100.0
    MAX_HEIGHT = 50.0

    # Delivery zones
    DROPZONE_RADIUS = 2.0
    BASE_RADIUS = 3.0

    # Episode limits
    MAX_STEPS = 2000
    MAX_DELIVERIES_PER_EPISODE = 5

    # Observation space dimension
    OBS_DIM = 31

    def __init__(
        self,
        task: str = "delivery_route",
        difficulty: float = 0.5,
        num_deliveries: int = 3,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the drone delivery environment.

        Args:
            task: Task type ("delivery_route" for now)
            difficulty: Difficulty level 0.0-1.0 (affects wind, distance, etc.)
            num_deliveries: Number of deliveries per episode
            render_mode: "human", "rgb_array", or None
        """
        super().__init__()

        self.task = task
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        self.num_deliveries = min(num_deliveries, self.MAX_DELIVERIES_PER_EPISODE)
        self.render_mode = render_mode

        # Action space: 4 motor thrusts
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space: 31 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )

        # Base position (home)
        self.base_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Mission planner
        self.mission_planner = MissionPlanner(
            base_position=self.base_position,
            battery_capacity=1000.0
        )

        # Battery system
        self.battery_level = 1.0
        self.battery_drain_rate = 0.0001        # Per step base drain
        self.motor_drain_multiplier = 0.5       # Extra drain per motor thrust unit

        # State variables (initialized in reset)
        self.position = None
        self.velocity = None
        self.orientation = None  # roll, pitch, yaw
        self.angular_velocity = None
        self.target_position = None
        self.current_delivery = None

        # Phase tracking
        self.route_phase = "outbound"  # outbound, dropping, return, reload
        self.phase_to_index = {
            "outbound": 0,
            "dropping": 1,
            "return": 2,
            "reload": 3
        }

        # Delivery tracking
        self.deliveries_completed = 0
        self.deliveries_successful = 0
        self.route_score = 0.0
        self.just_completed_delivery = False

        # Episode tracking
        self.steps = 0
        self.episode_reward = 0.0

        # Reward weights
        self.reward_weights = {
            # Navigation rewards
            'distance_improvement': 1.0,
            'target_reached': 50.0,
            'collision': -100.0,
            'out_of_bounds': -50.0,
            'step_penalty': -0.01,

            # Stability rewards
            'stability': 0.5,
            'smooth_control': 0.2,
            'altitude_maintenance': 0.3,

            # Delivery rewards
            'delivery_success': 100.0,
            'delivery_failed': -50.0,
            'phase_transition': 10.0,

            # Mission planning rewards
            'multi_delivery_bonus': 50.0,
            'priority_bonus': 25.0,
            'time_window_met': 30.0,
            'time_window_missed': -20.0,
            'battery_efficiency': 10.0,
            'optimal_route': 15.0,
            'return_with_battery': 5.0,

            # Battery rewards
            'battery_depleted': -200.0,
            'low_battery_warning': -5.0,
        }

        # Wind (difficulty-based)
        self.wind_strength = 0.0
        self.wind_direction = np.zeros(3)

        # Visualization
        self._renderer = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)

        # Reset position to base
        self.position = self.base_position.copy()
        self.position[2] = 1.0  # Start at 1m altitude

        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)

        # Reset battery
        self.battery_level = 1.0

        # Reset mission planner
        self.mission_planner.reset()

        # Setup multi-delivery mission
        self._setup_multi_delivery(self.num_deliveries)

        # Start with first delivery
        self.current_delivery = self.mission_planner.select_next_delivery()
        if self.current_delivery:
            self.target_position = self.current_delivery.dropzone_position.copy()
        else:
            self.target_position = self.base_position.copy()

        # Reset phase
        self.route_phase = "outbound"

        # Reset tracking
        self.deliveries_completed = 0
        self.deliveries_successful = 0
        self.route_score = 0.0
        self.just_completed_delivery = False
        self.steps = 0
        self.episode_reward = 0.0

        # Setup wind based on difficulty
        self._setup_wind()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _setup_multi_delivery(self, num_deliveries: int) -> None:
        """
        Setup multiple delivery targets for one episode.

        Args:
            num_deliveries: Number of deliveries to create
        """
        for i in range(num_deliveries):
            dropzone = self._generate_random_dropzone()
            priority = self.np_random.integers(1, 4)  # 1, 2, or 3

            # Time window for some deliveries
            time_window = None
            if self.np_random.random() < 0.3:  # 30% have time constraints
                earliest = 100.0 + i * 200.0
                latest = earliest + 400.0 + (3 - priority) * 200.0
                time_window = (earliest, latest)

            # Reward scales with priority
            reward_value = 100.0 * priority

            self.mission_planner.create_delivery(
                dropzone_position=dropzone,
                priority=priority,
                time_window=time_window,
                weight=0.5 + self.np_random.random(),  # 0.5-1.5 kg
                reward_value=reward_value
            )

    def _generate_random_dropzone(self) -> np.ndarray:
        """
        Generate a random dropzone position.

        Returns:
            Random position within world bounds
        """
        # Distance from base increases with difficulty
        min_distance = 10.0 + self.difficulty * 20.0
        max_distance = 30.0 + self.difficulty * 40.0

        angle = self.np_random.random() * 2 * np.pi
        distance = self.np_random.uniform(min_distance, max_distance)

        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        z = self.np_random.uniform(2.0, 10.0)  # 2-10m altitude

        return np.array([x, y, z], dtype=np.float32)

    def _setup_wind(self) -> None:
        """Setup wind conditions based on difficulty."""
        self.wind_strength = self.difficulty * 2.0  # 0-2 m/s
        angle = self.np_random.random() * 2 * np.pi
        self.wind_direction = np.array([
            np.cos(angle),
            np.sin(angle),
            0.0
        ], dtype=np.float32)

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Motor thrust values (4 floats, 0-1)

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1
        self.just_completed_delivery = False

        # Clamp action
        action = np.clip(action, 0.0, 1.0)

        # Apply battery drain
        self._update_battery(action)

        # Check battery depletion
        if self.battery_level <= 0:
            observation = self._get_observation()
            info = self._get_info()
            reward = self.reward_weights['battery_depleted']
            return observation, reward, True, False, info

        # Physics update
        self._physics_step(action)

        # Check bounds
        out_of_bounds = self._check_bounds()

        # Calculate reward
        reward = self._calculate_reward(action, out_of_bounds)
        self.episode_reward += reward

        # Check phase transitions and mission decisions
        self._check_phase_transitions()
        self._check_mission_decisions()

        # Update mission time
        self.mission_planner.update_time(self.DT)

        # Check termination conditions
        terminated = self._check_terminated(out_of_bounds)
        truncated = self.steps >= self.MAX_STEPS

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _update_battery(self, action: np.ndarray) -> None:
        """
        Update battery level based on motor usage.

        Args:
            action: Motor thrust values
        """
        motor_power = np.sum(action)
        drain = (
            self.battery_drain_rate +
            motor_power * self.motor_drain_multiplier * 0.0001
        )
        self.battery_level = max(0.0, self.battery_level - drain)

        # Sync with mission planner
        self.mission_planner.update_battery(self.battery_level)

    def _physics_step(self, action: np.ndarray) -> None:
        """
        Apply physics for one timestep.

        Simplified quadrotor dynamics.

        Args:
            action: Motor thrust values
        """
        # Total thrust (upward force)
        total_thrust = np.sum(action) * 4.0  # Scale factor

        # Thrust vector (affected by orientation)
        thrust_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Apply roll/pitch rotation to thrust
        roll, pitch, _ = self.orientation
        # Simplified rotation effect
        thrust_direction[0] = -np.sin(pitch)
        thrust_direction[1] = np.sin(roll)
        thrust_direction[2] = np.cos(roll) * np.cos(pitch)

        thrust_force = thrust_direction * total_thrust

        # Gravity
        gravity_force = np.array([0.0, 0.0, -self.GRAVITY * self.MASS])

        # Wind force
        wind_force = self.wind_direction * self.wind_strength

        # Total force
        total_force = thrust_force + gravity_force + wind_force

        # Acceleration
        acceleration = total_force / self.MASS

        # Update velocity (with damping)
        damping = 0.98
        self.velocity = self.velocity * damping + acceleration * self.DT

        # Update position
        self.position = self.position + self.velocity * self.DT

        # Ground collision
        if self.position[2] < 0.0:
            self.position[2] = 0.0
            self.velocity[2] = 0.0

        # Update orientation based on motor differentials
        # Front-back difference affects pitch
        pitch_torque = (action[0] + action[1] - action[2] - action[3]) * 0.1
        # Left-right difference affects roll
        roll_torque = (action[0] + action[2] - action[1] - action[3]) * 0.1
        # Diagonal difference affects yaw
        yaw_torque = (action[0] + action[3] - action[1] - action[2]) * 0.05

        torques = np.array([roll_torque, pitch_torque, yaw_torque])

        # Update angular velocity (with damping)
        angular_damping = 0.9
        self.angular_velocity = (
            self.angular_velocity * angular_damping + torques
        )

        # Update orientation
        self.orientation = self.orientation + self.angular_velocity * self.DT

        # Clamp orientation
        self.orientation = np.clip(
            self.orientation,
            [-np.pi/4, -np.pi/4, -np.pi],
            [np.pi/4, np.pi/4, np.pi]
        )

    def _check_bounds(self) -> bool:
        """
        Check if drone is out of bounds.

        Returns:
            True if out of bounds
        """
        if abs(self.position[0]) > self.WORLD_SIZE:
            return True
        if abs(self.position[1]) > self.WORLD_SIZE:
            return True
        if self.position[2] > self.MAX_HEIGHT:
            return True
        return False

    def _calculate_reward(
        self,
        action: np.ndarray,
        out_of_bounds: bool
    ) -> float:
        """
        Calculate reward for current step.

        Args:
            action: Current action
            out_of_bounds: Whether drone is out of bounds

        Returns:
            Step reward
        """
        reward = 0.0

        # Step penalty
        reward += self.reward_weights['step_penalty']

        # Out of bounds penalty
        if out_of_bounds:
            reward += self.reward_weights['out_of_bounds']
            return reward

        # Distance to target
        if self.target_position is not None:
            distance = np.linalg.norm(self.position - self.target_position)

            # Distance improvement reward
            if hasattr(self, '_last_distance'):
                improvement = self._last_distance - distance
                reward += improvement * self.reward_weights['distance_improvement']
            self._last_distance = distance

            # Target reached reward
            if distance < self.DROPZONE_RADIUS:
                reward += self.reward_weights['target_reached']

        # Stability reward (penalize large orientations)
        orientation_penalty = np.sum(np.abs(self.orientation[:2]))  # Roll/pitch
        reward += self.reward_weights['stability'] * (1.0 - orientation_penalty)

        # Smooth control reward
        action_variance = np.var(action)
        reward += self.reward_weights['smooth_control'] * (1.0 - action_variance)

        # Battery efficiency
        if self.battery_level > 0.3:
            reward += self.reward_weights['battery_efficiency'] * 0.01
        elif self.battery_level < 0.2:
            reward += self.reward_weights['low_battery_warning']

        return reward

    def _check_phase_transitions(self) -> None:
        """Check and handle phase transitions."""
        if self.target_position is None:
            return

        distance = np.linalg.norm(self.position - self.target_position)

        if self.route_phase == "outbound":
            # Check if reached dropzone
            if distance < self.DROPZONE_RADIUS:
                self.route_phase = "dropping"
                self._drop_timer = 50  # 1 second of dropping

        elif self.route_phase == "dropping":
            # Count down drop timer
            if hasattr(self, '_drop_timer'):
                self._drop_timer -= 1
                if self._drop_timer <= 0:
                    self._complete_delivery(success=True)

        elif self.route_phase == "return":
            # Check if reached base
            base_distance = np.linalg.norm(self.position - self.base_position)
            if base_distance < self.BASE_RADIUS:
                self.route_phase = "reload"
                self._reload_timer = 25  # 0.5 seconds reload

        elif self.route_phase == "reload":
            # Count down reload timer
            if hasattr(self, '_reload_timer'):
                self._reload_timer -= 1
                if self._reload_timer <= 0:
                    # Recharge battery at base
                    self.battery_level = min(1.0, self.battery_level + 0.3)
                    self.mission_planner.update_battery(self.battery_level)

                    # Get next delivery
                    next_delivery = self.mission_planner.select_next_delivery()
                    if next_delivery:
                        self.current_delivery = next_delivery
                        self.target_position = next_delivery.dropzone_position.copy()
                        self.route_phase = "outbound"
                    else:
                        # Mission complete
                        self.route_phase = "return"
                        self.target_position = None

    def _complete_delivery(self, success: bool) -> None:
        """
        Complete current delivery.

        Args:
            success: Whether delivery was successful
        """
        self.just_completed_delivery = True
        self.deliveries_completed += 1

        if success:
            self.deliveries_successful += 1

        # Get score from mission planner
        score = self.mission_planner.complete_current_delivery(success)
        self.route_score += score

    def _check_mission_decisions(self) -> None:
        """Check if mission-level decisions are needed."""
        if not self.just_completed_delivery:
            return

        # After successful delivery, decide next action
        if self.mission_planner.should_return_to_base():
            self.route_phase = "return"
            self.target_position = self.base_position.copy()
        else:
            next_delivery = self.mission_planner.select_next_delivery()
            if next_delivery:
                self.current_delivery = next_delivery
                self.target_position = next_delivery.dropzone_position.copy()
                self.route_phase = "outbound"
            else:
                # No more deliveries, return to base
                self.route_phase = "return"
                self.target_position = self.base_position.copy()

    def _check_terminated(self, out_of_bounds: bool) -> bool:
        """
        Check if episode should terminate.

        Args:
            out_of_bounds: Whether drone is out of bounds

        Returns:
            True if episode should end
        """
        # Out of bounds
        if out_of_bounds:
            return True

        # Battery depleted
        if self.battery_level <= 0:
            return True

        # Ground crash (falling too fast)
        if self.position[2] <= 0.1 and self.velocity[2] < -2.0:
            return True

        # Mission complete (at base with no pending deliveries)
        if self.route_phase == "return" and self.target_position is None:
            base_distance = np.linalg.norm(self.position - self.base_position)
            if base_distance < self.BASE_RADIUS:
                return True

        return False

    def _get_observation(self) -> np.ndarray:
        """
        Build observation vector.

        Returns:
            31-dimensional observation
        """
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Position (0-2)
        obs[0:3] = self.position / self.WORLD_SIZE

        # Velocity (3-5)
        obs[3:6] = self.velocity / 10.0  # Normalize

        # Orientation (6-8)
        obs[6:9] = self.orientation / np.pi

        # Angular velocity (9-11)
        obs[9:12] = self.angular_velocity / np.pi

        # Target relative position (12-14)
        if self.target_position is not None:
            relative = self.target_position - self.position
            obs[12:15] = relative / self.WORLD_SIZE
        else:
            obs[12:15] = np.zeros(3)

        # Target distance (15)
        if self.target_position is not None:
            distance = np.linalg.norm(self.target_position - self.position)
            obs[15] = distance / 100.0
        else:
            obs[15] = 0.0

        # Route phase one-hot (16-19)
        phase_idx = self.phase_to_index.get(self.route_phase, 0)
        obs[16 + phase_idx] = 1.0

        # Delivery progress (20-22)
        obs[20] = self.deliveries_completed / self.MAX_DELIVERIES_PER_EPISODE
        obs[21] = self.deliveries_successful / max(1, self.deliveries_completed)
        obs[22] = self.route_score / 1000.0  # Normalize

        # Mission state (23-26)
        obs[23] = self.battery_level
        obs[24] = len(self.mission_planner.state.pending_deliveries) / 10.0
        obs[25] = self.mission_planner.get_nearest_delivery_distance() / 100.0
        obs[26] = self.mission_planner.get_highest_priority() / 3.0

        # Reserved (27-30) - zeros for now

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """
        Get info dictionary.

        Returns:
            Info dict with debug/logging data
        """
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'battery_level': self.battery_level,
            'route_phase': self.route_phase,
            'deliveries_completed': self.deliveries_completed,
            'deliveries_successful': self.deliveries_successful,
            'route_score': self.route_score,
            'steps': self.steps,
            'episode_reward': self.episode_reward,
            'mission_summary': self.mission_planner.get_mission_summary(),
            'current_delivery': (
                self.current_delivery.id
                if self.current_delivery
                else None
            ),
        }

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", else None
        """
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from .visualization import DroneVisualizer
            self._renderer = DroneVisualizer(self)

        return self._renderer.render()

    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Register the environment with Gymnasium
def register_env():
    """Register the drone delivery environment."""
    gym.register(
        id='DroneDelivery-v1',
        entry_point='drone_ai.environment:DroneDeliveryEnv',
        max_episode_steps=2000,
    )
