"""
Drone AI Demo Script.

Run drone delivery simulations with visualization.

Usage:
    python -m drone_ai.demo --task delivery_route --difficulty 0.7 --render
    python -m drone_ai.demo --help
"""

import argparse
import time
import numpy as np
from typing import Optional

from .environment import DroneDeliveryEnv


def simple_controller(env: DroneDeliveryEnv, obs: np.ndarray) -> np.ndarray:
    """
    Simple proportional controller for demo purposes.

    Args:
        env: The environment
        obs: Current observation

    Returns:
        Action array (4 motor thrusts)
    """
    # Extract useful state
    position = env.position
    velocity = env.velocity
    target = env.target_position

    if target is None:
        # Hover in place
        return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    # Calculate error
    error = target - position
    distance = np.linalg.norm(error)

    # Normalize direction
    if distance > 0.1:
        direction = error / distance
    else:
        direction = np.zeros(3)

    # Proportional control gains
    kp_horizontal = 0.3
    kp_vertical = 0.5
    kd = 0.1

    # Desired velocity
    desired_vel = direction * min(distance * 0.5, 3.0)  # Cap at 3 m/s

    # Velocity error
    vel_error = desired_vel - velocity

    # Base thrust for hovering (counteract gravity)
    base_thrust = 0.5

    # Altitude control
    altitude_error = error[2]
    thrust_adjustment = altitude_error * kp_vertical - velocity[2] * kd

    # Horizontal control (differential thrust)
    roll_cmd = vel_error[1] * kp_horizontal - velocity[1] * kd
    pitch_cmd = -vel_error[0] * kp_horizontal + velocity[0] * kd

    # Convert to motor commands
    # Front-left, front-right, back-left, back-right
    motor_fl = base_thrust + thrust_adjustment - roll_cmd + pitch_cmd
    motor_fr = base_thrust + thrust_adjustment + roll_cmd + pitch_cmd
    motor_bl = base_thrust + thrust_adjustment - roll_cmd - pitch_cmd
    motor_br = base_thrust + thrust_adjustment + roll_cmd - pitch_cmd

    action = np.array([motor_fl, motor_fr, motor_bl, motor_br], dtype=np.float32)

    # Clamp
    return np.clip(action, 0.0, 1.0)


def run_demo(
    task: str = "delivery_route",
    difficulty: float = 0.5,
    num_deliveries: int = 3,
    render: bool = True,
    max_episodes: int = 3,
    seed: Optional[int] = None
) -> None:
    """
    Run the drone delivery demo.

    Args:
        task: Task type
        difficulty: Difficulty level (0.0-1.0)
        num_deliveries: Number of deliveries per episode
        render: Whether to show visualization
        max_episodes: Number of episodes to run
        seed: Random seed for reproducibility
    """
    render_mode = "human" if render else None

    env = DroneDeliveryEnv(
        task=task,
        difficulty=difficulty,
        num_deliveries=num_deliveries,
        render_mode=render_mode
    )

    print(f"\n{'='*60}")
    print("DRONE DELIVERY DEMO")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Difficulty: {difficulty}")
    print(f"Deliveries per episode: {num_deliveries}")
    print(f"Max episodes: {max_episodes}")
    print(f"Render: {render}")
    print(f"{'='*60}\n")

    total_reward = 0
    total_deliveries = 0
    total_successful = 0

    for episode in range(max_episodes):
        obs, info = env.reset(seed=seed + episode if seed else None)

        episode_reward = 0
        step = 0

        print(f"\n--- Episode {episode + 1}/{max_episodes} ---")
        print(f"Mission: {num_deliveries} deliveries to complete")

        while True:
            # Get action from simple controller
            action = simple_controller(env, obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            # Render
            if render:
                env.render()
                time.sleep(0.01)  # Slow down for visibility

            # Print progress occasionally
            if step % 200 == 0:
                battery = info['battery_level']
                phase = info['route_phase']
                completed = info['deliveries_completed']
                successful = info['deliveries_successful']
                print(
                    f"  Step {step}: Phase={phase}, "
                    f"Battery={battery:.1%}, "
                    f"Delivered={successful}/{completed}"
                )

            if terminated or truncated:
                break

        # Episode summary
        print(f"\nEpisode {episode + 1} Complete:")
        print(f"  Steps: {step}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Deliveries: {info['deliveries_successful']}/{info['deliveries_completed']}")
        print(f"  Final Battery: {info['battery_level']:.1%}")
        print(f"  Route Score: {info['route_score']:.2f}")

        total_reward += episode_reward
        total_deliveries += info['deliveries_completed']
        total_successful += info['deliveries_successful']

    # Overall summary
    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"{'='*60}")
    print(f"Total episodes: {max_episodes}")
    print(f"Average reward: {total_reward / max_episodes:.2f}")
    print(f"Total deliveries: {total_successful}/{total_deliveries}")
    if total_deliveries > 0:
        success_rate = total_successful / total_deliveries
        print(f"Success rate: {success_rate:.1%}")
    print(f"{'='*60}\n")

    env.close()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Drone AI Delivery Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m drone_ai.demo --render
  python -m drone_ai.demo --difficulty 0.8 --deliveries 5
  python -m drone_ai.demo --no-render --episodes 10
        """
    )

    parser.add_argument(
        "--task",
        type=str,
        default="delivery_route",
        help="Task type (default: delivery_route)"
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        default=0.5,
        help="Difficulty level 0.0-1.0 (default: 0.5)"
    )
    parser.add_argument(
        "--deliveries",
        type=int,
        default=3,
        help="Number of deliveries per episode (default: 3)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Enable visualization (default: True)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable visualization"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    render = args.render and not args.no_render

    run_demo(
        task=args.task,
        difficulty=args.difficulty,
        num_deliveries=args.deliveries,
        render=render,
        max_episodes=args.episodes,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
