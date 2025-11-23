"""
Learning Sequence Script for Drone AI.

Implements curriculum learning for progressive skill acquisition:
1. Basic hovering and stability
2. Point-to-point navigation
3. Single delivery execution
4. Multi-delivery missions
5. Priority optimization and battery management

Usage:
    python -m drone_ai.learning_sequence --render --steps-per-age 10000
"""

import argparse
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from .environment import DroneDeliveryEnv


@dataclass
class CurriculumStage:
    """Definition of a curriculum learning stage."""
    name: str
    description: str
    difficulty: float
    num_deliveries: int
    success_threshold: float  # Success rate needed to advance
    max_steps: int


class LearningAgent:
    """
    Simple learning agent for demonstration.

    In production, this would be replaced with a proper RL algorithm
    like PPO, SAC, or similar.
    """

    def __init__(self, obs_dim: int, action_dim: int, learning_rate: float = 0.001):
        """
        Initialize the learning agent.

        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate for updates
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Simple linear policy (placeholder)
        self.weights = np.random.randn(action_dim, obs_dim) * 0.1
        self.bias = np.ones(action_dim) * 0.5  # Start with hovering

        # Exploration parameters
        self.exploration_std = 0.3
        self.min_exploration = 0.05

        # Experience buffer for simple learning
        self.experience_buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.buffer_size = 1000

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select an action given observation.

        Args:
            obs: Current observation
            explore: Whether to add exploration noise

        Returns:
            Action array
        """
        # Linear policy
        action = np.dot(self.weights, obs) + self.bias

        # Apply sigmoid for bounded output
        action = 1 / (1 + np.exp(-action))

        # Add exploration noise
        if explore:
            noise = np.random.randn(self.action_dim) * self.exploration_std
            action = action + noise

        return np.clip(action, 0.0, 1.0).astype(np.float32)

    def store_experience(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float
    ) -> None:
        """
        Store experience for learning.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
        """
        self.experience_buffer.append((obs, action, reward))
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)

    def learn(self) -> float:
        """
        Update policy based on collected experience.

        Uses simple policy gradient-like updates.

        Returns:
            Average reward of batch
        """
        if len(self.experience_buffer) < 100:
            return 0.0

        # Sample batch
        batch_size = min(64, len(self.experience_buffer))
        indices = np.random.choice(
            len(self.experience_buffer), batch_size, replace=False
        )

        total_reward = 0
        for idx in indices:
            obs, action, reward = self.experience_buffer[idx]
            total_reward += reward

            # Simple gradient update (reward-weighted)
            predicted = 1 / (1 + np.exp(-(np.dot(self.weights, obs) + self.bias)))
            gradient = np.outer(action - predicted, obs)

            # Update weights based on reward
            self.weights += self.learning_rate * reward * gradient * 0.01
            self.bias += self.learning_rate * reward * (action - predicted) * 0.01

        return total_reward / batch_size

    def decay_exploration(self, factor: float = 0.995) -> None:
        """Decay exploration rate."""
        self.exploration_std = max(
            self.min_exploration,
            self.exploration_std * factor
        )


def run_curriculum_learning(
    render: bool = True,
    steps_per_stage: int = 10000,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run curriculum learning sequence.

    Args:
        render: Whether to show visualization
        steps_per_stage: Training steps per curriculum stage
        seed: Random seed
        verbose: Whether to print progress

    Returns:
        Training statistics
    """
    # Define curriculum stages
    stages = [
        CurriculumStage(
            name="Hovering",
            description="Learn to hover stably",
            difficulty=0.0,
            num_deliveries=1,
            success_threshold=0.3,
            max_steps=500
        ),
        CurriculumStage(
            name="Navigation",
            description="Navigate to targets",
            difficulty=0.2,
            num_deliveries=1,
            success_threshold=0.5,
            max_steps=1000
        ),
        CurriculumStage(
            name="Single Delivery",
            description="Complete single deliveries",
            difficulty=0.4,
            num_deliveries=1,
            success_threshold=0.6,
            max_steps=1500
        ),
        CurriculumStage(
            name="Multi-Delivery",
            description="Handle multiple deliveries",
            difficulty=0.5,
            num_deliveries=3,
            success_threshold=0.5,
            max_steps=2000
        ),
        CurriculumStage(
            name="Advanced Missions",
            description="Optimize multi-delivery with battery management",
            difficulty=0.7,
            num_deliveries=5,
            success_threshold=0.4,
            max_steps=2000
        ),
    ]

    # Initialize agent
    agent = LearningAgent(obs_dim=31, action_dim=4)

    # Statistics tracking
    stats = {
        'stages_completed': 0,
        'total_episodes': 0,
        'total_steps': 0,
        'stage_results': []
    }

    render_mode = "human" if render else None

    if verbose:
        print("\n" + "=" * 70)
        print("DRONE AI CURRICULUM LEARNING")
        print("=" * 70)
        print(f"Steps per stage: {steps_per_stage}")
        print(f"Number of stages: {len(stages)}")
        print("=" * 70 + "\n")

    for stage_idx, stage in enumerate(stages):
        if verbose:
            print(f"\n{'='*50}")
            print(f"STAGE {stage_idx + 1}: {stage.name}")
            print(f"{'='*50}")
            print(f"Description: {stage.description}")
            print(f"Difficulty: {stage.difficulty}")
            print(f"Deliveries: {stage.num_deliveries}")
            print(f"Success threshold: {stage.success_threshold:.0%}")
            print(f"{'='*50}\n")

        # Create environment for this stage
        env = DroneDeliveryEnv(
            task="delivery_route",
            difficulty=stage.difficulty,
            num_deliveries=stage.num_deliveries,
            render_mode=render_mode
        )

        stage_stats = {
            'name': stage.name,
            'episodes': 0,
            'steps': 0,
            'total_reward': 0,
            'successful_deliveries': 0,
            'total_deliveries': 0,
            'success_rate': 0.0,
            'passed': False
        }

        steps_done = 0
        recent_success: List[float] = []

        while steps_done < steps_per_stage:
            # Reset environment
            obs, info = env.reset(
                seed=seed + stats['total_episodes'] if seed else None
            )

            episode_reward = 0
            episode_steps = 0

            while episode_steps < stage.max_steps:
                # Get action from agent
                action = agent.act(obs, explore=True)

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)

                # Store experience
                agent.store_experience(obs, action, reward)

                episode_reward += reward
                episode_steps += 1
                steps_done += 1

                obs = next_obs

                # Render
                if render:
                    env.render()
                    time.sleep(0.005)

                # Learn periodically
                if steps_done % 50 == 0:
                    agent.learn()
                    agent.decay_exploration()

                if terminated or truncated:
                    break

            # Episode complete
            stage_stats['episodes'] += 1
            stage_stats['steps'] += episode_steps
            stage_stats['total_reward'] += episode_reward
            stage_stats['successful_deliveries'] += info['deliveries_successful']
            stage_stats['total_deliveries'] += info['deliveries_completed']

            stats['total_episodes'] += 1
            stats['total_steps'] += episode_steps

            # Track recent success rate
            if info['deliveries_completed'] > 0:
                ep_success = info['deliveries_successful'] / info['deliveries_completed']
            else:
                ep_success = 0.0
            recent_success.append(ep_success)
            if len(recent_success) > 20:
                recent_success.pop(0)

            # Progress update
            if verbose and stage_stats['episodes'] % 5 == 0:
                avg_success = np.mean(recent_success) if recent_success else 0
                print(
                    f"  Episode {stage_stats['episodes']}: "
                    f"Steps={episode_steps}, "
                    f"Reward={episode_reward:.1f}, "
                    f"Deliveries={info['deliveries_successful']}/{info['deliveries_completed']}, "
                    f"Avg Success={avg_success:.1%}"
                )

            # Check if stage passed
            if len(recent_success) >= 10:
                avg_success = np.mean(recent_success)
                if avg_success >= stage.success_threshold:
                    stage_stats['passed'] = True
                    if verbose:
                        print(f"\n  Stage PASSED! Success rate: {avg_success:.1%}")
                    break

        env.close()

        # Stage complete
        if stage_stats['total_deliveries'] > 0:
            stage_stats['success_rate'] = (
                stage_stats['successful_deliveries'] /
                stage_stats['total_deliveries']
            )

        stats['stage_results'].append(stage_stats)

        if stage_stats['passed']:
            stats['stages_completed'] += 1
        else:
            if verbose:
                print(f"\n  Stage not passed. Moving to next stage anyway...")
            # Continue to next stage even if not passed (for demo purposes)
            stats['stages_completed'] += 1

    # Final summary
    if verbose:
        print("\n" + "=" * 70)
        print("CURRICULUM LEARNING COMPLETE")
        print("=" * 70)
        print(f"Stages completed: {stats['stages_completed']}/{len(stages)}")
        print(f"Total episodes: {stats['total_episodes']}")
        print(f"Total steps: {stats['total_steps']}")
        print("\nStage Results:")
        for result in stats['stage_results']:
            status = "PASSED" if result['passed'] else "INCOMPLETE"
            print(
                f"  {result['name']}: {status} "
                f"(Success rate: {result['success_rate']:.1%}, "
                f"Episodes: {result['episodes']})"
            )
        print("=" * 70 + "\n")

    return stats


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Drone AI Curriculum Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m drone_ai.learning_sequence --render
  python -m drone_ai.learning_sequence --steps-per-age 20000 --no-render
  python -m drone_ai.learning_sequence --seed 42
        """
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
        "--steps-per-age",
        type=int,
        default=10000,
        help="Training steps per curriculum stage (default: 10000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    render = args.render and not args.no_render
    verbose = not args.quiet

    run_curriculum_learning(
        render=render,
        steps_per_stage=args.steps_per_age,
        seed=args.seed,
        verbose=verbose
    )


if __name__ == "__main__":
    main()
