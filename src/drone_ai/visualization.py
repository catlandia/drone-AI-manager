"""
Drone AI Visualization Module.

Provides real-time visualization of drone delivery missions including:
- 3D drone position and trajectory
- Battery status bar
- Delivery queue with priorities
- Mission progress indicators
"""

import numpy as np
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import DroneDeliveryEnv

# Try to import pygame, fall back gracefully
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class DroneVisualizer:
    """
    Visualizer for drone delivery environment.

    Provides a top-down 2D view with side panel showing:
    - Battery level
    - Delivery queue
    - Mission statistics
    """

    # Colors
    COLORS = {
        'background': (30, 30, 40),
        'grid': (50, 50, 60),
        'drone': (100, 200, 255),
        'drone_outline': (200, 230, 255),
        'base': (100, 255, 100),
        'dropzone': (255, 200, 100),
        'dropzone_critical': (255, 80, 80),
        'dropzone_urgent': (255, 180, 50),
        'dropzone_normal': (100, 200, 100),
        'path': (100, 100, 150),
        'battery_full': (100, 255, 100),
        'battery_mid': (255, 255, 100),
        'battery_low': (255, 100, 100),
        'battery_bg': (60, 60, 70),
        'text': (220, 220, 230),
        'text_dim': (150, 150, 160),
        'panel_bg': (40, 40, 50),
        'priority_critical': (255, 80, 80),
        'priority_urgent': (255, 180, 50),
        'priority_normal': (100, 200, 100),
    }

    # Display settings
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 700
    MAP_SIZE = 600
    PANEL_WIDTH = 380
    MARGIN = 10
    FONT_SIZE = 16
    FONT_SIZE_SMALL = 12

    def __init__(self, env: 'DroneDeliveryEnv'):
        """
        Initialize the visualizer.

        Args:
            env: The drone environment to visualize
        """
        self.env = env
        self.screen = None
        self.font = None
        self.font_small = None
        self.clock = None
        self.trajectory: List[np.ndarray] = []
        self.max_trajectory_points = 500

        if not PYGAME_AVAILABLE:
            print("Warning: pygame not available. Visualization disabled.")
            return

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Drone Delivery Mission")
        self.font = pygame.font.Font(None, self.FONT_SIZE)
        self.font_small = pygame.font.Font(None, self.FONT_SIZE_SMALL)
        self.clock = pygame.time.Clock()

    def render(self) -> Optional[np.ndarray]:
        """
        Render current environment state.

        Returns:
            RGB array if render_mode is "rgb_array", else None
        """
        if not PYGAME_AVAILABLE or self.screen is None:
            return None

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        # Update trajectory
        self._update_trajectory()

        # Clear screen
        self.screen.fill(self.COLORS['background'])

        # Draw map area
        self._draw_map()

        # Draw side panel
        self._draw_panel()

        # Update display
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS max

        if self.env.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

        return None

    def _update_trajectory(self) -> None:
        """Update drone trajectory history."""
        if self.env.position is not None:
            self.trajectory.append(self.env.position.copy())
            if len(self.trajectory) > self.max_trajectory_points:
                self.trajectory.pop(0)

    def _world_to_screen(self, pos: np.ndarray) -> tuple:
        """
        Convert world coordinates to screen coordinates.

        Args:
            pos: World position (x, y, z)

        Returns:
            Screen (x, y) tuple
        """
        # Map world coordinates to screen
        world_size = self.env.WORLD_SIZE * 2
        scale = (self.MAP_SIZE - 2 * self.MARGIN) / world_size

        screen_x = self.MARGIN + self.MAP_SIZE // 2 + pos[0] * scale
        screen_y = self.MARGIN + self.MAP_SIZE // 2 - pos[1] * scale  # Flip Y

        return int(screen_x), int(screen_y)

    def _draw_map(self) -> None:
        """Draw the main map area."""
        # Map background
        map_rect = pygame.Rect(
            self.MARGIN, self.MARGIN,
            self.MAP_SIZE - 2 * self.MARGIN,
            self.MAP_SIZE - 2 * self.MARGIN
        )
        pygame.draw.rect(self.screen, self.COLORS['panel_bg'], map_rect)

        # Grid
        self._draw_grid()

        # Base
        self._draw_base()

        # Delivery zones (pending)
        self._draw_delivery_zones()

        # Current target
        self._draw_target()

        # Trajectory
        self._draw_trajectory()

        # Drone
        self._draw_drone()

        # Phase indicator
        self._draw_phase_indicator()

    def _draw_grid(self) -> None:
        """Draw background grid."""
        grid_spacing = 40
        for x in range(self.MARGIN, self.MAP_SIZE - self.MARGIN, grid_spacing):
            pygame.draw.line(
                self.screen, self.COLORS['grid'],
                (x, self.MARGIN), (x, self.MAP_SIZE - self.MARGIN)
            )
        for y in range(self.MARGIN, self.MAP_SIZE - self.MARGIN, grid_spacing):
            pygame.draw.line(
                self.screen, self.COLORS['grid'],
                (self.MARGIN, y), (self.MAP_SIZE - self.MARGIN, y)
            )

    def _draw_base(self) -> None:
        """Draw the home base."""
        base_screen = self._world_to_screen(self.env.base_position)
        pygame.draw.circle(
            self.screen, self.COLORS['base'],
            base_screen, 15
        )
        pygame.draw.circle(
            self.screen, self.COLORS['text'],
            base_screen, 15, 2
        )
        # Label
        text = self.font_small.render("BASE", True, self.COLORS['text'])
        self.screen.blit(text, (base_screen[0] - 15, base_screen[1] + 18))

    def _draw_delivery_zones(self) -> None:
        """Draw pending delivery zones."""
        for delivery in self.env.mission_planner.state.pending_deliveries:
            pos_screen = self._world_to_screen(delivery.dropzone_position)

            # Color based on priority
            if delivery.priority == 3:
                color = self.COLORS['dropzone_critical']
            elif delivery.priority == 2:
                color = self.COLORS['dropzone_urgent']
            else:
                color = self.COLORS['dropzone_normal']

            # Draw zone
            radius = 8 + delivery.priority * 2
            pygame.draw.circle(self.screen, color, pos_screen, radius)
            pygame.draw.circle(self.screen, self.COLORS['text'], pos_screen, radius, 1)

            # Priority label
            text = self.font_small.render(str(delivery.priority), True, self.COLORS['text'])
            self.screen.blit(text, (pos_screen[0] - 4, pos_screen[1] - 6))

    def _draw_target(self) -> None:
        """Draw current target with indicator."""
        if self.env.target_position is None:
            return

        target_screen = self._world_to_screen(self.env.target_position)

        # Pulsing ring effect
        pulse = (np.sin(pygame.time.get_ticks() / 200) + 1) / 2
        ring_radius = 20 + pulse * 10

        pygame.draw.circle(
            self.screen, self.COLORS['dropzone'],
            target_screen, int(ring_radius), 2
        )

        # Inner marker
        pygame.draw.circle(
            self.screen, self.COLORS['dropzone'],
            target_screen, 8
        )

    def _draw_trajectory(self) -> None:
        """Draw drone's flight path."""
        if len(self.trajectory) < 2:
            return

        points = [self._world_to_screen(p) for p in self.trajectory]

        # Draw fading trail
        for i in range(len(points) - 1):
            alpha = int(255 * (i / len(points)))
            color = (
                min(255, self.COLORS['path'][0] + alpha // 2),
                min(255, self.COLORS['path'][1] + alpha // 2),
                min(255, self.COLORS['path'][2] + alpha // 2)
            )
            pygame.draw.line(self.screen, color, points[i], points[i + 1], 1)

    def _draw_drone(self) -> None:
        """Draw the drone."""
        if self.env.position is None:
            return

        pos_screen = self._world_to_screen(self.env.position)

        # Drone size based on altitude
        base_size = 12
        altitude_factor = 1 + self.env.position[2] / 20
        size = int(base_size * altitude_factor)

        # Draw drone body
        pygame.draw.circle(
            self.screen, self.COLORS['drone'],
            pos_screen, size
        )
        pygame.draw.circle(
            self.screen, self.COLORS['drone_outline'],
            pos_screen, size, 2
        )

        # Draw orientation indicator
        yaw = self.env.orientation[2] if self.env.orientation is not None else 0
        end_x = pos_screen[0] + np.cos(yaw) * (size + 5)
        end_y = pos_screen[1] - np.sin(yaw) * (size + 5)
        pygame.draw.line(
            self.screen, self.COLORS['drone_outline'],
            pos_screen, (int(end_x), int(end_y)), 3
        )

        # Altitude indicator
        alt_text = f"Alt: {self.env.position[2]:.1f}m"
        text = self.font_small.render(alt_text, True, self.COLORS['text'])
        self.screen.blit(text, (pos_screen[0] - 25, pos_screen[1] - size - 15))

    def _draw_phase_indicator(self) -> None:
        """Draw current phase indicator."""
        phase_text = f"Phase: {self.env.route_phase.upper()}"
        text = self.font.render(phase_text, True, self.COLORS['text'])
        self.screen.blit(text, (self.MARGIN + 10, self.MAP_SIZE - 30))

    def _draw_panel(self) -> None:
        """Draw the side information panel."""
        panel_x = self.MAP_SIZE + self.MARGIN
        panel_y = self.MARGIN
        panel_height = self.WINDOW_HEIGHT - 2 * self.MARGIN

        # Panel background
        panel_rect = pygame.Rect(
            panel_x, panel_y,
            self.PANEL_WIDTH - 2 * self.MARGIN, panel_height
        )
        pygame.draw.rect(self.screen, self.COLORS['panel_bg'], panel_rect)
        pygame.draw.rect(self.screen, self.COLORS['grid'], panel_rect, 1)

        y_offset = panel_y + 15

        # Title
        title = self.font.render("MISSION STATUS", True, self.COLORS['text'])
        self.screen.blit(title, (panel_x + 15, y_offset))
        y_offset += 30

        # Battery bar
        y_offset = self._draw_battery_bar(panel_x + 15, y_offset)
        y_offset += 20

        # Mission statistics
        y_offset = self._draw_mission_stats(panel_x + 15, y_offset)
        y_offset += 20

        # Delivery queue
        y_offset = self._draw_delivery_queue(panel_x + 15, y_offset)

    def _draw_battery_bar(self, x: int, y: int) -> int:
        """
        Draw battery status bar.

        Args:
            x, y: Top-left position

        Returns:
            New y position after drawing
        """
        # Label
        text = self.font.render("BATTERY", True, self.COLORS['text'])
        self.screen.blit(text, (x, y))
        y += 20

        bar_width = self.PANEL_WIDTH - 60
        bar_height = 20

        # Background
        bg_rect = pygame.Rect(x, y, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLORS['battery_bg'], bg_rect)

        # Fill based on level
        level = self.env.battery_level
        fill_width = int(bar_width * level)

        if level > 0.5:
            color = self.COLORS['battery_full']
        elif level > 0.2:
            color = self.COLORS['battery_mid']
        else:
            color = self.COLORS['battery_low']

        fill_rect = pygame.Rect(x, y, fill_width, bar_height)
        pygame.draw.rect(self.screen, color, fill_rect)

        # Border
        pygame.draw.rect(self.screen, self.COLORS['text'], bg_rect, 1)

        # Percentage text
        pct_text = f"{int(level * 100)}%"
        text = self.font.render(pct_text, True, self.COLORS['text'])
        self.screen.blit(text, (x + bar_width + 10, y))

        return y + bar_height + 10

    def _draw_mission_stats(self, x: int, y: int) -> int:
        """
        Draw mission statistics.

        Args:
            x, y: Top-left position

        Returns:
            New y position after drawing
        """
        # Header
        text = self.font.render("STATISTICS", True, self.COLORS['text'])
        self.screen.blit(text, (x, y))
        y += 25

        stats = [
            f"Completed: {self.env.deliveries_completed}",
            f"Successful: {self.env.deliveries_successful}",
            f"Score: {self.env.route_score:.1f}",
            f"Steps: {self.env.steps}",
            f"Episode Reward: {self.env.episode_reward:.1f}",
        ]

        for stat in stats:
            text = self.font_small.render(stat, True, self.COLORS['text_dim'])
            self.screen.blit(text, (x + 10, y))
            y += 18

        return y + 5

    def _draw_delivery_queue(self, x: int, y: int) -> int:
        """
        Draw the delivery queue with priorities.

        Args:
            x, y: Top-left position

        Returns:
            New y position after drawing
        """
        # Header
        text = self.font.render("DELIVERY QUEUE", True, self.COLORS['text'])
        self.screen.blit(text, (x, y))
        y += 25

        # Current delivery
        if self.env.current_delivery:
            delivery = self.env.current_delivery
            priority_color = self._get_priority_color(delivery.priority)

            # Current indicator
            pygame.draw.circle(self.screen, priority_color, (x + 8, y + 8), 6)
            text = self.font_small.render(
                f"CURRENT: #{delivery.id} (P{delivery.priority})",
                True, self.COLORS['text']
            )
            self.screen.blit(text, (x + 20, y))
            y += 22

            # Distance to target
            if self.env.target_position is not None:
                dist = np.linalg.norm(self.env.position - self.env.target_position)
                dist_text = f"  Distance: {dist:.1f}m"
                text = self.font_small.render(dist_text, True, self.COLORS['text_dim'])
                self.screen.blit(text, (x + 10, y))
                y += 20

        # Pending deliveries
        pending = self.env.mission_planner.state.pending_deliveries
        if pending:
            y += 10
            text = self.font_small.render(
                f"Pending ({len(pending)}):", True, self.COLORS['text']
            )
            self.screen.blit(text, (x, y))
            y += 20

            for i, delivery in enumerate(pending[:5]):  # Show first 5
                priority_color = self._get_priority_color(delivery.priority)
                pygame.draw.circle(self.screen, priority_color, (x + 8, y + 6), 4)

                info = f"#{delivery.id} P{delivery.priority}"
                if delivery.time_window:
                    remaining = delivery.time_window[1] - self.env.mission_planner.state.elapsed_time
                    if remaining > 0:
                        info += f" ({remaining:.0f}s)"
                    else:
                        info += " (LATE!)"

                text = self.font_small.render(info, True, self.COLORS['text_dim'])
                self.screen.blit(text, (x + 18, y))
                y += 18

            if len(pending) > 5:
                text = self.font_small.render(
                    f"  +{len(pending) - 5} more...",
                    True, self.COLORS['text_dim']
                )
                self.screen.blit(text, (x, y))
                y += 18
        else:
            y += 10
            text = self.font_small.render("No pending deliveries", True, self.COLORS['text_dim'])
            self.screen.blit(text, (x + 10, y))
            y += 18

        # Completed/Failed summary
        y += 20
        completed = self.env.mission_planner.state.completed_deliveries
        failed = self.env.mission_planner.state.failed_deliveries

        text = self.font_small.render(
            f"Completed IDs: {completed if completed else 'None'}",
            True, self.COLORS['battery_full'] if completed else self.COLORS['text_dim']
        )
        self.screen.blit(text, (x, y))
        y += 18

        if failed:
            text = self.font_small.render(
                f"Failed IDs: {failed}",
                True, self.COLORS['battery_low']
            )
            self.screen.blit(text, (x, y))
            y += 18

        return y

    def _get_priority_color(self, priority: int) -> tuple:
        """
        Get color for a priority level.

        Args:
            priority: Priority value (1-3)

        Returns:
            RGB color tuple
        """
        if priority == 3:
            return self.COLORS['priority_critical']
        elif priority == 2:
            return self.COLORS['priority_urgent']
        return self.COLORS['priority_normal']

    def clear_trajectory(self) -> None:
        """Clear the trajectory history."""
        self.trajectory = []

    def close(self) -> None:
        """Clean up pygame resources."""
        if PYGAME_AVAILABLE and self.screen is not None:
            pygame.quit()
            self.screen = None
