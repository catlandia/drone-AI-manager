"""
Mission Planner Module for Drone AI.

Handles high-level decision making: what to deliver, where, when, and how to
optimize multi-delivery operations. Sits ABOVE path selection which handles
waypoint-to-waypoint navigation.

Hierarchy: Mission Planner -> Path Selection -> Flight Control
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class DeliveryRequest:
    """A single delivery task."""
    id: int
    pickup_position: np.ndarray      # Where to pick up package
    dropzone_position: np.ndarray    # Where to deliver
    priority: int = 1                # 1=normal, 2=urgent, 3=critical
    time_window: Optional[Tuple[float, float]] = None  # (earliest, latest) in seconds
    weight: float = 1.0              # Package weight in kg
    reward_value: float = 100.0      # Points for successful delivery

    def __post_init__(self):
        """Ensure positions are numpy arrays."""
        if not isinstance(self.pickup_position, np.ndarray):
            self.pickup_position = np.array(self.pickup_position, dtype=np.float32)
        if not isinstance(self.dropzone_position, np.ndarray):
            self.dropzone_position = np.array(self.dropzone_position, dtype=np.float32)


@dataclass
class MissionState:
    """Current mission status."""
    pending_deliveries: List[DeliveryRequest] = field(default_factory=list)
    completed_deliveries: List[int] = field(default_factory=list)  # IDs
    failed_deliveries: List[int] = field(default_factory=list)     # IDs
    current_delivery: Optional[DeliveryRequest] = None
    battery_level: float = 1.0              # 0.0 to 1.0
    total_score: float = 0.0
    elapsed_time: float = 0.0

    def reset(self):
        """Reset mission state for new episode."""
        self.pending_deliveries = []
        self.completed_deliveries = []
        self.failed_deliveries = []
        self.current_delivery = None
        self.battery_level = 1.0
        self.total_score = 0.0
        self.elapsed_time = 0.0


class MissionPlanner:
    """
    High-level mission planner for multi-delivery drone operations.

    Responsibilities:
    - Managing delivery queue
    - Selecting optimal next delivery
    - Battery management decisions
    - Route optimization (TSP-like)
    - Priority-based scheduling
    """

    # Battery thresholds
    BATTERY_CRITICAL = 0.15    # Must return to base
    BATTERY_LOW = 0.30         # Consider returning
    BATTERY_SAFE = 0.50        # Safe for new deliveries

    # Cost estimation factors
    DISTANCE_COST_FACTOR = 0.001      # Battery drain per unit distance
    HOVER_COST_PER_SECOND = 0.0001    # Battery drain for hovering
    DELIVERY_OVERHEAD = 0.02          # Fixed battery cost per delivery (landing/takeoff)

    def __init__(self, base_position: np.ndarray, battery_capacity: float = 1000.0):
        """
        Initialize the mission planner.

        Args:
            base_position: Home base position for pickups and recharging
            battery_capacity: Maximum battery capacity in abstract units
        """
        if not isinstance(base_position, np.ndarray):
            base_position = np.array(base_position, dtype=np.float32)
        self.base_position = base_position
        self.battery_capacity = battery_capacity
        self.state = MissionState()
        self._delivery_id_counter = 0

    def reset(self):
        """Reset planner state for new episode."""
        self.state.reset()
        self._delivery_id_counter = 0

    def add_delivery(self, request: DeliveryRequest) -> None:
        """
        Add a new delivery request to the queue.

        Args:
            request: The delivery request to add
        """
        self.state.pending_deliveries.append(request)
        # Sort by priority (highest first) then by time window
        self._sort_pending_deliveries()

    def create_delivery(
        self,
        dropzone_position: np.ndarray,
        priority: int = 1,
        time_window: Optional[Tuple[float, float]] = None,
        weight: float = 1.0,
        reward_value: float = 100.0
    ) -> DeliveryRequest:
        """
        Create and add a new delivery request.

        Args:
            dropzone_position: Delivery destination
            priority: 1=normal, 2=urgent, 3=critical
            time_window: Optional (earliest, latest) time bounds
            weight: Package weight in kg
            reward_value: Points for successful delivery

        Returns:
            The created DeliveryRequest
        """
        request = DeliveryRequest(
            id=self._delivery_id_counter,
            pickup_position=self.base_position.copy(),
            dropzone_position=dropzone_position,
            priority=priority,
            time_window=time_window,
            weight=weight,
            reward_value=reward_value
        )
        self._delivery_id_counter += 1
        self.add_delivery(request)
        return request

    def _sort_pending_deliveries(self) -> None:
        """Sort pending deliveries by priority (descending) and urgency."""
        def sort_key(d: DeliveryRequest) -> Tuple[int, float, float]:
            # Primary: priority (higher = earlier, so negate)
            # Secondary: time window urgency (earlier deadline = earlier)
            # Tertiary: distance from current position
            deadline = d.time_window[1] if d.time_window else float('inf')
            current_pos = (
                self.state.current_delivery.dropzone_position
                if self.state.current_delivery
                else self.base_position
            )
            distance = np.linalg.norm(d.dropzone_position - current_pos)
            return (-d.priority, deadline, distance)

        self.state.pending_deliveries.sort(key=sort_key)

    def select_next_delivery(self) -> Optional[DeliveryRequest]:
        """
        Choose optimal next delivery based on priority, distance, and battery.

        Selection criteria (in order):
        1. Highest priority deliveries first
        2. Time-critical deliveries (approaching deadline)
        3. Nearest delivery to minimize battery usage
        4. Battery must be sufficient for delivery + return

        Returns:
            Next delivery to attempt, or None if no viable delivery
        """
        if not self.state.pending_deliveries:
            return None

        # Re-sort with current position
        self._sort_pending_deliveries()

        # Find first delivery we can afford
        for delivery in self.state.pending_deliveries:
            cost = self.estimate_delivery_cost(delivery)
            # Ensure we can complete delivery AND return to base
            return_cost = self._estimate_return_cost(delivery.dropzone_position)

            if self.state.battery_level >= cost + return_cost + self.BATTERY_CRITICAL:
                self.state.pending_deliveries.remove(delivery)
                self.state.current_delivery = delivery
                return delivery

        # No affordable delivery found
        return None

    def complete_current_delivery(self, success: bool) -> float:
        """
        Mark current delivery as complete.

        Args:
            success: Whether delivery was successful

        Returns:
            Score gained/lost from this delivery
        """
        if not self.state.current_delivery:
            return 0.0

        delivery = self.state.current_delivery
        score = 0.0

        if success:
            self.state.completed_deliveries.append(delivery.id)
            score = delivery.reward_value

            # Priority bonus
            score += delivery.priority * 25.0

            # Time window bonus/penalty
            if delivery.time_window:
                if self.state.elapsed_time <= delivery.time_window[1]:
                    score += 30.0  # On-time bonus
                else:
                    score -= 20.0  # Late penalty
        else:
            self.state.failed_deliveries.append(delivery.id)
            score = -delivery.reward_value * 0.5  # Penalty for failure

        self.state.total_score += score
        self.state.current_delivery = None
        return score

    def should_return_to_base(self) -> bool:
        """
        Decide if drone should return for battery/reload.

        Returns True if:
        - Battery is critically low
        - Battery is low and no high-priority pending deliveries
        - No pending deliveries (mission complete)

        Returns:
            True if drone should return to base
        """
        # Critical battery - must return
        if self.state.battery_level <= self.BATTERY_CRITICAL:
            return True

        # No more deliveries
        if not self.state.pending_deliveries:
            return True

        # Low battery - check if any affordable deliveries remain
        if self.state.battery_level <= self.BATTERY_LOW:
            # Check if we can afford ANY delivery
            for delivery in self.state.pending_deliveries:
                cost = self.estimate_delivery_cost(delivery)
                return_cost = self._estimate_return_cost(delivery.dropzone_position)
                if self.state.battery_level >= cost + return_cost + self.BATTERY_CRITICAL:
                    # We can still make a delivery
                    return False
            # No affordable deliveries
            return True

        return False

    def estimate_delivery_cost(self, delivery: DeliveryRequest) -> float:
        """
        Estimate battery/time cost for a delivery.

        Considers:
        - Distance from current position to pickup
        - Distance from pickup to dropzone
        - Package weight impact
        - Delivery overhead (landing/takeoff)

        Args:
            delivery: The delivery to estimate cost for

        Returns:
            Estimated battery cost (0.0 to 1.0)
        """
        # Current position
        current_pos = (
            self.state.current_delivery.dropzone_position
            if self.state.current_delivery
            else self.base_position
        )

        # Distance to pickup
        to_pickup = np.linalg.norm(delivery.pickup_position - current_pos)

        # Distance from pickup to dropzone
        to_dropzone = np.linalg.norm(
            delivery.dropzone_position - delivery.pickup_position
        )

        total_distance = to_pickup + to_dropzone

        # Base distance cost
        cost = total_distance * self.DISTANCE_COST_FACTOR

        # Weight multiplier (heavier = more power)
        weight_multiplier = 1.0 + (delivery.weight - 1.0) * 0.1
        cost *= weight_multiplier

        # Delivery overhead
        cost += self.DELIVERY_OVERHEAD

        return min(cost, 1.0)  # Cap at full battery

    def _estimate_return_cost(self, from_position: np.ndarray) -> float:
        """
        Estimate cost to return to base from a position.

        Args:
            from_position: Position to return from

        Returns:
            Estimated battery cost
        """
        distance = np.linalg.norm(self.base_position - from_position)
        return distance * self.DISTANCE_COST_FACTOR + self.DELIVERY_OVERHEAD * 0.5

    def optimize_delivery_order(self) -> List[DeliveryRequest]:
        """
        Reorder pending deliveries for optimal route (TSP-like).

        Uses a greedy nearest-neighbor heuristic with priority weighting:
        1. Start from current position
        2. Consider priority as distance multiplier
        3. Always select nearest weighted delivery

        Returns:
            Optimized list of deliveries
        """
        if len(self.state.pending_deliveries) <= 1:
            return self.state.pending_deliveries.copy()

        remaining = self.state.pending_deliveries.copy()
        optimized = []

        current_pos = (
            self.state.current_delivery.dropzone_position
            if self.state.current_delivery
            else self.base_position
        )

        while remaining:
            best_delivery = None
            best_score = float('inf')

            for delivery in remaining:
                distance = np.linalg.norm(delivery.dropzone_position - current_pos)

                # Priority reduces effective distance
                # Higher priority = lower score = selected earlier
                priority_factor = 1.0 / (delivery.priority + 0.5)

                # Time urgency factor
                urgency_factor = 1.0
                if delivery.time_window:
                    time_remaining = delivery.time_window[1] - self.state.elapsed_time
                    if time_remaining > 0:
                        urgency_factor = min(1.0, time_remaining / 60.0)  # Within 60s = urgent

                score = distance * priority_factor * urgency_factor

                if score < best_score:
                    best_score = score
                    best_delivery = delivery

            if best_delivery:
                remaining.remove(best_delivery)
                optimized.append(best_delivery)
                current_pos = best_delivery.dropzone_position

        return optimized

    def update_battery(self, new_level: float) -> None:
        """
        Update the tracked battery level.

        Args:
            new_level: New battery level (0.0 to 1.0)
        """
        self.state.battery_level = max(0.0, min(1.0, new_level))

    def update_time(self, delta_time: float) -> None:
        """
        Update elapsed mission time.

        Args:
            delta_time: Time elapsed since last update
        """
        self.state.elapsed_time += delta_time

    def get_mission_summary(self) -> dict:
        """
        Get a summary of current mission status.

        Returns:
            Dictionary with mission statistics
        """
        return {
            'pending_count': len(self.state.pending_deliveries),
            'completed_count': len(self.state.completed_deliveries),
            'failed_count': len(self.state.failed_deliveries),
            'current_delivery_id': (
                self.state.current_delivery.id
                if self.state.current_delivery
                else None
            ),
            'battery_level': self.state.battery_level,
            'total_score': self.state.total_score,
            'elapsed_time': self.state.elapsed_time,
            'mission_complete': (
                len(self.state.pending_deliveries) == 0
                and self.state.current_delivery is None
            )
        }

    def get_nearest_delivery_distance(self) -> float:
        """
        Get distance to the nearest pending delivery.

        Returns:
            Distance to nearest delivery, or 0.0 if none pending
        """
        if not self.state.pending_deliveries:
            return 0.0

        current_pos = (
            self.state.current_delivery.dropzone_position
            if self.state.current_delivery
            else self.base_position
        )

        min_distance = float('inf')
        for delivery in self.state.pending_deliveries:
            distance = np.linalg.norm(delivery.dropzone_position - current_pos)
            min_distance = min(min_distance, distance)

        return min_distance if min_distance != float('inf') else 0.0

    def get_highest_priority(self) -> int:
        """
        Get the highest priority among pending deliveries.

        Returns:
            Highest priority value (1-3), or 0 if none pending
        """
        if not self.state.pending_deliveries:
            return 0
        return max(d.priority for d in self.state.pending_deliveries)
