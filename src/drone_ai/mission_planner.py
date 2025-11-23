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


@dataclass
class DroneSpecs:
    """
    Physical specifications of the drone.

    Used for realistic flight calculations.
    """
    # Mass properties
    empty_weight: float = 2.5          # Drone weight without payload (kg)
    max_payload: float = 2.0           # Maximum cargo weight (kg)

    # Battery properties
    battery_capacity_wh: float = 100.0  # Battery capacity in Watt-hours
    battery_voltage: float = 22.2       # Nominal voltage (6S LiPo)

    # Motor/propulsion properties
    num_motors: int = 4
    motor_efficiency: float = 0.85      # Motor efficiency factor
    propeller_efficiency: float = 0.75  # Propeller efficiency factor

    # Flight characteristics
    cruise_speed: float = 10.0          # Cruise speed in m/s
    max_speed: float = 15.0             # Maximum speed in m/s
    climb_rate: float = 3.0             # Vertical climb rate m/s
    descent_rate: float = 2.0           # Vertical descent rate m/s

    # Power consumption (Watts)
    hover_power_base: float = 150.0     # Base hover power (empty drone)
    power_per_kg: float = 50.0          # Additional power per kg payload
    cruise_power_factor: float = 1.2    # Multiplier vs hover for forward flight
    climb_power_factor: float = 1.8     # Multiplier vs hover for climbing

    # Safety margins
    reserve_battery: float = 0.15       # Reserve battery (don't use last 15%)

    def hover_power(self, payload_kg: float = 0.0) -> float:
        """Calculate hover power consumption in Watts."""
        total_weight = self.empty_weight + payload_kg
        return self.hover_power_base + (payload_kg * self.power_per_kg)

    def cruise_power(self, payload_kg: float = 0.0) -> float:
        """Calculate cruise flight power consumption in Watts."""
        return self.hover_power(payload_kg) * self.cruise_power_factor

    def climb_power(self, payload_kg: float = 0.0) -> float:
        """Calculate climbing power consumption in Watts."""
        return self.hover_power(payload_kg) * self.climb_power_factor


class MissionPlanner:
    """
    High-level mission planner for multi-delivery drone operations.

    Responsibilities:
    - Managing delivery queue
    - Selecting optimal next delivery
    - Battery management decisions
    - Route optimization (TSP-like)
    - Priority-based scheduling

    Uses physics-based calculations for:
    - Flight time estimation
    - Battery consumption
    - Safe travel distance with payload
    """

    # Battery thresholds (fraction of usable battery)
    BATTERY_CRITICAL = 0.15    # Must return to base
    BATTERY_LOW = 0.30         # Consider returning
    BATTERY_SAFE = 0.50        # Safe for new deliveries

    def __init__(
        self,
        base_position: np.ndarray,
        battery_capacity: float = 1000.0,
        drone_specs: Optional[DroneSpecs] = None
    ):
        """
        Initialize the mission planner.

        Args:
            base_position: Home base position for pickups and recharging
            battery_capacity: Maximum battery capacity in abstract units (legacy)
            drone_specs: Physical drone specifications for realistic calculations
        """
        if not isinstance(base_position, np.ndarray):
            base_position = np.array(base_position, dtype=np.float32)
        self.base_position = base_position
        self.battery_capacity = battery_capacity
        self.drone_specs = drone_specs or DroneSpecs()
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

        Uses physics-based calculations considering:
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

        # Calculate altitude changes
        alt_to_pickup = delivery.pickup_position[2] - current_pos[2]
        alt_to_dropzone = delivery.dropzone_position[2] - delivery.pickup_position[2]

        # Use physics-based energy calculation
        energy_to_pickup = self.get_energy_cost(to_pickup, 0, alt_to_pickup)
        energy_to_dropzone = self.get_energy_cost(to_dropzone, delivery.weight, alt_to_dropzone)

        # Add hover overhead (30s at pickup)
        hover_overhead = self.drone_specs.hover_power(delivery.weight) * (30 / 3600)

        total_energy = energy_to_pickup + energy_to_dropzone + hover_overhead

        # Convert to battery percentage
        cost = total_energy / self.drone_specs.battery_capacity_wh

        return min(cost, 1.0)  # Cap at full battery

    def _estimate_return_cost(self, from_position: np.ndarray) -> float:
        """
        Estimate cost to return to base from a position.

        Args:
            from_position: Position to return from

        Returns:
            Estimated battery cost (0.0 to 1.0)
        """
        distance = np.linalg.norm(self.base_position - from_position)
        alt_change = self.base_position[2] - from_position[2]

        return self.get_battery_percentage_cost(
            distance=distance,
            payload_kg=0,  # No payload on return
            altitude_change=alt_change
        )

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

    # =========================================================================
    # Physics-Based Calculations
    # =========================================================================

    def get_max_range(self, payload_kg: float = 0.0) -> float:
        """
        Calculate maximum one-way flight range with given payload.

        This is the theoretical max distance before battery is depleted.
        For safe operation, use get_safe_range() instead.

        Args:
            payload_kg: Cargo weight in kg

        Returns:
            Maximum range in meters
        """
        specs = self.drone_specs

        # Available energy (Wh) - excluding reserve
        usable_battery = specs.battery_capacity_wh * (1 - specs.reserve_battery)

        # Power consumption during cruise
        cruise_power = specs.cruise_power(payload_kg)

        # Flight time in hours
        flight_time_hours = usable_battery / cruise_power

        # Range in meters
        max_range = specs.cruise_speed * flight_time_hours * 3600

        return max_range

    def get_safe_range(self, payload_kg: float = 0.0) -> float:
        """
        Calculate safe round-trip range with given payload.

        Accounts for:
        - Outbound flight (with payload)
        - Return flight (without payload)
        - Reserve battery
        - Takeoff/landing overhead

        Args:
            payload_kg: Cargo weight in kg

        Returns:
            Safe one-way range in meters (total round trip = 2x this)
        """
        specs = self.drone_specs

        # Available energy (Wh)
        usable_battery = specs.battery_capacity_wh * (1 - specs.reserve_battery)

        # Subtract takeoff/landing overhead (assume 30 seconds hover each way)
        hover_energy = specs.hover_power(payload_kg) * (60 / 3600)  # 60s total hover
        usable_battery -= hover_energy

        # Power for outbound (with payload) and return (empty)
        outbound_power = specs.cruise_power(payload_kg)
        return_power = specs.cruise_power(0)

        # Time to fly distance D:
        # outbound: D / cruise_speed hours
        # return: D / cruise_speed hours
        # Energy: D/v * outbound_power + D/v * return_power = usable_battery
        # D * (outbound_power + return_power) / v = usable_battery
        # D = usable_battery * v / (outbound_power + return_power)

        safe_range = (usable_battery * specs.cruise_speed * 3600) / (outbound_power + return_power)

        return max(0, safe_range)

    def get_flight_time(
        self,
        distance: float,
        payload_kg: float = 0.0,
        include_vertical: bool = True,
        altitude_change: float = 0.0
    ) -> float:
        """
        Estimate flight time for a given distance and payload.

        Args:
            distance: Horizontal distance in meters
            payload_kg: Cargo weight in kg
            include_vertical: Whether to include climb/descent time
            altitude_change: Vertical distance in meters (positive = climb)

        Returns:
            Estimated flight time in seconds
        """
        specs = self.drone_specs

        # Horizontal flight time
        horizontal_time = distance / specs.cruise_speed

        # Vertical flight time
        vertical_time = 0.0
        if include_vertical and altitude_change != 0:
            if altitude_change > 0:
                vertical_time = altitude_change / specs.climb_rate
            else:
                vertical_time = abs(altitude_change) / specs.descent_rate

        return horizontal_time + vertical_time

    def get_energy_cost(
        self,
        distance: float,
        payload_kg: float = 0.0,
        altitude_change: float = 0.0
    ) -> float:
        """
        Calculate energy cost for a flight segment.

        Args:
            distance: Horizontal distance in meters
            payload_kg: Cargo weight in kg
            altitude_change: Vertical distance (positive = climb)

        Returns:
            Energy cost in Watt-hours
        """
        specs = self.drone_specs

        # Horizontal cruise energy
        horizontal_time_hours = (distance / specs.cruise_speed) / 3600
        cruise_energy = specs.cruise_power(payload_kg) * horizontal_time_hours

        # Vertical energy
        vertical_energy = 0.0
        if altitude_change > 0:
            climb_time_hours = (altitude_change / specs.climb_rate) / 3600
            vertical_energy = specs.climb_power(payload_kg) * climb_time_hours
        elif altitude_change < 0:
            # Descent uses less power (closer to hover)
            descent_time_hours = (abs(altitude_change) / specs.descent_rate) / 3600
            vertical_energy = specs.hover_power(payload_kg) * descent_time_hours

        return cruise_energy + vertical_energy

    def get_battery_percentage_cost(
        self,
        distance: float,
        payload_kg: float = 0.0,
        altitude_change: float = 0.0
    ) -> float:
        """
        Calculate battery percentage used for a flight segment.

        Args:
            distance: Horizontal distance in meters
            payload_kg: Cargo weight in kg
            altitude_change: Vertical distance (positive = climb)

        Returns:
            Battery percentage used (0.0 to 1.0)
        """
        energy_wh = self.get_energy_cost(distance, payload_kg, altitude_change)
        return energy_wh / self.drone_specs.battery_capacity_wh

    def can_complete_delivery(
        self,
        delivery: DeliveryRequest,
        current_position: Optional[np.ndarray] = None,
        current_battery: Optional[float] = None
    ) -> Tuple[bool, dict]:
        """
        Check if a delivery can be safely completed.

        Args:
            delivery: The delivery to check
            current_position: Current drone position (default: use state)
            current_battery: Current battery level 0-1 (default: use state)

        Returns:
            Tuple of (can_complete, details_dict)
        """
        if current_position is None:
            current_position = (
                self.state.current_delivery.dropzone_position
                if self.state.current_delivery
                else self.base_position
            )
        if current_battery is None:
            current_battery = self.state.battery_level

        # Calculate distances
        to_pickup = np.linalg.norm(delivery.pickup_position - current_position)
        to_dropzone = np.linalg.norm(
            delivery.dropzone_position - delivery.pickup_position
        )
        to_base = np.linalg.norm(self.base_position - delivery.dropzone_position)

        # Calculate altitude changes
        alt_to_pickup = delivery.pickup_position[2] - current_position[2]
        alt_to_dropzone = delivery.dropzone_position[2] - delivery.pickup_position[2]
        alt_to_base = self.base_position[2] - delivery.dropzone_position[2]

        # Energy costs
        energy_to_pickup = self.get_energy_cost(to_pickup, 0, alt_to_pickup)
        energy_to_dropzone = self.get_energy_cost(
            to_dropzone, delivery.weight, alt_to_dropzone
        )
        energy_to_base = self.get_energy_cost(to_base, 0, alt_to_base)

        # Add hover overhead for pickup and dropoff (30s each)
        hover_overhead = (
            self.drone_specs.hover_power(delivery.weight) * (30 / 3600) +
            self.drone_specs.hover_power(0) * (30 / 3600)
        )

        total_energy = energy_to_pickup + energy_to_dropzone + energy_to_base + hover_overhead

        # Available energy
        available_wh = current_battery * self.drone_specs.battery_capacity_wh
        reserve_wh = self.drone_specs.reserve_battery * self.drone_specs.battery_capacity_wh
        usable_wh = available_wh - reserve_wh

        # Flight times
        time_to_pickup = self.get_flight_time(to_pickup, 0, True, alt_to_pickup)
        time_to_dropzone = self.get_flight_time(
            to_dropzone, delivery.weight, True, alt_to_dropzone
        )
        time_to_base = self.get_flight_time(to_base, 0, True, alt_to_base)
        total_time = time_to_pickup + time_to_dropzone + time_to_base + 60  # +60s overhead

        details = {
            'can_complete': usable_wh >= total_energy,
            'energy_required_wh': total_energy,
            'energy_available_wh': usable_wh,
            'battery_cost_percent': total_energy / self.drone_specs.battery_capacity_wh,
            'margin_wh': usable_wh - total_energy,
            'estimated_time_seconds': total_time,
            'distance_total_m': to_pickup + to_dropzone + to_base,
            'segments': {
                'to_pickup_m': to_pickup,
                'to_dropzone_m': to_dropzone,
                'to_base_m': to_base,
            }
        }

        return details['can_complete'], details

    def get_range_with_payload(self, payload_kg: float) -> dict:
        """
        Get comprehensive range information for a given payload.

        Args:
            payload_kg: Cargo weight in kg

        Returns:
            Dictionary with range calculations
        """
        specs = self.drone_specs

        if payload_kg > specs.max_payload:
            return {
                'error': f'Payload {payload_kg}kg exceeds max {specs.max_payload}kg',
                'valid': False
            }

        max_range = self.get_max_range(payload_kg)
        safe_range = self.get_safe_range(payload_kg)

        # Flight time at safe range
        safe_flight_time = self.get_flight_time(safe_range * 2, payload_kg)  # Round trip

        # Power consumption
        hover_power = specs.hover_power(payload_kg)
        cruise_power = specs.cruise_power(payload_kg)

        # Hover endurance
        usable_wh = specs.battery_capacity_wh * (1 - specs.reserve_battery)
        hover_time = (usable_wh / hover_power) * 3600  # seconds

        return {
            'valid': True,
            'payload_kg': payload_kg,
            'max_one_way_range_m': max_range,
            'safe_round_trip_range_m': safe_range,
            'hover_endurance_seconds': hover_time,
            'safe_flight_time_seconds': safe_flight_time,
            'cruise_speed_mps': specs.cruise_speed,
            'power_consumption': {
                'hover_watts': hover_power,
                'cruise_watts': cruise_power,
            },
            'efficiency': {
                'meters_per_wh': safe_range * 2 / usable_wh,
                'wh_per_km': usable_wh / (safe_range * 2 / 1000) if safe_range > 0 else float('inf'),
            }
        }
