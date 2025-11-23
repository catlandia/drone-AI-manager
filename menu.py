#!/usr/bin/env python3
"""
Drone AI Manager - Interactive Menu

Easy-to-use interface for the drone mission planner.
Run: python menu.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from drone_ai import MissionPlanner, DroneSpecs, DeliveryRequest, DroneDeliveryEnv


class DroneMenu:
    def __init__(self):
        self.planner = None
        self.drone_specs = DroneSpecs()
        self.env = None
        self._init_planner()

    def _init_planner(self):
        """Initialize the mission planner."""
        self.planner = MissionPlanner(
            base_position=np.array([0, 0, 0]),
            drone_specs=self.drone_specs
        )

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self, title: str):
        """Print a section header."""
        print()
        print("=" * 50)
        print(f"  {title}")
        print("=" * 50)
        print()

    def print_menu(self):
        """Print the main menu."""
        self.clear_screen()
        print("""
    ╔═══════════════════════════════════════════════════╗
    ║         DRONE AI MISSION MANAGER                  ║
    ╠═══════════════════════════════════════════════════╣
    ║                                                   ║
    ║  1. Check Drone Range & Capabilities              ║
    ║  2. Calculate Delivery Feasibility                ║
    ║  3. Add Delivery to Queue                         ║
    ║  4. View Mission Status                           ║
    ║  5. Run Demo Simulation                           ║
    ║  6. Configure Drone Specs                         ║
    ║  7. Quick Range Calculator                        ║
    ║  8. Reset Mission                                 ║
    ║                                                   ║
    ║  0. Exit                                          ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
        """)

    def wait_for_enter(self):
        """Wait for user to press Enter."""
        print()
        input("Press Enter to continue...")

    def get_float(self, prompt: str, default: float = None) -> float:
        """Get a float input from user."""
        while True:
            try:
                if default is not None:
                    val = input(f"{prompt} [{default}]: ").strip()
                    if val == "":
                        return default
                else:
                    val = input(f"{prompt}: ").strip()
                return float(val)
            except ValueError:
                print("Please enter a valid number.")

    def get_int(self, prompt: str, default: int = None) -> int:
        """Get an integer input from user."""
        while True:
            try:
                if default is not None:
                    val = input(f"{prompt} [{default}]: ").strip()
                    if val == "":
                        return default
                else:
                    val = input(f"{prompt}: ").strip()
                return int(val)
            except ValueError:
                print("Please enter a valid integer.")

    def get_position(self, name: str) -> np.ndarray:
        """Get a 3D position from user."""
        print(f"Enter {name} coordinates:")
        x = self.get_float("  X (meters)", 0)
        y = self.get_float("  Y (meters)", 0)
        z = self.get_float("  Z (meters/altitude)", 5)
        return np.array([x, y, z])

    # === Menu Options ===

    def check_range(self):
        """Check drone range and capabilities."""
        self.print_header("DRONE RANGE & CAPABILITIES")

        specs = self.drone_specs
        print(f"Drone Specifications:")
        print(f"  Empty Weight:    {specs.empty_weight} kg")
        print(f"  Max Payload:     {specs.max_payload} kg")
        print(f"  Battery:         {specs.battery_capacity_wh} Wh")
        print(f"  Cruise Speed:    {specs.cruise_speed} m/s")
        print(f"  Reserve Battery: {specs.reserve_battery * 100:.0f}%")
        print()

        print("Range by Payload:")
        print("-" * 50)
        print(f"{'Payload':<10} {'Max Range':<15} {'Safe Range':<15} {'Hover Time'}")
        print("-" * 50)

        for payload in [0, 0.5, 1.0, 1.5, 2.0]:
            if payload <= specs.max_payload:
                info = self.planner.get_range_with_payload(payload)
                max_r = info['max_one_way_range_m'] / 1000
                safe_r = info['safe_round_trip_range_m'] / 1000
                hover = info['hover_endurance_seconds'] / 60
                print(f"{payload:<10.1f} {max_r:<15.1f}km {safe_r:<15.1f}km {hover:.1f} min")

        self.wait_for_enter()

    def calculate_delivery(self):
        """Calculate if a specific delivery is feasible."""
        self.print_header("DELIVERY FEASIBILITY CALCULATOR")

        print("Enter delivery details:")
        dropzone = self.get_position("dropzone")
        weight = self.get_float("Package weight (kg)", 1.0)
        battery = self.get_float("Current battery level (0-100%)", 100) / 100

        delivery = DeliveryRequest(
            id=0,
            pickup_position=self.planner.base_position.copy(),
            dropzone_position=dropzone,
            weight=weight
        )

        can_do, details = self.planner.can_complete_delivery(
            delivery,
            current_position=self.planner.base_position,
            current_battery=battery
        )

        print()
        print("=" * 50)
        print("RESULT:", "CAN COMPLETE" if can_do else "CANNOT COMPLETE")
        print("=" * 50)
        print()
        print(f"Total Distance:    {details['distance_total_m']:.0f} m")
        print(f"  - To dropzone:   {details['segments']['to_dropzone_m']:.0f} m")
        print(f"  - Return to base:{details['segments']['to_base_m']:.0f} m")
        print()
        print(f"Energy Required:   {details['energy_required_wh']:.2f} Wh")
        print(f"Energy Available:  {details['energy_available_wh']:.2f} Wh")
        print(f"Battery Cost:      {details['battery_cost_percent'] * 100:.1f}%")
        print(f"Safety Margin:     {details['margin_wh']:.1f} Wh")
        print()
        print(f"Estimated Time:    {details['estimated_time_seconds']:.0f} seconds")
        print(f"                   ({details['estimated_time_seconds']/60:.1f} minutes)")

        self.wait_for_enter()

    def add_delivery(self):
        """Add a delivery to the queue."""
        self.print_header("ADD DELIVERY TO QUEUE")

        dropzone = self.get_position("dropzone")
        weight = self.get_float("Package weight (kg)", 1.0)
        priority = self.get_int("Priority (1=normal, 2=urgent, 3=critical)", 1)

        delivery = self.planner.create_delivery(
            dropzone_position=dropzone,
            weight=weight,
            priority=priority
        )

        print()
        print(f"Delivery #{delivery.id} added to queue!")
        print(f"  Dropzone: [{dropzone[0]:.1f}, {dropzone[1]:.1f}, {dropzone[2]:.1f}]")
        print(f"  Weight:   {weight} kg")
        print(f"  Priority: {priority}")
        print()
        print(f"Total pending deliveries: {len(self.planner.state.pending_deliveries)}")

        self.wait_for_enter()

    def view_status(self):
        """View current mission status."""
        self.print_header("MISSION STATUS")

        summary = self.planner.get_mission_summary()
        state = self.planner.state

        print(f"Battery Level:     {state.battery_level * 100:.1f}%")
        print(f"Elapsed Time:      {state.elapsed_time:.1f} seconds")
        print(f"Total Score:       {state.total_score:.1f}")
        print()
        print(f"Pending Deliveries:   {summary['pending_count']}")
        print(f"Completed Deliveries: {summary['completed_count']}")
        print(f"Failed Deliveries:    {summary['failed_count']}")
        print()

        if state.current_delivery:
            d = state.current_delivery
            print(f"Current Delivery: #{d.id}")
            print(f"  Dropzone: [{d.dropzone_position[0]:.1f}, {d.dropzone_position[1]:.1f}, {d.dropzone_position[2]:.1f}]")
            print(f"  Weight:   {d.weight} kg")
            print(f"  Priority: {d.priority}")
        else:
            print("Current Delivery: None")

        if state.pending_deliveries:
            print()
            print("Pending Queue:")
            for d in state.pending_deliveries[:5]:
                dist = np.linalg.norm(d.dropzone_position - self.planner.base_position)
                print(f"  #{d.id}: P{d.priority}, {dist:.0f}m away, {d.weight}kg")
            if len(state.pending_deliveries) > 5:
                print(f"  ... and {len(state.pending_deliveries) - 5} more")

        self.wait_for_enter()

    def run_demo(self):
        """Run the demo simulation."""
        self.print_header("RUN DEMO SIMULATION")

        print("Demo Options:")
        print("  1. Headless (no visualization)")
        print("  2. With visualization (requires pygame)")
        print("  3. Cancel")
        print()

        choice = self.get_int("Choice", 1)

        if choice == 3:
            return

        render = (choice == 2)
        episodes = self.get_int("Number of episodes", 1)
        difficulty = self.get_float("Difficulty (0.0-1.0)", 0.5)
        deliveries = self.get_int("Deliveries per episode", 3)

        print()
        print("Starting demo...")
        print()

        # Import and run demo
        from drone_ai.demo import run_demo
        run_demo(
            difficulty=difficulty,
            num_deliveries=deliveries,
            render=render,
            max_episodes=episodes
        )

        self.wait_for_enter()

    def configure_specs(self):
        """Configure drone specifications."""
        self.print_header("CONFIGURE DRONE SPECS")

        print("Current specs:")
        print(f"  1. Empty Weight:     {self.drone_specs.empty_weight} kg")
        print(f"  2. Max Payload:      {self.drone_specs.max_payload} kg")
        print(f"  3. Battery Capacity: {self.drone_specs.battery_capacity_wh} Wh")
        print(f"  4. Cruise Speed:     {self.drone_specs.cruise_speed} m/s")
        print(f"  5. Hover Power:      {self.drone_specs.hover_power_base} W")
        print(f"  6. Reset to Default")
        print(f"  7. Back")
        print()

        choice = self.get_int("What to change", 7)

        if choice == 1:
            self.drone_specs.empty_weight = self.get_float("New empty weight (kg)")
        elif choice == 2:
            self.drone_specs.max_payload = self.get_float("New max payload (kg)")
        elif choice == 3:
            self.drone_specs.battery_capacity_wh = self.get_float("New battery capacity (Wh)")
        elif choice == 4:
            self.drone_specs.cruise_speed = self.get_float("New cruise speed (m/s)")
        elif choice == 5:
            self.drone_specs.hover_power_base = self.get_float("New hover power (W)")
        elif choice == 6:
            self.drone_specs = DroneSpecs()
            print("Reset to default specs.")

        # Reinitialize planner with new specs
        self._init_planner()

        if choice < 7:
            self.wait_for_enter()

    def quick_range(self):
        """Quick range calculator."""
        self.print_header("QUICK RANGE CALCULATOR")

        weight = self.get_float("Payload weight (kg)", 1.0)

        if weight > self.drone_specs.max_payload:
            print(f"\nERROR: Payload exceeds max capacity ({self.drone_specs.max_payload}kg)")
            self.wait_for_enter()
            return

        info = self.planner.get_range_with_payload(weight)

        print()
        print(f"With {weight}kg payload:")
        print("-" * 40)
        print(f"Max one-way range:     {info['max_one_way_range_m']/1000:.2f} km")
        print(f"Safe round-trip range: {info['safe_round_trip_range_m']/1000:.2f} km (each way)")
        print(f"Hover endurance:       {info['hover_endurance_seconds']/60:.1f} minutes")
        print(f"Safe flight time:      {info['safe_flight_time_seconds']/60:.1f} minutes")
        print()
        print(f"Power consumption:")
        print(f"  Hover:  {info['power_consumption']['hover_watts']:.0f} W")
        print(f"  Cruise: {info['power_consumption']['cruise_watts']:.0f} W")
        print()
        print(f"Efficiency:")
        print(f"  {info['efficiency']['meters_per_wh']:.1f} meters per Wh")
        print(f"  {info['efficiency']['wh_per_km']:.1f} Wh per km")

        self.wait_for_enter()

    def reset_mission(self):
        """Reset the mission planner."""
        self.print_header("RESET MISSION")

        print("This will clear all deliveries and reset the mission state.")
        confirm = input("Are you sure? (y/n): ").strip().lower()

        if confirm == 'y':
            self.planner.reset()
            print("Mission reset!")
        else:
            print("Cancelled.")

        self.wait_for_enter()

    def run(self):
        """Main menu loop."""
        while True:
            self.print_menu()
            choice = input("  Enter choice: ").strip()

            if choice == '1':
                self.check_range()
            elif choice == '2':
                self.calculate_delivery()
            elif choice == '3':
                self.add_delivery()
            elif choice == '4':
                self.view_status()
            elif choice == '5':
                self.run_demo()
            elif choice == '6':
                self.configure_specs()
            elif choice == '7':
                self.quick_range()
            elif choice == '8':
                self.reset_mission()
            elif choice == '0':
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
                self.wait_for_enter()


def main():
    try:
        menu = DroneMenu()
        menu.run()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease run the installer first:")
        print("  python install.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
