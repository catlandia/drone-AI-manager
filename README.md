# Drone AI Manager

**Layer 1: Offline Task Manager** for multi-delivery drone operations.

Part of the 4-layer drone AI architecture:
1. **Manager AI** (this) - Task queue, delivery planning
2. **Path Finder AI** - Route calculation
3. **Perception AI** - Environment sensing (always active)
4. **Flight Control AI** - Motor control (always active)

## Installation

### Quick Install
```bash
./install.sh          # Basic install
./install.sh --viz    # With visualization (pygame)
./install.sh --all    # All dependencies
```

### Manual Install
```bash
pip install -e .              # Basic
pip install -e ".[viz]"       # With pygame
pip install -e ".[all]"       # Everything
```

## Usage

### Test Import
```bash
python -c "from drone_ai import DroneDeliveryEnv, MissionPlanner; print('OK')"
```

### Run Demo
```bash
drone-demo --no-render --episodes 1    # Headless test
drone-demo --render                     # With visualization
drone-demo --difficulty 0.8 --deliveries 5
```

### Learning Sequence
```bash
drone-learn --no-render --steps-per-age 5000
drone-learn --render
```

### Python API
```python
from drone_ai import DroneDeliveryEnv, MissionPlanner
import numpy as np

# Create environment
env = DroneDeliveryEnv(
    difficulty=0.5,
    num_deliveries=3,
    render_mode=None  # or "human" for visualization
)

# Run episode
obs, info = env.reset()
for _ in range(1000):
    action = np.array([0.5, 0.5, 0.5, 0.5])  # hover
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Deliveries: {info['deliveries_successful']}/{info['deliveries_completed']}")
env.close()
```

## Architecture

```
Manager AI (Offline)
    │
    ↓ destinations
Path Finder AI
    │
    ↓ waypoints
Perception AI ←→ Flight Control AI (Always Active)
```

## Key Classes

- `MissionPlanner` - Task queue and delivery scheduling
- `DeliveryRequest` - Single delivery task definition
- `DroneDeliveryEnv` - Gymnasium environment for testing

## Current Implementation

**Math/Heuristics based** (not ML):
- Priority-based scheduling
- Greedy nearest-neighbor for route optimization
- Battery threshold decisions
