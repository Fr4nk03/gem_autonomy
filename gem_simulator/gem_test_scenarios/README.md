# CS588 Team 6 – Test Scenarios

Gazebo test scenarios for evaluating the VLM-based autonomous driving pipeline on the POLARIS GEM e4.

## Overview

| # | Scenario | World | Key Stimulus | Expected Action |
|---|----------|-------|--------------|-----------------|
| 01 | Stop Sign in Path | `track1.world` | Stop sign 20 m ahead | STOP |
| 02 | Pedestrian Crossing | `track1.world` | Actor crosses road | STOP → PROCEED |
| 03 | Worker Wave Gesture | `track1.world` | Worker waving near road | SLOW |
| 04 | Worker STOP Gesture | `track1.world` | Worker blocks road | STOP → PROCEED |
| 05 | Circular Loop Mixed | `highbay_track.world` | All of the above | Multi-step |

## Package Structure

```
gem_test_scenarios/
├── scenes/                         YAML scene definitions (objects + agents)
│   ├── scenario_01_stop_sign.yaml
│   ├── scenario_02_pedestrian_crossing.yaml
│   ├── scenario_03_worker_wave_gesture.yaml
│   ├── scenario_04_worker_stop_gesture.yaml
│   └── scenario_05_circular_loop_mixed.yaml
├── launch/                         One launch file per scenario
│   ├── scenario_01_stop_sign.launch
│   ├── scenario_02_pedestrian_crossing.launch
│   ├── scenario_03_worker_wave_gesture.launch
│   ├── scenario_04_worker_stop_gesture.launch
│   └── scenario_05_circular_loop_mixed.launch
├── scripts/
│   └── mock_perception_publisher.py  Simulates YOLO + VLM output
├── CMakeLists.txt
├── package.xml
└── README.md
```

## How to Run

### Build
```bash
cd ~/catkin_ws   # or wherever you source the GEM simulator workspace
catkin_make
source devel/setup.bash
```

### Launch a scenario
```bash
# Scenario 01 – Stop Sign
roslaunch gem_test_scenarios scenario_01_stop_sign.launch

# Scenario 02 – Pedestrian Crossing
roslaunch gem_test_scenarios scenario_02_pedestrian_crossing.launch

# Scenario 03 – Worker Wave (SLOW)
roslaunch gem_test_scenarios scenario_03_worker_wave_gesture.launch

# Scenario 04 – Worker STOP Gesture
roslaunch gem_test_scenarios scenario_04_worker_stop_gesture.launch

# Scenario 05 – Full Integration Loop (headless)
roslaunch gem_test_scenarios scenario_05_circular_loop_mixed.launch gui:=false
```

### Monitor topics
```bash
rostopic echo /yolo/detections
rostopic echo /vlm/scene_label
rostopic echo /safety/action_override    # from gem_safety ROS2 node via bridge
rostopic echo /test/scenario_phase
```

## How Gesture Simulation Works

Gazebo Classic does not support custom skeletal animations beyond the built-in
`walk.dae` motion. To simulate gesture detection:

1. **Gazebo**: An actor (Walking Person) is positioned near the road.  Its
   waypoint trajectory makes it oscillate slightly (wave scenario) or walk into
   the road (stop scenario) to produce a realistic visual.

2. **`mock_perception_publisher.py`**: A ROS1 node that publishes time-based
   phase sequences on `/yolo/detections` and `/vlm/scene_label`, mimicking
   what the real YOLO + VLM pipeline would output upon observing the actor.

3. **`gem_safety` (ROS2)**: Subscribes to the same topics (via `ros1_bridge`
   in a full setup, or directly when using ros2_noetic_compat) and publishes
   `/safety/action_override` to the trajectory planner.

## Scenario Coordinate Reference

### track1.world (Scenarios 01-04)
- Vehicle spawn: `x=0, y=0, yaw=0` (facing +x)
- Obstacles placed along +x axis at 12–25 m range

### highbay_track.world (Scenario 05)
- Vehicle spawn: `x=0, y=0, yaw=0`
- Oval loop; obstacles distributed at coordinates matching the road:
  - Zone A (cones/pedestrian): `x≈-11 to -14, y≈-15 to -23`
  - Zone B (stop sign / worker): `x≈-22 to -27, y≈-19`
  - Zone D (back straight): `x≈-32, y≈-17 to -23`

> **Coordinate tuning**: If the vehicle spawn position differs on your machine,
> adjust `xyz` values in the YAML files to keep obstacles in the vehicle's path.

## Evaluation Metrics

| Metric | Definition |
|--------|------------|
| Collision rate | # contacts with actors/objects / total runs |
| Correct action rate | # zones with correct action (per `/safety/action_override`) / total zones |
| Time-to-action | ms from VLM label publish to vehicle response |
| False positive rate | # unwarranted STOP events when path is clear |

## Adding Custom Scenarios

1. Create `scenes/scenario_NN_<name>.yaml` following the existing schema:
   ```yaml
   objects:        # static or dynamic objects (spawn_objects.py)
     - name: ...
       source: {type: sdf|fuel|mesh, uri: "..."}
       xyz: [x, y, z]
       rpy: [r, p, y]
       static: true|false

   agents:         # animated actors / rigid bodies (spawn_agents.py)
     - name: ...
       source: {type: fuel, uri: "..."}
       trajectory: [[t, x, y, z, r, p, yaw], ...]
   ```

2. Create `launch/scenario_NN_<name>.launch` mirroring any existing one.

3. Add a scenario entry to `mock_perception_publisher.py` under `SCENARIOS`.
