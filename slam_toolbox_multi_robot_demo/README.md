# slam_toolbox_multi_robot_demo

Out-of-the-box **multi-robot SLAM demo** for ROS 2 using `slam_toolbox`.  
Launch 2 TurtleBot3 robots in Gazebo (gz sim), each with a namespaced `slam_toolbox` instance and a shared **global odometry** frame (`global_odom → odom`).

---

![multirobot_slam_simulation](images/simulation.gif?raw=true "slam_toolbox_multi_robot_demo")

## Features

- Single-robot or **multi-robot** bringup (2 robots) with per-robot namespaces.  
- Starts `slam_toolbox` (decentralized multi-robot variant) per robot.  
- Publishes **static TF** per robot: `global_odom → odom` (new-style args).  
- Optional RViz per robot with a default config.

---

## Dependencies
**`slam_toolbox` (decentralized multi-robot variant)**
```bash
sudo apt install ros-kilted-diff-drive-controller ros-kilted-nav2-minimal-tb3-sim ros-kilted-teleop-twist-keyboard
```

---

## Build & Source

```bash
# From your workspace root
colcon build --packages-select slam_toolbox_multi_robot_demo
source install/setup.bash
```
