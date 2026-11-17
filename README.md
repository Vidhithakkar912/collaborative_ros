# MORSE(Multi-robot Orchestration & Reasoning Based Execution
## **1. Introduction**

With the increasing adoption of autonomous mobile robots and intelligent robotic systems, coordinating multiple robots to perform different tasks within a shared environment has become an important area of robotics development. In applications involving heterogeneous robots, different platforms may have different capabilities, such as autonomous navigation, object detection, robotic manipulation, and teleoperation. Managing these robots independently often requires separate interfaces and manual task assignments, making coordination increasingly complex as the number of robots and supported operations grows.

To address this challenge, we developed **MORSE (Multi-robot Orchestration, Reasoning and Synergistic Execution)**, a centralized, LLM-powered multi-robot coordination framework designed to manage and coordinate multiple robots through a unified system. The framework uses a locally hosted Large Language Model (LLM), such as Qwen2.5 3B or Phi-3 Mini, running on a host system to understand natural-language instructions, reason about task requirements, and determine suitable robots for task execution.

MORSE introduces a centralized planner node that acts as the primary interface between user instructions and multiple robotic systems. Instead of requiring users to specify individual robot commands and manage each robot separately, the system allows them to provide tasks through a common interface. The planner uses the LLM to interpret the requested operation and reason about robot capabilities, current status, and location to determine an appropriate task assignment. If the user explicitly specifies a robot name, the planner considers that robot for the requested task. If no robot is specified, the system identifies a suitable robot based on its capabilities and proximity to the target location.

For communication between the host system and individual robots, MORSE integrates **Zenoh bridges** with their respective ROS 2 environments. This architecture enables the centralized planner to exchange task-related information and communicate with multiple robot systems without requiring the LLM to run independently on every robot. By keeping the LLM on the host system, the framework provides a common reasoning and coordination layer for the connected robotic platforms.

The framework is designed to support multiple robotic operations through a unified task coordination mechanism, including:

- **Autonomous Navigation:** Assigning navigation tasks to suitable robots based on their capabilities and current location.
- **Object Detection:** Triggering perception-based operations to identify and detect objects in the environment.
- **Pick-and-Place Operations:** Coordinating manipulation tasks using robots equipped with the required robotic manipulation capabilities.
- **Teleoperation:** Providing a common mechanism for initiating and managing manual robot control.

By combining local LLM-based reasoning, capability-aware robot selection, centralized task planning, and Zenoh-based communication, MORSE aims to simplify the management of heterogeneous robotic systems. The framework provides a foundation for coordinating multiple robots through a single interface, reducing the need for independent robot-level task management and enabling more flexible and intelligent multi-robot collaboration.
