from launch import LaunchDescription
from launch_ros.actions import Node

PKG = "multi_robots"


def generate_launch_description():

    planner_node = Node(
        package=PKG,
        executable="planner",
        name="multi_robot_planner",
        output="screen",
    )

    return LaunchDescription([
        planner_node
    ])