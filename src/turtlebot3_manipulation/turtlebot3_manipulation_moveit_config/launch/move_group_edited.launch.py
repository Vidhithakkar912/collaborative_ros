import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    ld = LaunchDescription()
    declared_arguments = []
    
    
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )


def launch_setup(context, *args, **kwargs):

    moveit_config = (
        MoveItConfigsBuilder("turtlebot3_manipulation")
        .robot_description(file_path="urdf/turtlebot3_manipulation.urdf.xacro")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"]
        )
        .to_moveit_configs()
    )
    

    # Start the actual move_group node/action server
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

   
   

    
    

    # Static TF
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "odom", "base_footprint"],
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # ros2_control using FakeSystem as hardware
    # ros2_controllers_path = os.path.join(
    #     get_package_share_directory("open_manipulator_v5_moveit_config"),
    #     "config",
    #     "ros2_controllers.yaml",
    # )
    # ros2_control_node = Node(
    #     package="controller_manager",
    #     executable="ros2_control_node",
    #     parameters=[moveit_config.robot_description, ros2_controllers_path],
    #     output="both",
    # )

    # joint_state_broadcaster_spawner = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=[
    #         "joint_state_broadcaster",
    #         "--controller-manager-timeout",
    #         "300",
    #         "--controller-manager",
    #         "/controller_manager",
    #     ],
    # )

    # arm_controller_spawner = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=["arm_controller", "-c", "/controller_manager"],
    # )

    # hand_controller_spawner = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=["hand_controller", "-c", "/controller_manager"],
    # )
    nodes_to_start = [
        # static_tf,
        # robot_state_publisher,
        run_move_group_node,
        #ros2_control_node,
        #joint_state_broadcaster_spawner,
        #arm_controller_spawner,
        #hand_controller_spawner,
    ]

    return nodes_to_start