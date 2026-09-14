#/usr/bin/env python3
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    IncludeLaunchDescription,
    RegisterEventHandler,
    ExecuteProcess,
)
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Note: load_sdf_with_namespace might need adjustment for Gazebo Classic SDF requirements
from multi_robots.utils import load_sdf_with_namespace 

def generate_launch_description():
    # Paths
    tb3_multi_dir = get_package_share_directory('multi_robots')
    # CHANGED: Use gazebo_ros instead of ros_gz_sim
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')

    # Simulation config
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_path = os.path.join(tb3_multi_dir, 'worlds', 'tb3_world.world')

    # CHANGED: Launch Gazebo Classic Server
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_path}.items()
    )

    # CHANGED: Launch Gazebo Classic Client
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gzclient.launch.py')
        )
    )

    # UPDATED: Shutdown handler for Gazebo Classic processes
    kill_gz_on_shutdown = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                ExecuteProcess(
                    cmd=['pkill', '-9', 'gzserver'],
                    output='screen',
                ),
                ExecuteProcess(
                    cmd=['pkill', '-9', 'gzclient'],
                    output='screen',
                )
            ]
        )
    )

    ld = LaunchDescription()
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(kill_gz_on_shutdown)

    # Load robot config
    robot_config_path = os.path.join(tb3_multi_dir, 'config', 'robots.yaml')
    with open(robot_config_path, 'r') as f:
        config = yaml.safe_load(f)

    robots = [r for r in config['robots'] if r.get('enabled', True)]
    tb3_model = os.environ.get('TURTLEBOT3_MODEL', 'waffle')
    model_dir = f'turtlebot3_{tb3_model}'
    
    urdf_file_name = 'turtlebot3_' + tb3_model + '.urdf'
    urdf_path = os.path.join(tb3_multi_dir, 'urdf', urdf_file_name)

    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    for robot in robots:
        namespace = robot['name']

        # Robot State Publisher remains the same
        robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=namespace,
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_desc,
                'frame_prefix': namespace + '/'  # Added for multi-robot TF clarity
            }])

        # CHANGED: Use gazebo_ros spawn_entity instead of ros_gz_sim create
        spawner_node = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            namespace=namespace,
            arguments=[
                '-entity', f'{namespace}_{tb3_model}',
                '-file', os.path.join(tb3_multi_dir, 'models', model_dir, 'model.sdf'),
                '-x', str(robot['x_pose']),
                '-y', str(robot['y_pose']),
                '-z', '0.01',
                '-robot_namespace', namespace
            ],
            output='screen',
        )

        ld.add_action(robot_state_publisher)
        ld.add_action(spawner_node)
        
        # NOTE: Parameter Bridge and Image Bridges are REMOVED. 
        # Gazebo Classic handles this via plugins inside the URDF/SDF.

    # CHANGED: Environment variable for Gazebo Classic
    ld.add_action(AppendEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        os.path.join(tb3_multi_dir, 'models'))
    )

    return ld
