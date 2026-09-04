#/usr/bin/env python3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# Authors: Arshad Mehmood

import os
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    RegisterEventHandler,
    EmitEvent,
    ExecuteProcess,
)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from multi_robots.utils import load_sdf_with_namespace, create_namespaced_bridge_yaml


def generate_launch_description():
    # Paths
    tb3_multi_dir = get_package_share_directory('multi_robots')
    ros_gz_sim_dir = get_package_share_directory('ros_gz_sim')

    # Simulation config
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_path = os.path.join(tb3_multi_dir, 'worlds', 'tb3_world.world')

    # Launch Gazebo server using ExecuteProcess for clean kill control
    gzserver_cmd = ExecuteProcess(
        cmd=['ruby', 
             os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py'),
        ],
        output='screen',
        sigterm_timeout='5',
        sigkill_timeout='5',
    )

    # Use IncludeLaunchDescription but wrap with shutdown handler
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r -s -v2 {world_path}',
            'on_exit_shutdown': 'true'
        }.items()
    )
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': '-g -v2',
            'on_exit_shutdown': 'true'
        }.items()
    )

    # ✅ This is the key fix:
    # On shutdown (Ctrl+C), forcefully kill all gz/ign child processes
    # that may not respond to normal SIGINT propagation.
    kill_gz_on_shutdown = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                ExecuteProcess(
                    cmd=['bash', '-c',
                         'pkill -SIGTERM -f "gz sim" || true; '
                         'pkill -SIGTERM -f "ign gazebo" || true; '
                         'sleep 1; '
                         'pkill -SIGKILL -f "gz sim" || true; '
                         'pkill -SIGKILL -f "ign gazebo" || true; '
                         'pkill -SIGKILL -f "ruby.*gz_sim" || true'
                    ],
                    output='screen',
                )
            ]
        )
    )

    # Main LaunchDescription
    ld = LaunchDescription()
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(kill_gz_on_shutdown)  # ✅ Register the shutdown handler

    # Load robot config
    robot_config_path = os.path.join(tb3_multi_dir, 'config', 'robots.yaml')
    with open(robot_config_path, 'r') as f:
        config = yaml.safe_load(f)

    robots = [r for r in config['robots'] if r.get('enabled', True)]
    tb3_model = os.environ.get('TURTLEBOT3_MODEL', 'waffle')
    model_dir = f'turtlebot3_{tb3_model}'
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
    frame_prefix = LaunchConfiguration('frame_prefix', default='')
    urdf_file_name = 'turtlebot3_' + tb3_model + '.urdf'
    urdf_path = os.path.join(tb3_multi_dir, 'urdf', urdf_file_name)

    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    for robot in robots:
        namespace = robot['name']

        sdf_path = os.path.join(tb3_multi_dir, 'models', model_dir, 'model.sdf')
        patched_sdf = load_sdf_with_namespace(sdf_path, namespace)

        robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=namespace,
            remappings=remappings,
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_desc,
            }])

        spawner_node = Node(
            package='ros_gz_sim',
            executable='create',
            namespace=namespace,
            arguments=[
                '-name', f'{namespace}_{tb3_model}',
                '-string', patched_sdf,
                '-x', str(robot['x_pose']),
                '-y', str(robot['y_pose']),
                '-z', '0.01',
            ],
            output='screen',
        )

        bridge_template = os.path.join(tb3_multi_dir, 'params', f'{tb3_model}_bridge.yaml')
        namespaced_bridge = create_namespaced_bridge_yaml(bridge_template, namespace)

        bridge_node = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['--ros-args', '-p', f'config_file:={namespaced_bridge}'],
            output='screen',
        )
        rgb_image_bridge = None
        depth_image_bridge = None

        if tb3_model == 'waffle':
            # RGB camera
            rgb_image_bridge = Node(
            package='ros_gz_image',
            executable='image_bridge',
            namespace=namespace,
            arguments=['/' + namespace + '/camera/image_raw'],
            output='screen',
        )

    # Depth camera
            depth_image_bridge = Node(
            package='ros_gz_image',
            executable='image_bridge',
            namespace=namespace,
            arguments=['/' + namespace + '/camera/depth/image_raw'],
             output='screen',
            )

        ld.add_action(robot_state_publisher)
        ld.add_action(spawner_node)
        ld.add_action(bridge_node)
        if rgb_image_bridge:    
            ld.add_action(rgb_image_bridge)
        if depth_image_bridge:    
            ld.add_action(depth_image_bridge)
    # Global clock bridge
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        output='screen',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
    )
    ld.add_action(clock_bridge)

    # Add GZ model path to env
    ld.add_action(AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(tb3_multi_dir, 'models'))
    )

    return ld