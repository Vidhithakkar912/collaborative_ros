
#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command


def generate_launch_description():



    tb3_gazebo = get_package_share_directory(
        'turtlebot3_gazebo'
    )

    tb3_manipulation = get_package_share_directory(
        'turtlebot3_manipulation_gazebo'
    )

    gazebo_ros = get_package_share_directory(
        'gazebo_ros'
    )


    world = os.path.join(
        tb3_gazebo,
        'worlds',
        'turtlebot3_world.world'
    )


    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                gazebo_ros,
                'launch',
                'gzserver.launch.py'
            )
        ),
        launch_arguments={
            'world': world
        }.items()
    )


    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                gazebo_ros,
                'launch',
                'gzclient.launch.py'
            )
        )
    )


    waffle_model = os.path.join(
        tb3_gazebo,
        'models',
        'turtlebot3_waffle',
        'model.sdf'
    )

    transporter = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',

        arguments=[
            '-entity',
            'transporter',

            '-file',
            waffle_model,

            '-x',
            '-2.0',

            '-y',
            '-0.5',

            '-z',
            '0.01',

            '-robot_namespace',
            'transporter'
        ],

        output='screen'
    )
    
    
    transporter_urdf = os.path.join(
            tb3_gazebo,
            'urdf',
            'turtlebot3_waffle.urdf'
        )

    with open(transporter_urdf, 'r') as f:
            transporter_description = f.read()

    transporter_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace='transporter',
            parameters=[
                {
                    'robot_description': transporter_description,
                    #'frame_prefix': 'transporter/',
                    'prefix': 'transporter/',
                    'use_sim_time': True
                }
            ],
            remappings=[
                ('/joint_states', '/transporter/joint_states')
            ],
            output='screen'
        )


    manipulator_xacro = os.path.join(
        tb3_manipulation,
        'urdf',
        'turtlebot3_manipulation.urdf.xacro'
    )

    manipulator_description = Command([
        'xacro',
        ' ',
        manipulator_xacro,
        ' ',
        'prefix:=manipulator/',
        ' ',
        'use_sim:=true',
        ' ',
        'use_fake_hardware:=false',
        ' ',
        'fake_sensor_commands:=false'
    ])

 
    manipulator_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',

        namespace='manipulator',

        parameters=[
            {
                'robot_description': manipulator_description,
                'use_sim_time': True,

                # 'frame_prefix': 'manipulator/'
            }
        ],

        output='screen'
    )

  
    manipulator = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',

        arguments=[
            '-entity',
            'manipulator',

            '-topic',
            '/manipulator/robot_description',

            '-x',
            '2.0',

            '-y',
            '0.5',

            '-z',
            '0.01',

            '-robot_namespace',
            'manipulator'
        ],

        output='screen'
    )
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/manipulator/controller_manager'],
        output='screen'
    )

    
    load_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', '--controller-manager', '/manipulator/controller_manager'],
        output='screen'
    )
    load_gripper_controller = Node( package='controller_manager', executable='spawner', arguments=[ 'gripper_controller', '--controller-manager', '/manipulator/controller_manager' ], output='screen' )

    
    load_controllers_callback = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=manipulator,
            on_exit=[load_joint_state_broadcaster, load_arm_controller,load_gripper_controller]
        )
    )
  

    return LaunchDescription([

        
        gzserver,
        gzclient,

        
        transporter,
        transporter_state_publisher,
       
        manipulator_state_publisher,
        manipulator,
        load_controllers_callback
    ])

