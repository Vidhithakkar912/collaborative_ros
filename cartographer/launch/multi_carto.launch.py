import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription


def generate_launch_description():
    config_1 = os.path.join(get_package_share_directory('cartographer'),'config')
  
    return LaunchDescription([

        Node(
            package = 'cartographer_ros',
            executable= 'cartographer_node',
            namespace= 'TB3_1',
            name = 'cartographer_node',
            output = 'screen',
            parameters=[{"use_sim_time": True}],
            remappings=[
                ("scan", "/TB3_1/scan"),
                ("odom", "/TB3_1/odom"),
            ],
            arguments=[
                "-configuration_directory", config_1,
                "-configuration_basename", "tb3_1.lua",
            ],
        ),
        Node(
                    package = 'cartographer_ros',
                    executable= 'cartographer_node',
                    namespace= 'TB3_2',
                    name = 'cartographer_node',
                    output = 'screen',
                    parameters=[{"use_sim_time":True}],
                    remappings=[
                        ("scan","/TB3_2/scan"),
                        ("odom","/TB3_2/odom"),
                    ],
                    arguments=[
                        "-configuration_directory",config_1,
                        "-configuration_basename","tb3_2.lua",
                    ],
                ),
        Node(
                package='cartographer_ros',
                executable='cartographer_occupancy_grid_node',
                namespace="TB3_1",
                name="occupancy_grid_node",
                  output="screen",
            parameters=[{"use_sim_time": True}],
            arguments=[
                "-resolution", "0.05",
                "-publish_period_sec", "1.0",
            ],
        
        ),

        Node(
            package="cartographer_ros",
            executable="cartographer_occupancy_grid_node",
            namespace="TB3_2",
            name="occupancy_grid_node",
            output="screen",
            parameters=[{"use_sim_time": True}],
            arguments=[
                "-resolution", "0.05",
                "-publish_period_sec", "1.0",
            ],
        ),
    ])

