from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='merge_map', # Change this if your package name in CMakeLists.txt is different
            executable='merge_map', # Or whatever the executable name is defined as in your cloned repo
            name='map_merge_node',
            output='screen',
            parameters=[{
                'robot_map_topic': 'map',
                'world_frame': 'map',
                'known_init_poses': False,  # Set to True if you want to define static starting offsets
                'merging_rate': 2.0,        # Frequency to merge maps (Hz)
                'discovery_rate': 0.05,
                'estimation_rate': 0.5,
                'independent_resolution': False
            }],
            remappings=[
                ('/map1', '/TB3_1/map'),
                ('/map2', '/TB3_2/map'),
                ('/merge_map', '/map')
            ]
        )
    ])
