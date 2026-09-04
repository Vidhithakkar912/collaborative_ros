import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
def generate_launch_description():
    nav2_yaml_0= os.path.join(get_package_share_directory("localization_server"),'config',
    'amcl_config_0.yaml')
    nav2_yaml_1= os.path.join(get_package_share_directory("localization_server"),'config',
        'amcl_config_1.yaml')
    map_file=os.path.join(get_package_share_directory('map_server'),'config','map.yaml')
    return LaunchDescription([
        Node(
            package='nav2_map_server',
            executable='map_server',
            output='screen',
            parameters=[{'use_sim_time':True},
                        {'topic_name':'map'},
                        {'frame_id':'map'},
                        {'yaml_filename':map_file}],
            ),
        Node(
            namespace='TB3_1',
            package='nav2_amcl',
            executable='amcl',
            output='screen',
            parameters=[nav2_yaml_0],
            ),
         Node(
                    namespace='TB3_2',
                    package='nav2_amcl',
                    executable='amcl',
                    output='screen',
                    parameters=[nav2_yaml_1],
            ),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{'use_sim_time':True},{'autostart':True},
                        {'bond_timeout':0.0},
                        {'node_names':['map_server','TB3_1/amcl','TB3_2/amcl']}],
            ),
    ])