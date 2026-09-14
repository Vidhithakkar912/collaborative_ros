import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
def generate_launch_description():
    nav2_yaml_0= os.path.join(get_package_share_directory("localization_server"),'config',
    'nav_config_0.yaml')
    nav2_yaml_1= os.path.join(get_package_share_directory("localization_server"),'config',
        'nav_config_1.yaml')
    #map_file=os.path.join(get_package_share_directory('map_server'),'config','map.yaml')
  

    return LaunchDescription([
        # Node(
        #     package='nav2_map_server',
        #     executable='map_server',
        #     output='screen',
        #     parameters=[{'use_sim_time':True},
        #                 {'topic_name':'map'},
        #                 {'frame_id':'map'},
        #                 {'yaml_filename':map_file}],
        #     ),
        Node(   namespace='TB3_1',
                package='nav2_controller',
                executable='controller_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
               
            ),
        Node(
                namespace='TB3_1',
                package='nav2_smoother',
                executable='smoother_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
               
            ),
        Node(
                namespace='TB3_1',
                package='nav2_planner',
                executable='planner_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
                
            ),
        Node(
                namespace='TB3_1',
                package='nav2_behaviors',
                executable='behavior_server',
                name='behavior_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
                
            ),
         Node(
                namespace='TB3_1',
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
               
            ),

            Node(
                namespace='TB3_1',
                package='nav2_waypoint_follower',
                executable='waypoint_follower',
                name='waypoint_follower',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
                
               
            ),

            Node(
                namespace='TB3_1',
                package='nav2_velocity_smoother',
                executable='velocity_smoother',
                name='velocity_smoother',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
                
            ), 
        Node(   namespace='TB3_2',
                package='nav2_controller',
                executable='controller_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
                
            ),
        Node(
                namespace='TB3_2',
                package='nav2_smoother',
                executable='smoother_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
               
            ),
        Node(
                namespace='TB3_2',
                package='nav2_planner',
                executable='planner_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
                
            ),
        Node(
                namespace='TB3_2',
                package='nav2_behaviors',
                executable='behavior_server',
                name='behavior_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
               
            ),
         Node(
                namespace='TB3_2',
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
      
            ),

            Node(
                namespace='TB3_2',
                package='nav2_waypoint_follower',
                executable='waypoint_follower',
                name='waypoint_follower',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
             
            ),

            Node(
                namespace='TB3_2',
                package='nav2_velocity_smoother',
                executable='velocity_smoother',
                name='velocity_smoother',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
                
             
            ), 

        

        # Node(
        #     package='nav2_lifecycle_manager',
        #     executable='lifecycle_manager',
        #     name='lifecycle_manager_localization',
        #     output='screen',
        #     parameters=[{'use_sim_time':True},{'autostart':True},
        #                 {'bond_timeout':0.0},
        #                 {'node_names':['map_server']}],
        #     ),
         Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{'use_sim_time':True},{'autostart':True},
                        {'bond_timeout':0.0},
                        {'node_names':['TB3_1/controller_server','TB3_1/smoother_server',
                                       'TB3_1/planner_server','TB3_1/behavior_server','TB3_1/bt_navigator','TB3_1/waypoint_follower',
                                       'TB3_1/velocity_smoother','TB3_2/controller_server','TB3_2/smoother_server',
                                       'TB3_2/planner_server','TB3_2/behavior_server','TB3_2/bt_navigator','TB3_2/waypoint_follower',
                                       'TB3_2/velocity_smoother']}],
            ),
        
    ])
