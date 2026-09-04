import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
def generate_launch_description():
    nav2_yaml_0= os.path.join(get_package_share_directory("localization_server"),'config',
    'manipulator.yaml')
    nav2_yaml_1= os.path.join(get_package_share_directory("localization_server"),'config',
        'transporter.yaml')
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
        Node(   namespace='manipulator',
                package='nav2_controller',
                executable='controller_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
               
            ),
        Node(
                namespace='manipulator',
                package='nav2_smoother',
                executable='smoother_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
               
            ),
        Node(
                namespace='manipulator',
                package='nav2_planner',
                executable='planner_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
                
            ),
        Node(
                namespace='manipulator',
                package='nav2_behaviors',
                executable='behavior_server',
                name='behavior_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
                
            ),
         Node(
                namespace='manipulator',
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
               
            ),

            Node(
                namespace='manipulator',
                package='nav2_waypoint_follower',
                executable='waypoint_follower',
                name='waypoint_follower',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
                
               
            ),

            Node(
                namespace='manipulator',
                package='nav2_velocity_smoother',
                executable='velocity_smoother',
                name='velocity_smoother',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_0],
                
            ), 
        Node(   namespace='transporter',
                package='nav2_controller',
                executable='controller_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
                
            ),
        Node(
                namespace='transporter',
                package='nav2_smoother',
                executable='smoother_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
               
            ),
        Node(
                namespace='transporter',
                package='nav2_planner',
                executable='planner_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
                
            ),
        Node(
                namespace='transporter',
                package='nav2_behaviors',
                executable='behavior_server',
                name='behavior_server',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
               
            ),
         Node(
                namespace='transporter',
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
      
            ),

            Node(
                namespace='transporter',
                package='nav2_waypoint_follower',
                executable='waypoint_follower',
                name='waypoint_follower',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
             
            ),

            Node(
                namespace='transporter',
                package='nav2_velocity_smoother',
                executable='velocity_smoother',
                name='velocity_smoother',
                output='screen',
                respawn_delay=2.0,
                parameters=[nav2_yaml_1],
                
             
            ), 

         Node(
                    namespace='manipulator',
                    package='nav2_amcl',
                    executable='amcl',
                    output='screen',
                    parameters=[nav2_yaml_0],
                    ),
                 Node(
                            namespace='transporter',
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
                         {'node_names':['map_server','manipulator/amcl','transporter/amcl']}],
             ),
         Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{'use_sim_time':True},{'autostart':True},
                        {'bond_timeout':0.0},
                        {'node_names':['manipulator/controller_server','manipulator/smoother_server',
                                       'manipulator/planner_server','manipulator/behavior_server','manipulator/bt_navigator','manipulator/waypoint_follower',
                                       'manipulator/velocity_smoother','transporter/controller_server','transporter/smoother_server',
                                       'transporter/planner_server','transporter/behavior_server','transporter/bt_navigator','transporter/waypoint_follower',
                                       'transporter/velocity_smoother']}],
            ),
        
    ])
