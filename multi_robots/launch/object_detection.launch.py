from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python import get_package_share_directory
def generate_launch_description():

    
    manipulator_object_detection_node=Node(
        package="multi_robots",
        executable="object_detection",
        namespace="manipulator",
        name="manipulator_object_detection_node",
        output="screen",
        parameters=[
            {"use_sim_time": True}
        ]
    )
    transporter_object_detection_node=Node(
            package="multi_robots",
            executable="object_detection",
            namespace="transporter",
            name="transporter_object_detection_node",
            output="screen",
            parameters=[
                        {"use_sim_time": True}
                    ]
        )
    return LaunchDescription([
    
            
            manipulator_object_detection_node,
            transporter_object_detection_node
        ])

    
