# Copyright (c) 2022 Macenski, T. Foote, B. Gerkey, C. Lalancette, W. Woodall, “Robot Operating System 2: Design, architecture, and uses in the wild,” Science Robotics vol. 7, May 2022.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================================================
# Copyright (c) 2007 Open Robotics
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# =======================================================================================================
# NOTICE: e-Infochips Private Limited has developed code based on the ROS2 package.

# Copyright (c) 2024 e-Infochips Private Limited
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:#
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation  #and or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software #without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE #IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE #LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS #OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT #LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH #DAMAGE.
# If any term or provision set forth herein is deemed to be invalid, illegal, or unenforceable in any jurisdiction, such invalidity, illegality, or unenforceability will not affect any other term or provision or invalidate or render unenforceable such term or provision in any other jurisdiction. Upon a court determination that any term or provision is invalid, illegal, or unenforceable, the court may modify these terms and conditions to affect our original intent as closely as possible in order that the transactions contemplated hereby be consummated to the greatest extent possible as originally contemplated.
# =========================================================================================================

import os
import sys
import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch_ros.actions import Node, SetRemap
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource




def generate_launch_description():
    prefix_address = get_package_share_directory('turtlebot3_navigation2')
    config_directory = "/home/vidhi123/tb3_ws/src/turtlebot3/turtlebot3_navigation2/config"
    slam_config_basename = "2d_slam.lua"
    localization_config_basename =  "2d_localization.lua"
    res = LaunchConfiguration('resolution', default='0.05')
    publish_period = LaunchConfiguration('publish_period_sec', default='1.0')
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam = LaunchConfiguration('slam')
    nav2_launch_dir = os.path.join(prefix_address, 'launch')
    params_file = os.path.join(prefix_address, 'config', NAV_CONFIG_FILE)
    

    navigation_launch_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, 'navigation_launch.py')),
        launch_arguments={
            'params_file': params_file,
        }.items(),
    )

    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    print("CONFIG DIR:", config_directory)
    print("SLAM FILE :", slam_config_basename)
    return LaunchDescription(
        [
            SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
            SetRemap(src='/tf_static', dst='tf_static'),
            SetRemap(src='/tf', dst='tf'),
            navigation_launch_cmd,
            launch.actions.DeclareLaunchArgument(
                name='use_sim_time',
                default_value='True',
                description='Flag to enable use_sim_time',
            ),
            launch.actions.DeclareLaunchArgument(
                name='slam', default_value='True', description='Flag to enable use_sim_time'
            ),
            DeclareLaunchArgument(
                'resolution', default_value=res, description='configure the resolution'
            ),
            DeclareLaunchArgument(
                'publish_period_sec',
                default_value=publish_period,
                description='publish period in seconds',
            ),
            ################### cartographer_ros_node ###################
            DeclareLaunchArgument(
                'configuration_directory',
                default_value=config_directory,
                description='path to the .lua files',
            ),
            DeclareLaunchArgument(
                'slam_configuration_basename',
                default_value=slam_config_basename,
                description='name of .lua file to be used',
            ),
            DeclareLaunchArgument(
                'localization_configuration_basename',
                default_value=localization_config_basename,
                description='name of .lua file to be used',
            ),
            Node(
                package='cartographer_ros',
                condition=IfCondition(slam),
                executable='cartographer_node',
                name='as21_cartographer_node',
                arguments=[
                    '-configuration_directory',
                    config_directory,
                    '-configuration_basename',
                    slam_config_basename,
                ],
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen',
            ),
            Node(
                package='cartographer_ros',
                condition=IfCondition(PythonExpression(['not ', slam])),
                executable='cartographer_node',
                name='as21_cartographer_node',
                arguments=[
                    '-configuration_directory',
                    config_directory,
                    '-configuration_basename',
                    localization_config_basename,
                ],
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen',
            ),
            Node(
                package='cartographer_ros',
                condition=IfCondition(slam),
                executable='cartographer_occupancy_grid_node',
                name='cartographer_occupancy_grid_node',
                arguments=['-resolution', res, '-publish_period_sec', publish_period],
            ),
        ]
    )

