################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.5)
project(turtlebot3_manipulation_navigation2)

################################################################################
# Find ament packages and libraries for ament and system dependencies
################################################################################
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
add_executable(cmd_vel_bridge src/cmd_vel_bridge.cpp)
ament_target_dependencies(cmd_vel_bridge
  rclcpp
  geometry_msgs
)
install(TARGETS
  cmd_vel_bridge
  DESTINATION lib/${PROJECT_NAME}
)

################################################################################
# Install
################################################################################
install(
  DIRECTORY launch map param rviz
  DESTINATION share/${PROJECT_NAME}
)

################################################################################
# Macro for ament package
################################################################################
ament_package()
