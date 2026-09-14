cmake_minimum_required(VERSION 3.5)
project(slam_toolbox_multi_robot_demo)

find_package(ament_cmake REQUIRED)
find_package(slam_toolbox REQUIRED)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY rviz DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
