cmake_minimum_required(VERSION 2.8.3)
project(gem_dwa_sim)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  ackermann_msgs
  visualization_msgs
  tf
  gazebo_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
