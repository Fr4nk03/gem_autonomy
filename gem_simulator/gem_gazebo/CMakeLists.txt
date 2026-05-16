cmake_minimum_required(VERSION 2.8.3)

project(gem_gazebo)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  ackermann_msgs
  std_msgs
  septentrio_gnss_driver
)

catkin_package()

catkin_install_python(PROGRAMS scripts/gem_control.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS scripts/gem_sensor_info.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS scripts/insnavgeod_publisher.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS scripts/yolo_detector.py
                                scripts/stop_sign_behavior.py
                                scripts/vlm_node.py
                                scripts/spawn_objects.py
                                scripts/spawn_agents.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY config launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
