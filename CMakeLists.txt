cmake_minimum_required(VERSION 3.0.2)
project(mando_vision_2026)

find_package(catkin REQUIRED COMPONENTS
  rospy std_msgs sensor_msgs geometry_msgs vision_msgs
  cv_bridge image_transport tf2_ros
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs vision_msgs
                 cv_bridge image_transport tf2_ros
)

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  src/lane_detector.py
  src/traffic_light_detector.py
  src/object_detector.py
  src/stopline_detector.py
  src/detection_monitor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch urdf weights tools docs
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "*.pt" EXCLUDE
)
