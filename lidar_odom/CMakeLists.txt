cmake_minimum_required(VERSION 3.0.2)
project(lidar_odom)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  tf
  pcl_ros
  pcl_conversions
)

find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs tf pcl_ros pcl_conversions
  DEPENDS PCL
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

add_executable(lidar_system_node src/lidar_system_node.cpp)
target_link_libraries(lidar_system_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(lidar_odometry src/lidar_odometry.cpp)
target_link_libraries(lidar_odometry ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES})

add_executable(test_odometry src/test_odometry.cpp)
target_link_libraries(test_odometry ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES})

add_executable(test_loop src/test_loop.cpp)
target_link_libraries(test_loop ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES})

add_executable(test_offline_loop src/test_offline_loop.cpp)
target_link_libraries(test_offline_loop ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES})
