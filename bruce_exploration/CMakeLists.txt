cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(bruce_exploration)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(CATKIN_PACKAGES
  roscpp
  nav_msgs
  sensor_msgs
  move_base_msgs
  grid_map_ros
  bruce_msgs
  pluginlib
  navfn
)

find_package(catkin REQUIRED COMPONENTS ${CATKIN_PACKAGES})
find_package(GTSAM REQUIRED)
set(GTSAM_LIBRARIES gtsam)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS ${CATKIN_PACKAGES}
  DEPENDS GTSAM
)
include_directories(
  include
  ${GTSAM_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/3rdparty/halton.cpp
  src/3rdparty/cubic_spline_interpolator.cpp
  src/bruce_exploration/base/math.cpp
  src/bruce_exploration/base/targets.cpp
  src/bruce_exploration/base/cost_map.cpp
  src/bruce_exploration/base/virtual_map.cpp
  src/bruce_exploration/base/utility_function.cpp
  src/bruce_exploration/base/belief_propagation.cpp
  src/bruce_exploration/base/path_library.cpp
)
target_link_libraries(${PROJECT_NAME} ${GTSAM_LIBRARIES} ${catkin_LIBRARIES})

add_executable(${PROJECT_NAME}_server 
  src/bruce_exploration/exploration_server.cpp
  src/bruce_exploration/planner/nf_exploration_server.cpp
  src/bruce_exploration/planner/nbv_exploration_server.cpp
  src/bruce_exploration/planner/heuristic_exploration_server.cpp
  src/bruce_exploration/planner/em_exploration_server.cpp
)
target_link_libraries(${PROJECT_NAME}_server ${PROJECT_NAME} ${GTSAM_LIBRARIES} ${catkin_LIBRARIES})

add_executable(${PROJECT_NAME}_client src/bruce_exploration/exploration_client.cpp)
target_link_libraries(${PROJECT_NAME}_client ${catkin_LIBRARIES})

add_library(${PROJECT_NAME}_planner src/bruce_exploration/exploration_planner.cpp)
target_link_libraries(${PROJECT_NAME}_planner ${catkin_LIBRARIES})
