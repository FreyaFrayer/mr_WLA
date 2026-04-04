cmake_minimum_required(VERSION 3.8)
project(panda_try_rviz)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(moveit_msgs REQUIRED)

find_package(Qt5 QUIET COMPONENTS Widgets)
if(Qt5_FOUND)
  set(QT_VERSION_MAJOR 5)
else()
  find_package(Qt6 REQUIRED COMPONENTS Widgets)
  set(QT_VERSION_MAJOR 6)
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_library(panda_try_control_panel SHARED
  src/panda_try_control_panel.cpp
  include/panda_try_rviz/panda_try_control_panel.hpp
)

set_target_properties(panda_try_control_panel PROPERTIES
  AUTOMOC ON
)

target_include_directories(panda_try_control_panel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(panda_try_control_panel PRIVATE "-Wl,--no-undefined")
endif()

ament_target_dependencies(panda_try_control_panel
  pluginlib
  rclcpp
  rviz_common
  sensor_msgs
  std_msgs
)

target_link_libraries(panda_try_control_panel
  Qt${QT_VERSION_MAJOR}::Widgets
)

add_executable(panda_try_backend_node
  src/panda_try_backend_node.cpp
)

ament_target_dependencies(panda_try_backend_node
  rclcpp
  sensor_msgs
  std_msgs
  visualization_msgs
  geometry_msgs
  tf2
  tf2_ros
  moveit_msgs
)

pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

install(TARGETS
  panda_try_control_panel
  panda_try_backend_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME}
)

install(FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(panda_try_control_panel)

ament_package()
