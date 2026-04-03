cmake_minimum_required(VERSION 3.8)
project(panda_ik_benchmark_viz_rviz_plugins)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# RViz2 (Jazzy+) supports both Qt5 and Qt6; try Qt5 first, fallback to Qt6.
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

add_library(ik_benchmark_playback_panel SHARED
  src/ik_benchmark_playback_panel.cpp
  # Add the header to the target sources so CMake's AUTOMOC can see Q_OBJECT.
  # Without this, moc code may not be generated/linked, leading to a
  # dlopen-time "undefined symbol: vtable" error when RViz loads the plugin.
  include/panda_ik_benchmark_viz_rviz_plugins/ik_benchmark_playback_panel.hpp
)

# Be explicit: this library needs Qt moc.
set_target_properties(ik_benchmark_playback_panel PROPERTIES
  AUTOMOC ON
)

target_include_directories(ik_benchmark_playback_panel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Some toolchains allow unresolved symbols in shared libraries which then fail
# only when RViz tries to dlopen() the plugin. Make that a build-time error.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(ik_benchmark_playback_panel PRIVATE "-Wl,--no-undefined")
endif()

ament_target_dependencies(ik_benchmark_playback_panel
  pluginlib
  rclcpp
  rviz_common
  std_msgs
  sensor_msgs
)

target_link_libraries(ik_benchmark_playback_panel
  Qt${QT_VERSION_MAJOR}::Widgets
)

# Register the plugin description file so rviz2 can find the panel.
pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

install(TARGETS ik_benchmark_playback_panel
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(ik_benchmark_playback_panel)

ament_package()
