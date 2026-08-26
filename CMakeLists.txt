cmake_minimum_required(VERSION 3.8)
project(renee_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(renee_scan_planner
  src/circular_pattern.cpp
  src/scan_config_loader.cpp
  src/scan_plan_generator.cpp
  src/scan_plan_writer.cpp
)

target_include_directories(renee_scan_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(renee_scan_planner
  geometry_msgs
)

target_link_libraries(renee_scan_planner
  yaml-cpp
)

add_executable(planner_node
  src/main.cpp
  src/planner_node.cpp
)

target_include_directories(planner_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(planner_node
  rclcpp
  geometry_msgs
)

target_link_libraries(planner_node
  renee_scan_planner
)

install(TARGETS
  renee_scan_planner
  planner_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/plot_scan_waypoints.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  find_package(ament_cmake_gtest REQUIRED)

  add_test(
    NAME test_plot_structure
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_plot_structure.py
  )
  set_tests_properties(test_plot_structure PROPERTIES ENVIRONMENT "PYTHONNOUSERSITE=1")

  ament_add_gtest(test_scan_plan_generator
    test/test_scan_plan_generator.cpp
  )
  target_link_libraries(test_scan_plan_generator
    renee_scan_planner
  )
  ament_target_dependencies(test_scan_plan_generator
    geometry_msgs
  )
  target_include_directories(test_scan_plan_generator PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
