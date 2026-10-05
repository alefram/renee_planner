cmake_minimum_required(VERSION 3.8)
project(renee_trajectory_generation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Scan-pose generator interfaces. rosidl takes the kind from the parent folder
# (msg/, action/), so they can live under interfaces/.
rosidl_generate_interfaces(${PROJECT_NAME}
  "interfaces/msg/CameraPose.msg"
  "interfaces/msg/CameraTrajectory.msg"
  "interfaces/action/GenerateScanPoses.action"
  DEPENDENCIES geometry_msgs std_msgs
)

# The Python package (scan-pose generator, hqp/, legacy/). rosidl already
# calls ament_python_install_package(${PROJECT_NAME}) for the generated msg/
# and action/ modules, and a second call clashes (duplicate CMake targets), so
# the modules are installed into the same folder directly.
install(DIRECTORY ${PROJECT_NAME}/
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(DIRECTORY
  config
  launch
  behavior_trees
  DESTINATION share/${PROJECT_NAME}
)

# The viewer's committed Vite build (no Node needed to build the package).
install(DIRECTORY web/dist DESTINATION share/${PROJECT_NAME}/web)

install(PROGRAMS
  src/scan_pose_generator_node.py
  src/scan_viewer_server_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Legacy (frozen): its launch files keep their `ros2 launch` names, its
# configs are under share/<pkg>/legacy/config.
install(DIRECTORY legacy/launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY legacy/config DESTINATION share/${PROJECT_NAME}/legacy)
install(PROGRAMS
  legacy/src/trajectory_controller_node.py
  legacy/src/arm_static_planner_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES requirements.txt requirements-hqp.txt README.md DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
