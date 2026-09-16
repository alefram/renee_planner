cmake_minimum_required(VERSION 3.8)
project(renee_trajectory_generation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/hqp_scan_optimizer.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES requirements-hqp.txt README.md DESTINATION share/${PROJECT_NAME})

ament_package()
