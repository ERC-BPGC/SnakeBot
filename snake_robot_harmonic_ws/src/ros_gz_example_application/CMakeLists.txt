cmake_minimum_required(VERSION 3.5)
project(ros_gz_example_application)

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# Install Python scripts
install(
  PROGRAMS
    src/publish_joint_angles.py
    src/publish_joint_angles_cpg_from_paper_faked.py  # List your Python node(s) here
    src/snakebot_cpg_ros2_keycontrol.py
    src/compute_com.py
    # List your Python node(s) here
  DESTINATION lib/${PROJECT_NAME}  # Install to the appropriate ROS 2 path
)

# Declare this as a ROS 2 package
ament_package()
