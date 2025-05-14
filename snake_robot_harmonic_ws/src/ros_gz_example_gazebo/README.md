while true; do
    fswatch -1 /home/harikrishnan/ROS_PROJECTS/snake_robot_harmonic_ws/src/ros_gz_example_description/models/snakebot/snake_robot_V2.xacro
    ros2 param set /robot_state_publisher robot_description "$(xacro /home/harikrishnan/ROS_PROJECTS/snake_robot_harmonic_ws/src/ros_gz_example_description/models/snakebot/snake_robot_V2.xacro)"
    echo "Parameter updated. Watching for changes..."
done
