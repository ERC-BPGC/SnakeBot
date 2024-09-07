import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():
    # Get the path to the URDF file
    urdf_file_name = 'snake_robot.urdf'
    urdf_file_path = "/home/harikrishnan/ROS_PROJECTS/snake_bot_ws/src/snake_bot/urdf/snake_robot.urdf"

    # Check if the file exists
    if not os.path.exists(urdf_file_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_file_path}")

    # Create a RobotStatePublisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_file_path]),
            'use_sim_time': False
        }]
    )

    # Create an RViz2 node
    rviz2_node =Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(get_package_share_directory('snake_bot'), 'rviz', 'rviz_basic_settings.rviz')],
            output='screen',
        )

    # Create a URDF updater node
    urdf_updater_node = Node(
        package='snake_bot',
        executable='urdf_updater.py',
        name='urdf_updater',
        output='screen',
        parameters=[{'urdf_file_path': urdf_file_path}]
    )

    # Return the LaunchDescription
    return LaunchDescription([
        robot_state_publisher_node,
        rviz2_node,
        urdf_updater_node,
    ])