import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration, Command

def generate_launch_description():

    # Check if we're told to use sim time
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Get the URDF/XACRO file path
    path_to_urdf = os.path.join(
        get_package_share_directory('ros_gz_example_description'), 
        'models', 
        'snakebot', 
        'snake_robot_scaleddown.xacro'
    )

    # Create a robot_state_publisher node
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(Command(['xacro ', str(path_to_urdf)]), value_type=str)
        }]
    )

    # Get the path to the custom world file (frictional_world.sdf)
    path_to_world = os.path.join(
        get_package_share_directory('ros_gz_example_description'),
        'models',
        'snakebot',
        'frictional_world.sdf'
    )

    # Launch Gazebo simulation with the custom world
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("ros_gz_sim"),
                    "launch",
                    "gz_sim.launch.py",
                )
            ]
        ),
        launch_arguments={"gz_args": [f"-r -v 4 {path_to_world}"]}.items(),
    )

    # Spawn the robot in Gazebo
    spawn_entity = Node(
    package="ros_gz_sim",
    executable="create",
    arguments=[
        "-name", "robot1",
        "-topic", "/robot_description",
        "-x", "0",
        "-y", "0",
        "-z", "0.09",
        "-R", "0",  # Roll = 0
        "-P", "0",  # Pitch = 0
        "-Y", "0"   # Yaw = 0
    ],
    output="screen",
)
    
    # Add RViz node (optional)
    rviz = Node(
       package='rviz2',
       executable='rviz2',
    )

    # Add ros_gz_bridge parameter bridge for joint states
    bridge_args = [
        '/joint_state1@std_msgs/msg/Float64@gz.msgs.Double',
        '/joint_state2@std_msgs/msg/Float64@gz.msgs.Double',
        '/joint_state3@std_msgs/msg/Float64@gz.msgs.Double',
        '/joint_state4@std_msgs/msg/Float64@gz.msgs.Double',
        '/joint_state5@std_msgs/msg/Float64@gz.msgs.Double',
        '/joint_state6@std_msgs/msg/Float64@gz.msgs.Double',
        '/joint_state7@std_msgs/msg/Float64@gz.msgs.Double',
        '/joint_state8@std_msgs/msg/Float64@gz.msgs.Double',
        '/joint_state9@std_msgs/msg/Float64@gz.msgs.Double',
        '/joint_state10@std_msgs/msg/Float64@gz.msgs.Double',
        '/joint_state11@std_msgs/msg/Float64@gz.msgs.Double',
        '/joint_state12@std_msgs/msg/Float64@gz.msgs.Double',
    ]

    parameter_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_parameter_bridge',
        arguments=bridge_args,
        output='screen',
    )

    # Launch!
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use sim time if true'),

        node_robot_state_publisher,
        gz_sim,
        spawn_entity,
        parameter_bridge_node,
       
    ])
