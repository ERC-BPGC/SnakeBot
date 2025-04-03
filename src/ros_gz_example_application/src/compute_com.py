#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf2_ros import TransformListener, Buffer
import math

class LinkMarkerPublisher(Node):
    def __init__(self):
        super().__init__('link_marker_publisher')

        # Publisher for MarkerArray
        self.marker_pub = self.create_publisher(MarkerArray, '/link_markers', 10)

        # TF2 listener to get link positions
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to periodically publish markers
        self.timer = self.create_timer(1.0, self.publish_markers)

        # List of link names (extracted from your XACRO file)
        self.link_names = [
            "base_link",
            "cross1",
            "segment1",
            "segment2a",
            "cross2",
            "segment2b",
            "segment3a",
            "cross3",
            "segment3b",
            # Add more links as needed
        ]

    def publish_markers(self):
        marker_array = MarkerArray()

        for i, link_name in enumerate(self.link_names):
            try:
                # Get the transform of the link in the base_footprint frame
                transform = self.tf_buffer.lookup_transform("base_footprint", link_name, rclpy.time.Time())

                # Create a marker for the link
                marker = Marker()
                marker.header.frame_id = "base_footprint"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "links"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                # Set the marker position
                marker.pose.position.x = transform.transform.translation.x
                marker.pose.position.y = transform.transform.translation.y
                marker.pose.position.z = transform.transform.translation.z

                # Set the marker orientation
                marker.pose.orientation = transform.transform.rotation

                # Set the marker scale
                marker.scale.x = 0.05  # Sphere diameter
                marker.scale.y = 0.05
                marker.scale.z = 0.05

                # Set the marker color (blue)
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0

                # Add the marker to the array
                marker_array.markers.append(marker)

            except Exception as e:
                self.get_logger().warn(f"Failed to get transform for {link_name}: {str(e)}")

        # Publish the MarkerArray
        self.marker_pub.publish(marker_array)
        self.get_logger().info("Published markers for all links")

def main(args=None):
    rclpy.init(args=args)
    node = LinkMarkerPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()