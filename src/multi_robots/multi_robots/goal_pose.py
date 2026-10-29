import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped


class GoalPublisherNode(Node):

    def __init__(self):

        super().__init__('goal_publisher_node')

        self.object_sub = self.create_subscription(
            Pose,
            '/object_pose',
            self.object_pose_callback,
            10
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        self.get_logger().info("Goal Publisher Node Started")

    def object_pose_callback(self, msg):

        self.get_logger().info(
            f"Received object pose: x={msg.position.x}, y={msg.position.y}"
        )

        goal = PoseStamped()

       
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()

       
        goal.pose.position.x = msg.position.x
        goal.pose.position.y = msg.position.y
        goal.pose.position.z = 0.0

        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = 0.0
        goal.pose.orientation.w = 1.0

      
        self.goal_pub.publish(goal)

        self.get_logger().info(
            f"Goal published to /goal_pose: ({goal.pose.position.x}, {goal.pose.position.y})"
        )


def main(args=None):

    rclpy.init(args=args)

    node = GoalPublisherNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
