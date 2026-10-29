#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, PoseStamped, Pose
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from tf2_msgs.msg import TFMessage                        
from nav2_simple_commander.robot_navigator import BasicNavigator
from rclpy.duration import Duration
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from transforms3d.euler import euler2quat
from time import sleep 
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Lock
import math
import numpy as np
from threading import Thread


class GolfBallPipeline(Node):

    def __init__(self):

        
        super().__init__('golf_ball_pipeline')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.last_goal_x = None
        self.last_goal_y = None
        callback_group_1 = MutuallyExclusiveCallbackGroup()
        callback_group_2 = MutuallyExclusiveCallbackGroup()
        self.is_navigating = False
        self.target_reached = False
        self.lock = Lock()
        self.robot_x=0.0
        self.robot_y=0.0
        self.visited_objects = []
       
        self.full_rotation_done = False
        self.robot_busy = False
        self.currently_navigating = False
        self.golf_bal_pose_sub = self.create_subscription(
            PoseArray, 'box_poses', self.golf_ball_pose_callback, 1,
            callback_group=callback_group_1)

        #self.timer = self.create_timer(5.0, self.go_to_home,
                          #             callback_group=callback_group_2)

        # ── TF: robot publishes on /tb1/tf, not the default /tf ──────────────
        # TransformListener alone won't see /tb1/tf, so we feed it manually.
        self.tf_buffer   = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self,
                                             spin_thread=False)

        self._tf_sub = self.create_subscription(
            TFMessage, 'tf',
            lambda msg: [
                self.tf_buffer.set_transform(t, 'default_authority')
                for t in msg.transforms
            ],
            100,
        )
        self._tf_static_sub = self.create_subscription(
            TFMessage, 'tf_static',
            lambda msg: [
                self.tf_buffer.set_transform_static(t, 'default_authority')
                for t in msg.transforms
            ],
            100,
        )
        self.get_logger().info("TF remapped: /tb1/tf → Buffer")
        

        self.count = 0
        self.rotate_inable = True
        self.rotation_speed = 0.15
        self.rotation_duration = 5.0
        self.got_goal = False
        self.map_frame = 'map'
        self.robot_base_frame = 'base_footprint'
        
        self.navigator = BasicNavigator()
        self.spin_angle = 20 * np.pi / 180.0
        self.current_golf_ball_poses = None

        self.received_poses = None
        self.last_goal = None

        self.at_home = False
        self.going_home = False
        self.cancel_goal = False
        self.scanning = False
        self.object_detected = False

        self.scan_timer = self.create_timer(
             0.1,
            self.scan_callback
        )   

        self.lock = Lock()
    def rotate_in_place(self, speed=1.0, direction=-1):

        twist = Twist()

        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = speed * direction

        self.cmd_vel_pub.publish(twist)
    def stop_robot(self):

        self.cmd_vel_pub.publish(Twist())
    def scan_callback(self):

        if not self.scanning:
            return

        if self.currently_navigating:
            return

        t = self.get_robot_pose()

        if t is None:
            return

       

       

       

            

        self.stop_robot()

        self.scanning = False
        

        # continue rotating
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.5

        self.cmd_vel_pub.publish(twist)
    def get_robot_pose(self):

        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_base_frame,
                rclpy.time.Time()
            )

            return t

        except TransformException as ex:

            self.get_logger().warn(
                f'Could not transform {self.map_frame} to '
                f'{self.robot_base_frame}: {ex}'
            )

            return None
        

    def go_to_home(self):

        if self.is_navigating:
            return

        t = self.get_robot_pose()
        if t is None:
            return

        distance = np.sqrt(
            t.transform.translation.x ** 2 +
            t.transform.translation.y ** 2
        )

        if distance < 0.5:
            self.get_logger().info("Already at home.")
            return

        self.get_logger().info("Going to home position.")

        home_pose = Pose()
        home_pose.position.x = 0.0
        home_pose.position.y = 0.0
        home_pose.orientation.w = 1.0

        self.is_navigating = True

        try:
            self.send_goal(
                home_pose,
                keep_orientation=False,
                add_offset=False
            )
        finally:
           self.is_navigating = False
    def get_closest_goal(self, poses):
        t = self.get_robot_pose()
        if t is None:
            return
        distances = [
            np.sqrt((t.transform.translation.x - pose.position.x) ** 2 +
                    (t.transform.translation.y - pose.position.y) ** 2)
            for pose in poses
        ]
        return distances

    def golf_ball_pose_callback(self, msg):

                if self.robot_busy:
                    return

                if len(msg.poses) == 0:
                    return
                try:
                    t = self.get_robot_pose()
                    if t is None:
                            return

                    self.robot_x = t.transform.translation.x
                    self.robot_y = t.transform.translation.y

                except Exception as e:
                    self.get_logger().warn(f"TF failed: {e}")
                    return

                valid_objects = []
                distances = []
                for pose in msg.poses:

                    if self.is_object_visited(pose):
                        continue

                    robot_dist = math.sqrt(
                        (pose.position.x - self.robot_x)**2 +
                        (pose.position.y - self.robot_y)**2
                    )

                    valid_objects.append(pose)
                    distances.append(robot_dist)


                        # IMPORTANT FIX
               
                
                if len(valid_objects) == 0:
                    self.get_logger().info("No new valid objects found")
                    return
                # stop scanning immediately
                self.scanning = False
                
                self.full_rotation_done = False

                self.stop_robot()
                nearest_idx = np.argmin(distances)
                target_pose = valid_objects[nearest_idx]
                

                self.robot_busy = True

                Thread(
                    target=self.send_goal,
                    args=(target_pose,),
                    daemon=True
                    ).start()

               
   
    def is_object_visited(self, pose):

        for vx, vy in self.visited_objects:

            dist = math.sqrt(
                (pose.position.x - vx) ** 2 +
                (pose.position.y - vy) ** 2
            )

            if dist < 1.0:
                return True

        return False                
    def send_goal(self, pose, keep_orientation=False, add_offset=True):
            if self.currently_navigating:
                return
            
            twist_msg = Twist()
            self.cmd_vel_pub.publish(twist_msg)

            goal_pose = PoseStamped()

            goal_pose.header.frame_id = self.map_frame
            goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()

            
            t = self.get_robot_pose()
            if t is None:
                return

            robot_x = t.transform.translation.x
            robot_y = t.transform.translation.y

            
            theta = np.arctan2(
                pose.position.y - robot_y,
                pose.position.x - robot_x
            )

            if add_offset:
                stop_distance = 0.2

                offset_x = stop_distance * np.cos(theta)
                offset_y = stop_distance * np.sin(theta)
            else:
                offset_x = 0.0
                offset_y = 0.0

            if keep_orientation:
                w = pose.orientation.w
                z = pose.orientation.z
            else:
                w, x, y, z = euler2quat(0, 0, theta)

        
            goal_pose.pose.position.x = pose.position.x - offset_x
            goal_pose.pose.position.y = pose.position.y - offset_y
            goal_pose.pose.position.z = 0.0

            goal_pose.pose.orientation.w = w
            goal_pose.pose.orientation.z = z

            self.get_logger().info(
                f"Sending goal -> "
                f"x={goal_pose.pose.position.x:.2f}, "
                f"y={goal_pose.pose.position.y:.2f}"
            )

            try:

                self.currently_navigating = True

                self.stop_robot()
                sleep(0.5)

                self.navigator.goToPose(goal_pose)

                while not self.navigator.isTaskComplete():
                    sleep(0.1)

                result = self.navigator.getResult()

                if result == TaskResult.SUCCEEDED:

                    self.get_logger().info("Goal succeeded!")
                    self.get_logger().info("Resuming scan...")
                   
                    self.stop_robot()
                    sleep(1.0)
                    self.visited_objects.append((
                        pose.position.x,
                        pose.position.y
                    ))
                    sleep(3.0)
                   
                    self.full_rotation_done = False
                    self.scanning = True

                elif result == TaskResult.CANCELED:

                    self.get_logger().warn("Goal canceled!")
                    self.scanning = True

                elif result == TaskResult.FAILED:

                    self.get_logger().error("Goal failed!")
                    self.scanning = True

            finally:

                self.currently_navigating = False
                self.robot_busy = False

                
      


def main(args=None):
    rclpy.init()
    node = GolfBallPipeline()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()