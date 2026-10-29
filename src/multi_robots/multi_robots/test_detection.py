#!/usr/bin/env python3
import rclpy
import tf2_ros
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState

rclpy.init()
n = Node("ik_test")
js = None

def cb(m):
    global js
    js = m

n.create_subscription(JointState, "/joint_states", cb, 10)
buf = tf2_ros.Buffer()
tf2_ros.TransformListener(buf, n)
ik = n.create_client(GetPositionIK, "/compute_ik")
ik.wait_for_service()
while js is None:
    rclpy.spin_once(n, timeout_sec=0.1)
for _ in range(30):               # let TF fill
    rclpy.spin_once(n, timeout_sec=0.1)

state = RobotState()
state.joint_state = js

for frame in ("base_link", "link1", "base_footprint", "odom", "map", "world"):
    try:
        t = buf.lookup_transform(frame, "end_effector_link", rclpy.time.Time())
    except Exception as e:
        print(f"{frame:16s} TF unavailable")
        continue
    tr, ro = t.transform.translation, t.transform.rotation
    req = GetPositionIK.Request()
    q = req.ik_request
    q.group_name = "arm"
    q.ik_link_name = "end_effector_link"
    q.robot_state = state
    q.pose_stamped.header.frame_id = frame
    p = q.pose_stamped.pose
    p.position.x, p.position.y, p.position.z = tr.x, tr.y, tr.z
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = ro.x, ro.y, ro.z, ro.w
    q.timeout.sec = 1
    f = ik.call_async(req)
    rclpy.spin_until_future_complete(n, f)
    print(f"{frame:16s} pos=({tr.x:.3f},{tr.y:.3f},{tr.z:.3f}) val={f.result().error_code.val}")
rclpy.shutdown()