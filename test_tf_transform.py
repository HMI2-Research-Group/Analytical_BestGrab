import rospy
import tf_conversions
import tf2_ros
import geometry_msgs.msg
from math import pi


def handle_turtle_pose():
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.frame_id = "panda_hand_tcp"
    t.child_frame_id = "panda_link0"
    t.transform.translation.x = 0.07
    t.transform.translation.y = 0.0
    t.transform.translation.z = -0.103
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, pi / 2.0)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]

    while 1:
        t.header.stamp = rospy.Time.now()
        br.sendTransform(t)


if __name__ == "__main__":
    rospy.init_node("tf2_turtle_broadcaster", anonymous=True)
    handle_turtle_pose()
