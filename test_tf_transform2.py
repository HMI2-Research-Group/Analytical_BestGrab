import tf
import rospy
from math import pi


def main():
    # publish a tf frame connecting two frames
    rospy.init_node("make_robot_connections")
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(100.0)
    while 1:
        br.sendTransform(
            (0.07, 0.0, -0.103),
            tf.transformations.quaternion_from_euler(0, 0, pi / 2.0),
            rospy.Time.now(),
            "camera_color_optical_frame",
            "panda_hand_tcp",
        )
        rate.sleep()


if __name__ == "__main__":
    main()
