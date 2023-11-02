import sys

sys.path.append("/usr/lib/python3/dist-packages")
import rospy
import moveit_commander


def read_franka_joint_values():
    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    current_joint_values = [round(x, 2) for x in move_group.get_current_joint_values()]
    print(f"Current joint values: {current_joint_values}")


if __name__ == "__main__":
    rospy.init_node("go_to_pose", anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)
    read_franka_joint_values()
