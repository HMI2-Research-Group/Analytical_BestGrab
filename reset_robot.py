import sys

sys.path.append("/usr/lib/python3/dist-packages")
import rospy
from scripts.panda_moveit_library import FrankaOperator
from scripts.project_constants import PANDA_HOME_JOINTS_VISION

if __name__ == "__main__":
    rospy.init_node("reset_robot", anonymous=True)
    my_franka_robot = FrankaOperator()
    my_franka_robot.move_to_pose(PANDA_HOME_JOINTS_VISION)
    my_franka_robot.rotate_gripper_to_default()
    my_franka_robot.open_gripper()
