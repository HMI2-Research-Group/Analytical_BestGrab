import rospy
import moveit_commander
import tf
import moveit_msgs.msg
from math import pi
import actionlib
import franka_gripper.msg
import franka_msgs.msg
from time import sleep
from multimethod import multimethod
import geometry_msgs.msg
from sensor_msgs.msg import JointState


class FrankaOperator:
    def __init__(self) -> None:
        """
        This class is used to control the Franka Emika Panda robot.
        Be sure to activate ROS node and initialize moveit_commander before using this class.
        """
        self.robot = moveit_commander.RobotCommander()
        group_name = "panda_manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        self.move_group.set_planning_time(10.0)
        print(f"============ End effector link: {self.move_group.get_end_effector_link()}")
        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20
        )
        self.panda_recovery = rospy.Publisher(
            "/franka_control/error_recovery/goal", franka_msgs.msg.ErrorRecoveryActionGoal, queue_size=20
        )
        self.movegroup_feedback_subscriber = rospy.Subscriber(
            "/move_group/feedback", moveit_msgs.msg.MoveGroupActionFeedback, self._post_execution_status
        )

    def display_franka_trajectory(self, trajectory: moveit_msgs.msg.RobotTrajectory) -> None:
        # ToDo: FeatureRequest: Add a function to display the trajectory of the robot constantly
        pass

    def close_gripper(self) -> None:
        client = actionlib.SimpleActionClient("/franka_gripper/grasp", franka_gripper.msg.GraspAction)
        client.wait_for_server()
        goal = franka_gripper.msg.GraspGoal()
        goal.epsilon.inner = 10.0
        goal.epsilon.outer = 10.0
        goal.speed = 0.08
        # goal.force = 50.0
        client.send_goal(goal)
        client.wait_for_result()
        # Wait for message
        left_finger_width = rospy.wait_for_message("/franka_gripper/joint_states", JointState).position[0]
        if left_finger_width < 0.0005:
            rospy.loginfo("Gripper Failed to close, pausing further execution")
            self.open_gripper()
            input("Fix franka errors and press Enter to continue...")
            self.close_gripper()

    def partiall_close_gripper(self) -> None:
        client = actionlib.SimpleActionClient("/franka_gripper/move", franka_gripper.msg.MoveAction)
        client.wait_for_server()
        goal = franka_gripper.msg.MoveGoal()
        goal.width = 0.06  # TODO: Find the best possible value for this
        goal.speed = 0.08
        client.send_goal(goal)
        client.wait_for_result()

    def open_gripper(self) -> None:
        client = actionlib.SimpleActionClient("/franka_gripper/move", franka_gripper.msg.MoveAction)
        client.wait_for_server()
        goal = franka_gripper.msg.MoveGoal()
        goal.width = 0.08
        goal.speed = 0.08
        client.send_goal(goal)
        client.wait_for_result()

    def panda_recover(self) -> None:
        for _ in range(3):
            # Write the text in red
            print("\033[91m" + "Sending Recovery Message" + "\033[0m")
            self.panda_recovery.publish(franka_msgs.msg.ErrorRecoveryActionGoal())

    def _post_execution_status(self, execution_status) -> None:
        if execution_status.status.text == "CONTROL_FAILED":
            print("\033[91m" + f"Panda Hit object, Please fix the errors" + "\033[0m")

    @multimethod
    def move_to_pose(self, target_pose: geometry_msgs.msg.PoseStamped) -> bool:
        waypoints = []
        waypoints.append(target_pose.pose)
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(display_trajectory)
        # Move Group Execute Motion Plan of the robot
        self.move_group.set_pose_target(target_pose)
        # my_pub = rospy.Publisher("/robot_pickuppose", geometry_msgs.msg.PoseStamped, queue_size=10)
        # my_pub.publish(target_pose)
        execution_status = self.move_group.go(wait=True)
        if not execution_status:
            print("Execution Failed, pausing further execution")
            input("Fix franka errors and press Enter to continue...")
            return self.move_to_pose(target_pose)
        return execution_status

    @multimethod
    def move_to_pose(self, target_pose: list) -> bool:
        self.move_group.set_joint_value_target(target_pose)
        plan = self.move_group.plan()
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan[1])
        self.display_trajectory_publisher.publish(display_trajectory)
        # Move Group Execute Motion Plan of the robot
        execution_status = self.move_group.go(wait=True)
        if not execution_status:
            print("Execution Failed, pausing further execution")
            input("Fix franka errors and press Enter to continue...")
            return self.move_to_pose(target_pose)
        return execution_status

    def lift_object(self, height: float) -> bool:
        """
        Lift the object by the given height.
        """
        current_pose = self.move_group.get_current_pose()
        # current_pose = transform_pose_stamped("panda_EE", current_pose)
        current_pose.pose.position.z += height
        return self.move_to_pose(current_pose)
