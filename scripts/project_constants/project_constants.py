import os


# ROS Environment URI's
RIDGEBACK_ROS_MASTER_URI = "http://cpr-r100-0101:11311"
PANDA_DROP_OBJECTS_JOINTS = [1.67, 0.66, -0.05, -1.56, -0.0, 2.16, 0.87]
PANDA_HOME_JOINTS = [-0.02, -0.78, -0.02, -1.83, -0.01, 1.01, 0.82]
PANDA_HOME_JOINTS_VISION = [-0.03, -1.76, -0.02, -2.86, -0.03, 2.05, 0.81]
PANDA_HOME_JOINTS_WORK_SCENARIO = [-1.68, -0.52, -0.34, -2.35, -0.18, 1.85, 0.46]
GO_TO_GRAB_POSITION = [0.05, -1.19, -0.15, -2.46, -0.06, 1.29, -0.72]

script_dir = os.path.dirname(os.path.realpath(__file__))
YOLO_CHECKPOINT = os.path.join(script_dir, "weights_and_secrets/model_weights/YOLOv8/kitchen_yolo.pt")

D405_REALSENSE_CAMERA_ID = "128422272486"
