import os


# ROS Environment URI's
RIDGEBACK_ROS_MASTER_URI = "http://cpr-r100-0101:11311"
PANDA_DROP_OBJECTS_JOINTS = [1.67, 0.66, -0.05, -1.56, -0.0, 2.16, 0.87]
PANDA_HOME_JOINTS = [-0.02, -0.78, -0.02, -1.83, -0.01, 1.01, 0.82]
PANDA_HOME_JOINTS_VISION = [-0.03, -1.76, -0.02, -2.86, -0.03, 2.05, 0.81]
PANDA_HOME_JOINTS_WORK_SCENARIO = [-1.68, -0.52, -0.34, -2.35, -0.18, 1.85, 0.46]
GO_TO_GRAB_POSITION = [0.05, -1.19, -0.15, -2.46, -0.06, 1.29, -0.72]
# Poses
# my_point = geometry_msgs.msg.PoseStamped()
# quaternion = tf.transformations.quaternion_from_euler(0, -pi / 2.5, 0)
# my_point.pose.orientation.x = quaternion[0]
# my_point.pose.orientation.y = quaternion[1]
# my_point.pose.orientation.z = quaternion[2]
# my_point.pose.orientation.w = quaternion[3]
# OBJECT_GRAB_ORIENTATION = my_point.pose.orientation
# ROS TF Names
CAMERA_TF_FRAME = "camera_color_optical_frame"
# ROS Topic names
SPEECH_TO_TEXT_ROS_TOPIC = "speech_to_text"
SPEECH_INTENT_ROS_TOPIC = "speech_intent"
# Get secrets from Google Cloud, for that you need to use https://github.com/astrada/google-drive-ocamlfuse
# But first lets get the current folder of this script

script_dir = os.path.dirname(os.path.realpath(__file__))
# Now the secrets are in weights folder
GOOGLE_APPLICATION_CREDENTIALS = os.path.join(
    script_dir, "weights_and_secrets/Secrets and Model Weights/Speech To Text API Key", "key.json"
)
OBJ_SHAPE_LOCATION_CSV_PATH = os.path.join(script_dir, "obj_shape_location.csv")

# Realsense Camera IDs
D455_REALSENSE_CAMERA_ID = "138322250591"
D405_REALSENSE_CAMERA_ID = "128422272486"

# Language Model Checkpoints
BERT_CHECKPOINT = os.path.join(script_dir, "weights_and_secrets/Secrets and Model Weights/BERT")
GPT_CHECKPOINT = os.path.join(
    script_dir, "weights_and_secrets/Secrets and Model Weights/EleutherAI/gpt-neo-1.3B-finetuned-collaborative-cooking"
)
YOLO_CHECKPOINT = os.path.join(script_dir, "weights_and_secrets/Secrets and Model Weights/YOLOv8/kitchen_yolo.pt")

# Plux Device Constants
PLUX_DEVICE_MAC_ADDRESS = "00:07:80:0F:30:ED"
PLUX_DEVICE_CHANNELS = 0x0F  # because our's is a 4 channel device
PLUX_SAMPLING_FREQUENCY = 1000
PLUX_ROS_TOPIC = "plux_data"

# Muse Device Constants
MUSE_DEVICE_MAC_ADDRESS = "00:55:DA:B9:77:5B"
MUSE_ROS_TOPIC = "muse_data"
