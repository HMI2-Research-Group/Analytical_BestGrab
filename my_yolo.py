import torch
from ultralytics import YOLO
import numpy as np
import torch
import cv2
from time import sleep
import pyrealsense2
import sys

sys.path.append("/usr/lib/python3/dist-packages")  # For rospkg dependency
from scripts.project_constants import YOLO_CHECKPOINT
from scripts.panda_moveit_library import FrankaOperator
from scripts.project_constants import PANDA_HOME_JOINTS_VISION, D405_REALSENSE_CAMERA_ID
import rospy
from tf2_geometry_msgs import PointStamped
import tf2_ros
from collections import defaultdict


def transform_point_stamped(x, y, z, tfBuffer, target_frame="panda_link0"):
    # make a listener

    # wait for the transform to be available
    point_stamped_msg = PointStamped()
    point_stamped_msg.header.frame_id = "camera_color_optical_frame"
    point_stamped_msg.header.stamp = rospy.Time(0)
    point_stamped_msg.point.x = x
    point_stamped_msg.point.y = y
    point_stamped_msg.point.z = z
    while not rospy.is_shutdown():
        try:
            # now = rospy.Time.now()
            # listener.waitForTransform(target_frame, point_stamped_msg.header.frame_id, now, rospy.Duration(4.0))
            transformed_point = tfBuffer.transform(point_stamped_msg, target_frame)
            # Round off the x, y, z values to 2 decimal places
            transformed_point.point.x = round(transformed_point.point.x, 2)
            transformed_point.point.y = round(transformed_point.point.y, 2)
            transformed_point.point.z = round(transformed_point.point.z, 2)
            return transformed_point.point.x, transformed_point.point.y, transformed_point.point.z
        except Exception as e:
            # rospy.logerr("Failed to transform point: %s", e)
            sleep(0.1)
            continue


def main():
    rospy.init_node("test_yolo", anonymous=True)
    tfBuffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tfBuffer)
    # my_franka_robot = FrankaOperator()
    # my_franka_robot.move_to_pose(PANDA_HOME_JOINTS_VISION)
    model = YOLO(YOLO_CHECKPOINT)  # load a pretrained model (recommended for training)
    model.to("cuda")  # optionally change device
    # Intialize the pipeline
    pipeline = pyrealsense2.pipeline()
    config = pyrealsense2.config()
    config.enable_device(D405_REALSENSE_CAMERA_ID)
    config.enable_stream(pyrealsense2.stream.depth, 640, 480, pyrealsense2.format.z16, 30)
    config.enable_stream(pyrealsense2.stream.color, 640, 480, pyrealsense2.format.bgr8, 30)
    align = pyrealsense2.align(pyrealsense2.stream.color)
    pipeline.start(config)
    # write a one line function to round the tensor to nearest integer
    round_tensor = lambda x: round(float(x.data))
    while 1:
        frames = pipeline.wait_for_frames()
        # align the depth and color frames
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame_np = np.asanyarray(depth_frame.get_data())
        # cv2 resize image to 640, 640
        color_image = cv2.resize(color_image, (640, 640))
        # convert color image numpy image to cuda
        image_tensor = torch.from_numpy(color_image)
        image_tensor = image_tensor.float() / 255.0
        # convert the tensor 480, 640, 3 to 1, 3, 640, 640
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        # change BGR tensor to RGB tensor
        image_tensor = image_tensor[:, [2, 1, 0], :, :]
        # testing  with random image
        # image_tensor = torch.rand(1, 3, 640, 640, dtype=torch.float32)
        # normalize the image
        image_tensor.to("cuda")
        results = model(image_tensor)
        for result in results:
            bb_boxes = result.boxes.xyxy
            # draw the bounding box
            for i, bb_box in enumerate(bb_boxes):
                # Plot only if confidence is greater than 0.8
                if float(result.boxes.conf[i].data) > 0.5:
                    cv2.rectangle(
                        color_image,
                        (round_tensor(bb_box[0]), round_tensor(bb_box[1])),
                        (round_tensor(bb_box[2]), round_tensor(bb_box[3])),
                        (0, 0, 255),
                        2,
                    )
                    # Place the text on top of the bounding box
                    class_name = result.names[int(result.boxes.cls[i].data)]
                    cv2.putText(
                        color_image,
                        result.names[int(result.boxes.cls[i].data)],
                        (round_tensor(bb_box[0]), round_tensor(bb_box[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if class_name == "Chilli":
                        break
        # Method 1: Just find the closest point depth from camera (Doesn't work well)
        # # mask depth except the bounding box
        # mask = np.zeros_like(depth_frame)
        # mask[
        #     round_tensor(bb_box[1]) : round_tensor(bb_box[3]), round_tensor(bb_box[0]) : round_tensor(bb_box[2])
        # ] = depth_frame[
        #     round_tensor(bb_box[1]) : round_tensor(bb_box[3]), round_tensor(bb_box[0]) : round_tensor(bb_box[2])
        # ]
        # # Find the pixel coordinates of the lowest value other than zero in the mask

        # highest_pixel = np.where(mask == np.min(mask[np.nonzero(mask)]))
        # cv2.circle(color_image, (highest_pixel[1][0], highest_pixel[0][0]), 10, (0, 255, 0), -1)
        # Method 2: Find the 3D points of entire bounding box and find the closest point
        # Get the 3D points of the bounding box
        # Get the 3D points of the bounding box
        bb_box_3d = defaultdict(list)
        for i in range(
            round_tensor(bb_box[0] * (480.0 / 640.0)),
            round_tensor(bb_box[2] * (480.0 / 640.0)),
        ):
            for j in range(round_tensor(bb_box[1]), round_tensor(bb_box[3])):
                pixel_depth = depth_frame.get_distance(j, i)
                three_d_point = pyrealsense2.rs2_deproject_pixel_to_point(depth_intrin, [i, j], pixel_depth)
                bb_box_3d[i, j] = transform_point_stamped(*three_d_point, tfBuffer)
        # find the point with highest z value
        highest_z = 0
        highest_z_point = None
        for key, value in bb_box_3d.items():
            if value[2] > highest_z:
                highest_z = value[2]
                highest_z_point = key
        # Find all the points which are +- 0.1 of the highest z point
        # highest_z_points = []
        # threshold = 0.1
        # for key, value in bb_box_3d.items():
        #     if value[2] > highest_z - threshold and value[2] < highest_z + threshold:
        #         highest_z_points.append(key)
        # for highest_z_point in highest_z_points:
        # cv2.circle(color_image, (int(highest_z_point[0] * 640 / 480), highest_z_point[1]), 5, (0, 255, 0), -1)
        # Find highest z points along each column
        # for j in range(round_tensor(bb_box[1]), round_tensor(bb_box[3])):
        #     highest_z = 0
        #     highest_z_point = None
        #     for i in range(
        #         round_tensor(bb_box[0] * (480.0 / 640.0)),
        #         round_tensor(bb_box[2] * (480.0 / 640.0)),
        #     ):
        #         pixel_depth = depth_frame.get_distance(j, i)
        #         three_d_point = pyrealsense2.rs2_deproject_pixel_to_point(depth_intrin, [i, j], pixel_depth)
        #         bb_box_3d[i, j] = transform_point_stamped(*three_d_point)
        cv2.circle(color_image, (int(highest_z_point[0] * 640 / 480), highest_z_point[1]), 5, (0, 255, 0), -1)
        cv2.imshow("image", color_image)
        # sleep for 1 ms
        # sleep(0.01)
        # wait for q key to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
