from typing import Any
import torch
from ultralytics import YOLO
import numpy as np
import torch
import cv2
from time import sleep
import pyrealsense2
import sys
import time
import math
from matplotlib import pyplot as plt

sys.path.append("/usr/lib/python3/dist-packages")
from scripts.project_constants import YOLO_CHECKPOINT
from scripts.panda_moveit_library import FrankaOperator
from scripts.project_constants import PANDA_HOME_JOINTS_VISION, D405_REALSENSE_CAMERA_ID
import rospy
from tf2_geometry_msgs import PointStamped, PoseStamped
import tf2_ros
from collections import defaultdict


def transform_point_to_posestamped(x, y, z, tfBuffer, target_frame="panda_EE"):
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
            # transformed_point = tfBuffer.transform(transformed_point, "panda_EE")
            my_point_stamped = PoseStamped()
            my_point_stamped.pose.position.x = transformed_point.point.x
            my_point_stamped.pose.position.y = transformed_point.point.y
            my_point_stamped.pose.position.z = transformed_point.point.z
            my_point_stamped.header.frame_id = "panda_EE"
            return my_point_stamped
        except Exception as e:
            # rospy.logerr("Failed to transform point: %s", e)
            print(e)
            sleep(0.1)
            continue


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
            round_off_decimals = 3
            transformed_point.point.x = round(transformed_point.point.x, round_off_decimals)
            transformed_point.point.y = round(transformed_point.point.y, round_off_decimals)
            transformed_point.point.z = round(transformed_point.point.z, round_off_decimals)
            return transformed_point.point.x, transformed_point.point.y, transformed_point.point.z
        except Exception as e:
            # rospy.logerr("Failed to transform point: %s", e)
            print(e)
            sleep(0.1)
            continue


def perpendicular_distance(point: list, line_points: list) -> float:
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]

    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1

    x0, y0 = point
    distance = np.abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)
    return distance


def find_gripper_angle(all_points, best_point):
    all_x = [best_point[0]]
    all_y = [best_point[1]]
    for point in all_points:
        all_x.append(point[0])
        all_y.append(point[1])
    x = np.array(all_x)
    y = np.array(all_y)
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
    a, b, c = coefficients
    # Plot the quadratic curve
    x = np.linspace(int(min(all_x) - 1), 0.1, int(max(all_x) + 1))
    y = a * x**2 + b * x + c
    plt.clf()
    plt.plot(x, y)
    gripper_angle = math.atan(-1 / (2 * a * best_point[0] + b))
    return gripper_angle


round_tensor = lambda x: np.rint(x).astype(int)


class YOLODetector:
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)
        self.model = YOLO(YOLO_CHECKPOINT)  # load a pretrained model (recommended for training)
        self.model.to("cuda")
        self.pipeline = pyrealsense2.pipeline()
        config = pyrealsense2.config()
        config.enable_device(D405_REALSENSE_CAMERA_ID)
        config.enable_stream(pyrealsense2.stream.depth, 640, 480, pyrealsense2.format.z16, 30)
        config.enable_stream(pyrealsense2.stream.color, 640, 480, pyrealsense2.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = pyrealsense2.align(pyrealsense2.stream.color)
        self.interested_object = "Garlic"
        self.my_franka = FrankaOperator()

    def detect(self):
        frames = self.pipeline.wait_for_frames()
        # align the depth and color frames
        frames = self.align.process(frames)
        self.color_frame = frames.get_color_frame()
        self.depth_frame = frames.get_depth_frame()
        self.depth_intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.color_image_resized = cv2.resize(self.color_image, (640, 640))
        # convert color image numpy image to cuda
        image_tensor = torch.from_numpy(self.color_image_resized)
        image_tensor = image_tensor.float() / 255.0
        # convert the tensor 480, 640, 3 to 1, 3, 640, 640
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        # change BGR tensor to RGB tensor
        image_tensor = image_tensor[:, [2, 1, 0], :, :]
        # testing  with random image
        # image_tensor = torch.rand(1, 3, 640, 640, dtype=torch.float32)
        # normalize the image
        image_tensor.to("cuda")
        with torch.no_grad():
            results = self.model(image_tensor)
            for result in results:
                bb_boxes = result.boxes.xyxy
                # draw the bounding box
                for i, bb_box in enumerate(bb_boxes):
                    # Plot only if confidence is greater than 0.8
                    if float(result.boxes.conf[i].data) > 0.5:
                        bb_box = bb_box.cpu().detach().numpy()
                        bb_box[1] *= 480.0 / 640.0
                        bb_box[3] *= 480.0 / 640.0
                        cv2.rectangle(
                            self.color_image,
                            (round_tensor(bb_box[0]), round_tensor(bb_box[1])),
                            (round_tensor(bb_box[2]), round_tensor(bb_box[3])),
                            (0, 0, 255),
                            2,
                        )
                        # Place the text on top of the bounding box
                        class_name = result.names[int(result.boxes.cls[i].data)]
                        cv2.putText(
                            self.color_image,
                            result.names[int(result.boxes.cls[i].data)],
                            (round_tensor(bb_box[0]), round_tensor(bb_box[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        if class_name == self.interested_object:
                            return bb_box
        return None

    def get_pixel_and_angle(self, bb_box):
        bb_box_3d = defaultdict(list)
        all_pixels = []
        highest_heights = []
        for i in range(
            round_tensor(bb_box[0]),
            round_tensor(bb_box[2]),
        ):
            highest_z = 0
            highest_z_point = None
            for j in range(round_tensor(bb_box[1]), round_tensor(bb_box[3])):
                try:
                    pixel_depth = self.depth_frame.get_distance(i, j)
                except:
                    continue
                if pixel_depth == 0.0:
                    continue
                three_d_point = pyrealsense2.rs2_deproject_pixel_to_point(self.depth_intrin, [i, j], pixel_depth)
                three_d_point = transform_point_stamped(*three_d_point, self.tfBuffer)
                bb_box_3d[i, j] = three_d_point
                if three_d_point[2] > highest_z:
                    highest_z = three_d_point[2]
                    highest_z_point = [i, j]
            if highest_z_point is not None:
                all_pixels.append(highest_z_point)
                highest_heights.append(highest_z)
        # For all the points in all_pixels, make a small circle
        if len(highest_heights) == 0:
            return None, None
        max_height = max(highest_heights)
        thresh = 0.001
        max_dist_from_center = 0.0
        best_pixel = None
        line_points = [
            [float((bb_box[0] + bb_box[2] / 2)), float(bb_box[1])],
            [float((bb_box[0] + bb_box[2] / 2)), float(bb_box[3])],
        ]
        interesting_pixels = []
        for pixel, curr_height in zip(all_pixels, highest_heights):
            if curr_height >= max_height - thresh:
                interesting_pixels.append(pixel)
                dist_from_center = perpendicular_distance(pixel, line_points)
                if dist_from_center > max_dist_from_center:
                    max_dist_from_center = dist_from_center
                    best_pixel = pixel
        if best_pixel is not None:
            # In the interesting pixels, find the four closest pixels to the best pixel
            closest_pixels = []
            for pixel in interesting_pixels:
                dist = np.linalg.norm(np.array([pixel[0], pixel[1]]) - np.array([best_pixel[0], best_pixel[1]]))
                closest_pixels.append([pixel, dist])
            closest_pixels = [pixel for pixel in closest_pixels if pixel[1] > 0.0]
            closest_pixels.sort(key=lambda x: x[1])
            closest_pixels = closest_pixels[:4]
            # for each_pixel in closest_pixels:
            #     cv2.circle(color_image, (int(each_pixel[0][0] * 640 / 480), each_pixel[0][1]), 5, (0, 255, 0), -1)
            cv2.circle(self.color_image, (best_pixel[0], best_pixel[1]), 5, (0, 255, 0), -1)
            my_angle = find_gripper_angle(
                [bb_box_3d[pixel[0][0], pixel[0][1]] for pixel in closest_pixels],
                bb_box_3d[best_pixel[0], best_pixel[1]],
            )
            return best_pixel, my_angle

    def pick_object(self, best_pixel, my_angle):
        tf_stamped = transform_point_to_posestamped(
            *pyrealsense2.rs2_deproject_pixel_to_point(
                self.depth_intrin, best_pixel, self.depth_frame.get_distance(*best_pixel)
            ),
            self.tfBuffer,
        )
        # tf_stamped.pose.position.z -= 0.05
        self.my_franka.move_to_pose(tf_stamped)
        # Now rotate the gripper
        self.my_franka.rotate_gripper(my_angle)
        self.my_franka.close_gripper()
        # Now move the gripper down
        # tf_stamped.pose.position.z += 0.06
        # self.my_franka.move_to_pose(tf_stamped)
        self.my_franka.move_to_pose(PANDA_HOME_JOINTS_VISION)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        start_time = time.time()
        while 1:
            my_bbox = self.detect()
            if my_bbox is not None:
                best_pixel, my_angle = self.get_pixel_and_angle(my_bbox)
                if best_pixel is not None:
                    if time.time() - start_time > 5 and my_angle > 0.0:
                        cv2.imshow("YOLO", self.color_image)
                        cv2.waitKey(1)
                        self.pick_object(best_pixel, my_angle)
                        break
            cv2.imshow("YOLO", self.color_image)
            cv2.waitKey(1)
            sleep(0.01)


if __name__ == "__main__":
    rospy.init_node("yolo_detector", anonymous=True)
    YOLODetector()()
