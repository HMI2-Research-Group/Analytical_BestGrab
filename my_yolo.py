import sys
from scripts.project_constants import YOLO_CHECKPOINT

# from scripts.panda_moveit_library import FrankaOperator
from scripts.project_constants import PANDA_HOME_JOINTS_VISION, D405_REALSENSE_CAMERA_ID
import torch
from ultralytics import YOLO
import sys
import numpy as np
import torch
import cv2
from time import sleep
import pyrealsense2

# import rospy


def main():
    # rospy.init_node("test_yolo", anonymous=True)
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
    pipeline.start(config)
    # write a one line function to round the tensor to nearest integer
    round_tensor = lambda x: round(float(x.data))
    while 1:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = np.asanyarray(depth_frame.get_data())
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
                    if class_name == "Tomato Sauce":
                        break
        # mask depth except the bounding box
        mask = np.zeros_like(depth_frame)
        mask[
            round_tensor(bb_box[1]) : round_tensor(bb_box[3]), round_tensor(bb_box[0]) : round_tensor(bb_box[2])
        ] = depth_frame[
            round_tensor(bb_box[1]) : round_tensor(bb_box[3]), round_tensor(bb_box[0]) : round_tensor(bb_box[2])
        ]
        # Find the pixel coordinates of the lowest value other than zero in the mask

        # highest_pixel = np.where(mask == np.min(mask[np.nonzero(mask)]))
        # cv2.circle(color_image, (highest_pixel[1][0], highest_pixel[0][0]), 10, (0, 255, 0), -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Find the convex hull for each contour
        for contour in contours:
            hull = cv2.convexHull(contour)
            cv2.drawContours(color_image, [hull], -1, (255, 255, 255), 2)

        cv2.imshow("image", color_image)
        # sleep for 1 ms
        sleep(0.1)
        # wait for q key to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
