import cv2
import EasyPySpin
from charuco.charuco import ChArucoModule

# cap = EasyPySpin.VideoCapture(0)

# ret, frame = cap.read()

# cv2.imwrite("frame.png", frame)

# cap.release()

charuco = ChArucoModule(camera_id=0)
import time
time.sleep(1)
color_image = charuco.get_board_pose()