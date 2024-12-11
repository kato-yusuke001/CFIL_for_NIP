import cv2
import os, sys, inspect, json, subprocess, csv
import numpy as np
from scipy.spatial.transform import Rotation as R
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
from pytransform3d.transform_manager import TransformManager

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)

import EasyPySpin

# インストール
# https://python-academia.com/opencv-aruco/
# OpenCVとバージョンを合わせる必要がある
# pip install opencv-contrib-python==4.9.0.80

# OpenCVでArUcoのCharucoBoardを作る
# https://openi.jp/blogs/main/software-opencv-aruco-charuco

ppi = 350  # 印刷の場合
# ppi = 82 # Philips 273B モニタの場合
save_image = False

def transform_matrix_to_vector(transform):
    tvec = transform[:3, 3]
    rvec = R.from_matrix(transform[:3, :3]).as_rotvec()
    return np.hstack([tvec, rvec])

class ChArucoModule:
    def __init__(self, camera_id):
        self.aruco = cv2.aruco
        self.aruco_dict = self.aruco.getPredefinedDictionary(self.aruco.DICT_6X6_1000)
        self.squares_x = 9
        self.squares_y = 5
        self.square_length = 0.0055
        self.marker_length = 0.0035
        self.board_size_x = self.squares_x * self.square_length
        self.board_size_y = self.squares_y * self.square_length
        self.axis_length = 0.01
        self.camera_id = camera_id
        # The constructor cv2.aruco.CharucoBoard_create has been renamed to cv2.aruco.CharucoBoard.
        # https://stackoverflow.com/questions/75085270/cv2-aruco-charucoboard-create-not-found-in-opencv-4-7-0
        self.charucoBoard = self.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_length, self.marker_length, self.aruco_dict)
        # 【OpenCV】4.6.0から4.7.0でのArUcoモジュールの更新箇所抜粋
        # https://qiita.com/koichi_baseball/items/d51373e7fd6dddb57d1f
        self.arucoParams = self.aruco.DetectorParameters()
        if save_image:
            image_x = round(self.squares_x * self.square_length * 1000 / 25.4 * ppi)
            image_y = round(self.squares_y * self.square_length * 1000 / 25.4 * ppi)
            image = self.charucoBoard.generateImage((image_x, image_y))
            fp = os.path.join(currentdir, "charuco_board.png")
            cv2.imwrite(fp, image)
            # ppiを変更
            subprocess.run(['convert', '-density', str(ppi), '-units', 'pixelsperinch', fp, fp])
        # Start realsense pipeline
        self.cap = EasyPySpin.VideoCapture(0)
        self.cameraMatrix = self.cap.cameraMatrix_woCrop[camera_id]
        self.distCoeffs = self.cap.distCoeffs[camera_id]

    def posture_estimation(self):
        # Camera Posture Estimation Using A ChArUco Board
        # https://longervision.github.io/2017/03/13/ComputerVision/OpenCV/opencv-external-posture-estimation-ChArUco-board/
        # https://github.com/opencv/opencv-python/issues/755
        board_trans, board_rotvec, board_euler = [], [], []
        while(True):
            cam_T_center, corners, color_image = self.get_board_pose()
            print(cam_T_center)
            if np.any(cam_T_center != None):
                board_trans.append(cam_T_center[:3])
                board_rotvec.append(cam_T_center[3:])
                board_euler.append(R.from_rotvec(board_rotvec[-1]).as_euler("XYZ", degrees=True))
                print(board_trans[-1] * 1000, "[mm]", board_euler[-1], "[deg]")
                color_image = cv2.drawFrameAxes(
                    color_image, self.cameraMatrix, self.distCoeffs,
                    board_rotvec[-1], board_trans[-1], self.axis_length, thickness=4)
            if np.any(corners != None):
                # posture estimation of single markers
                rvecs, tvecs, objPoints = self.aruco.estimatePoseSingleMarkers(
                    corners, markerLength=self.marker_length,
                    cameraMatrix=self.cameraMatrix,
                    distCoeffs=self.distCoeffs)
                for rvec, tvec in zip(rvecs, tvecs):
                    color_image = cv2.drawFrameAxes(
                        color_image, self.cameraMatrix, self.distCoeffs,
                        rvec, tvec, self.axis_length, thickness=2)
            multiple = 2
            height = color_image.shape[0]
            width = color_image.shape[1]
            color_image = cv2.resize(color_image, (int(width * multiple), int(height * multiple)))
            cv2.imshow("charucoboard", color_image)
            if cv2.waitKey(2) & 0xFF == ord("q"):
                break
        # Print posture
        trans = np.array(board_trans).mean(axis=0)
        rotvec = np.array(board_rotvec).mean(axis=0)
        euler = np.array(board_euler).mean(axis=0)
        print("trans", trans * 1000, "[mm]")
        print("rotvec", rotvec, "[rad]")
        print("euler", euler, "[deg]")
        # Save csv
        csv_fp = os.path.join(currentdir, "charuco_origin.csv")
        with open(csv_fp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(trans.tolist() + rotvec.tolist())

    def get_board_pose(self):
        # color_images, _, _, _ = self.cap.get_image(crop=False, get_mask=False)
        _, color_image = self.cap.read()
        # color_image = color_images[self.camera_id]
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # Detect markers
        corners, ids, rejectedImgPoints = self.aruco.detectMarkers(
            gray_image, self.aruco_dict, parameters=self.arucoParams)
        self.aruco.refineDetectedMarkers(
            gray_image, self.charucoBoard, corners, ids, rejectedImgPoints)
        # posture estimation from a charuco board
        if np.all(ids != None):  # if there is at least one marker detected
            charucoretval, charucoCorners, charucoIds = self.aruco.interpolateCornersCharuco(
                corners, ids, gray_image, self.charucoBoard)
            if charucoretval > 0:
                color_image = self.aruco.drawDetectedCornersCharuco(
                    color_image, charucoCorners, charucoIds, (0, 255, 0))
                retval, rvec, tvec = self.aruco.estimatePoseCharucoBoard(
                    charucoCorners, charucoIds, self.charucoBoard, self.cameraMatrix, self.distCoeffs,
                    np.empty(1), np.empty(1), useExtrinsicGuess=False)
                if retval == True:
                    tm = TransformManager()
                    # Make board center origin
                    cam_T_corner = pytr.transform_from(
                        pyrot.matrix_from_compact_axis_angle(rvec[:, 0]), tvec[:, 0])
                    tm.add_transform("corner", "cam", cam_T_corner)
                    corner_T_center = pytr.transform_from(
                        pyrot.matrix_from_compact_axis_angle(
                            R.from_euler("XYZ", [0, 0, 180], degrees=True).as_rotvec()),
                            [self.board_size_x * 0.5, self.board_size_y * 0.5, 0])
                    tm.add_transform("center", "corner", corner_T_center)
                    cam_T_center = transform_matrix_to_vector(tm.get_transform("center", "cam"))
                else:
                    cam_T_center = None
            else:
                cam_T_center = None
        else:
            corners = None
        return cam_T_center, corners, color_image

if __name__ == "__main__":
    charuco = ChArucoModule(camera_id=0)
    # charuco.posture_estimation()