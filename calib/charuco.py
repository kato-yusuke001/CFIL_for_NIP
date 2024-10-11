import cv2
import os, math, datetime, subprocess, csv
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
from pytransform3d.transform_manager import TransformManager

from rs_utils import RealSense
from flir_utils import Flir
from usbcam_utils import UsbCam

# インストール
# https://python-academia.com/opencv-aruco/
# OpenCVとバージョンを合わせる必要がある
# pip install opencv-python==4.7.0.68
# pip install opencv-contrib-python==4.7.0.68

# OpenCVでArUcoのCharucoBoardを作る
# https://openi.jp/blogs/main/software-opencv-aruco-charuco

class ChArucoModule:
    def __init__(self, squares_x=12, squares_y=9, square_length=0.06, marker_length=0.045,
            aruco_dict="5X5_100", ppi=300, save_charuco_image=True, axis_length=0.01, camera_id=0, device="D405"):
        self.aruco = cv2.aruco
        if aruco_dict == "5X5_100":
            self.aruco_dict = self.aruco.getPredefinedDictionary(self.aruco.DICT_5X5_100)
        elif aruco_dict == "6X6_250":
            self.aruco_dict = self.aruco.getPredefinedDictionary(self.aruco.DICT_6X6_250)
        else:
            raise NotImplementedError
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        self.board_size_x = self.squares_x * self.square_length
        self.board_size_y = self.squares_y * self.square_length
        self.axis_length = axis_length
        self.camera_id = camera_id
        self.device = device
        # The constructor cv2.aruco.CharucoBoard_create has been renamed to cv2.aruco.CharucoBoard.
        # https://stackoverflow.com/questions/75085270/cv2-aruco-charucoboard-create-not-found-in-opencv-4-7-0
        self.charucoBoard = self.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_length, self.marker_length, self.aruco_dict)
        # 【OpenCV】4.6.0から4.7.0でのArUcoモジュールの更新箇所抜粋
        # https://qiita.com/koichi_baseball/items/d51373e7fd6dddb57d1f
        self.arucoParams = self.aruco.DetectorParameters()
        if save_charuco_image:
            image_x = round(self.squares_x * self.square_length * 1000 / 25.4 * ppi)
            image_y = round(self.squares_y * self.square_length * 1000 / 25.4 * ppi)
            image = self.charucoBoard.generateImage((image_x, image_y))
            fp = os.path.join("charuco_board.png")
            cv2.imwrite(fp, image)
            # ppiを変更
            subprocess.run(['convert', '-density', str(ppi), '-units', 'pixelsperinch', fp, fp])
        # Start realsense pipeline
        # BundleSDFのデフォルト画像サイズは640x480
        if self.device=="D405":
            cam_settings = {"width": 640, "height": 480, "fps": 30, "clipping_distance": 1.0}
            self.cam = RealSense(**cam_settings)
            self.cameraMatrix = self.cam.cameraMatrix_woCrop[camera_id]
            self.distCoeffs = self.cam.distCoeffs[camera_id]
        elif self.device=="flir":
            self.cam = Flir()
        elif self.device=="usb_cam":
            self.cam = UsbCam()
            self.cameraMatrix = np.load("camera_matrix.npy")
            self.distCoeffs = np.load("distortion_coefficients.npy")
        else:
            raise NotImplementedError
        self.tm = TransformManager()

    def make_save_directory(self):
        save_dir = os.path.join("images", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        for subdir in ["rgb", "depth_enhanced", "cam_in_ob"]:
            if not os.path.exists(os.path.join(save_dir, subdir)):
                os.makedirs(os.path.join(save_dir, subdir))
        return save_dir

    def save_data(self, color, depth, cam_in_ob, save_dir, i):
        np.save(os.path.join(save_dir, "rgb", "{:07d}.npy".format(i)), color)
        np.save(os.path.join(save_dir, "depth_enhanced", "{:07d}.npy".format(i)), depth)
        np.save(os.path.join(save_dir, "cam_in_ob", "{:07d}.npy".format(i)), cam_in_ob)

    def posture_estimation(self, save=False, verpose=False):
        # Camera Posture Estimation Using A ChArUco Board
        # https://longervision.github.io/2017/03/13/ComputerVision/OpenCV/opencv-external-posture-estimation-ChArUco-board/
        # https://github.com/opencv/opencv-python/issues/755
        save_dir = self.make_save_directory()
        # self.cam.save_camera_param(save_dir)
        board_trans, board_rotvec, board_euler = [], [], []
        i = 0
        while(True):
            cam_T_center, corners, charucoCorners, color_image, depth_image = self.get_board_pose()
            if verpose:
                print(cam_T_center)
            if np.any(cam_T_center != None):
                board_trans.append(cam_T_center[:3])
                board_rotvec.append(cam_T_center[3:])
                board_euler.append(R.from_rotvec(board_rotvec[-1]).as_euler("XYZ", degrees=True))
                if verpose:
                    print(board_trans[-1] * 1000, "[mm]", board_euler[-1], "[deg]")
                color_image = self.draw_coord(color_image, board_trans[-1], board_rotvec[-1], size=4)
            if np.any(corners != None):
                # posture estimation of single markers
                rvecs, tvecs, objPoints = self.aruco.estimatePoseSingleMarkers(
                    corners, markerLength=self.marker_length,
                    cameraMatrix=self.cameraMatrix,
                    distCoeffs=self.distCoeffs)
                for rvec, tvec in zip(rvecs, tvecs):
                    color_image = self.draw_coord(color_image, tvec, rvec, size=2)
            if self.device=="D405":
                depth_image = depth_image / self.cam.clipping_distance
                depth_image = np.clip(depth_image, 0, 1)
                depth_image = (depth_image * 255.).astype(np.uint8)
                depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
                vis = np.hstack([color_image, depth_image])
            else:
                vis = color_image
            multiple = 2
            height = vis.shape[0]
            width = vis.shape[1]
            vis = cv2.resize(vis, (int(width * multiple), int(height * multiple)))
            cv2.imshow("charucoboard", vis)
            key = cv2.waitKey(2)
            if key & 0xFF == ord("q"):
                break
            elif key & 0xFF == ord("s"):
                cam_in_ob = self.tm.get_transform("cam", "center")
                obj_T_cam = self.transform_matrix_to_vector(cam_in_ob)
                print(obj_T_cam[:3] * 1000, "[mm]")
                print(R.from_rotvec(obj_T_cam[3:]).as_euler("XYZ", degrees=True), "[deg]")
                self.save_data(self.color_image_original, self.depth_image_original, cam_in_ob, save_dir, i)
                i += 1
        cv2.destroyAllWindows()
        # Print posture
        trans = np.array(board_trans).mean(axis=0)
        rotvec = np.array(board_rotvec).mean(axis=0)
        euler = np.array(board_euler).mean(axis=0)
        print("trans", trans * 1000, "[mm]")
        if verpose:
            print("rotvec", rotvec, "[rad]")
        print("euler", euler, "[deg]")
        # Save csv
        if save:
            csv_fp = os.path.join("charuco_origin.csv")
            with open(csv_fp, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(trans.tolist() + rotvec.tolist())

    def get_board_pose(self, draw_corners=True):
        color_images, depth_images, _, _ = self.cam.get_image(crop=False, get_mask=False)
        color_image = color_images[self.camera_id]
        self.color_image_original = color_image.copy()
        if self.device=="D405":
            depth_image = depth_images[self.camera_id]
            self.depth_image_original = depth_image.copy()
        else:
            depth_image = None
            self.depth_image_original = None
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
                if draw_corners:
                    color_image = self.aruco.drawDetectedCornersCharuco(
                        color_image, charucoCorners, charucoIds, (0, 255, 0))
                retval, rvec, tvec = self.aruco.estimatePoseCharucoBoard(
                    charucoCorners, charucoIds, self.charucoBoard, self.cameraMatrix, self.distCoeffs,
                    np.empty(1), np.empty(1), useExtrinsicGuess=False)
                if retval == True:
                    # Make board center origin
                    self.register_pose(tvec[:, 0], rvec[:, 0], "cam", "corner")
                    rvec_co_T_ce = R.from_euler("XYZ", [180, 0, 0], degrees=True).as_rotvec()
                    tvec_co_T_ce = [
                        self.square_length * math.ceil(self.squares_x / 2),
                        self.square_length * math.ceil(self.squares_y / 2),
                        0]
                    self.register_pose(tvec_co_T_ce, rvec_co_T_ce, "corner", "center")
                    cam_T_center = self.get_pose("cam", "center")
                else:
                    cam_T_center = None
            else:
                cam_T_center = None
        else:
            cam_T_center = None
            corners = None
            charucoCorners = None
        return cam_T_center, corners, charucoCorners, color_image, depth_image

    def transform_matrix_to_vector(self, transform):
        tvec = transform[:3, 3]
        rvec = R.from_matrix(transform[:3, :3]).as_rotvec()
        # 回転ベクトルの絶対値を180°以下にする。
        while True:
            dr = np.linalg.norm(rvec)
            if dr > np.radians(180):
                rvec = rvec * (dr - np.radians(360)) / dr
            else:
                break
        return np.hstack([tvec, rvec])

    def register_pose(self, tvec, rvec, source, target):
        source_T_target = pytr.transform_from(
            pyrot.matrix_from_compact_axis_angle(rvec), tvec)
        self.tm.add_transform(target, source, source_T_target)

    def remove_pose(self, source, target):
        self.tm.remove_transform(target, source)

    def get_pose(self, source, target):
        return self.transform_matrix_to_vector(self.tm.get_transform(target, source))

    def get_matrix(self, source, target):
        return self.tm.get_transform(target, source)

    def draw_coord(self, image, tvec, rvec, size=1):
        return cv2.drawFrameAxes(image, self.cameraMatrix, self.distCoeffs,
            rvec, tvec, self.axis_length * size, thickness=size)

    def close(self):
        self.cam.close()

if __name__ == "__main__":
    # 印刷の場合 ppi = 300
    # Philips 273B モニタの場合 ppi = 82 
    charuco = ChArucoModule(
        squares_x=7, squares_y=5, square_length=0.02, marker_length=0.01,
        aruco_dict="6X6_250", ppi=300, save_charuco_image=False,
        axis_length=0.01, camera_id=0, device="usb_cam")
    charuco.posture_estimation()