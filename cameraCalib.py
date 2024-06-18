import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=3, suppress=True)

import rtde_control
import rtde_receive

import EasyPySpin

""" 参考URL
カメラの外部パラメータ行列を最小二乗法で推定する
https://dev.classmethod.jp/articles/estimate-camera-external-parameter-matrix-with-least-squares/
cv2.calibrateCamera()関数の解説
https://kamino.hatenablog.com/entry/opencv_calibrate_camera
座標系の定義
https://stackoverflow.com/questions/67072289/eye-in-hand-calibration-opencv
LookAt関数の実装例
https://github.com/stemkoski/three.py/blob/master/three.py/mathutils/MatrixFactory.py#L104
"""

# ロボットのIPアドレス
ROBOT_IP = "192.168.11.100"

# チェスボードのサイズ
SQUARE_SIZE = 5.5       # 正方形の1辺のサイズ[mm]
PATTERN_SIZE = (8, 4)  # 交差ポイントの数

# リアルセンスの設定
WIDTH, HEIGHT = 640, 480  # 画像サイズ
CLIPPING_DISTANCE = 0.5   # Depthに応じてマスクをかける時の閾値 (今回は使わない)
CROP_SIZE = 224           # クロップ後の画像サイズ
CROP_CENTER = [320, 240]  # クロップ中心

# ロボットの姿勢変更に関する設定
RANGE_X, RANGE_Y, RANGE_Z = 0.005, 0.005, 0.0001  # ロボットが動き回る範囲 [+/- m]
RANGE_THETA = 10  # カメラの光軸周り角度の変更範囲 [+/- deg]
N = 5  # 姿勢を変えて N * N 枚の画像を撮影

# ロボットのフランジから見たカメラの姿勢
# 画像の下方向がy+, 画像の右方向がx+
# おおよその値を入れておき、cv2.calibrateHandEye()で正確な値を求める。
# INITIAL_TCP = [0.0, -0.1, 0.01, 0.0, 0.0, 0.0]
INITIAL_TCP = [0.00, 0.05, 0.3185, 0.0, 0.0, 0.0]

# ベース座標系で画像の上方向がおおよそどっちを向いてるか
UP = [0, 1, 0]

# ロボットのホームポジション
# INITIAL_ROBOT_POSE = [0.6, 0.0, 0.5, 2.22, -2.22, 0.0]
INITIAL_ROBOT_POSE = [0.01312, -0.535, 0.013, 0.0, 3.14, 0.0]

# ターゲットのチェスボードのベース座標系におけるおおよその中心位置
# INITIAL_TARGET_POSE = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0]
INITIAL_TARGET_POSE = [0.01312, -0.535, 0.0, 0.0, 0.0, 0.0]

class RobotContoller:
    def __init__(self, robot_ip):
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.debug = False

    def stop(self):
        self.rtde_c.jogStop()
        self.rtde_c.servoStop()
        self.rtde_c.stopScript()
        
    def set_tcp(self, point):
        self.rtde_c.setTcp(point)

    def moveL(self, pose, vel=0.1, acc=1.2, asynchronous=False):
        self.rtde_c.moveL(pose, vel, acc, asynchronous)

    def getTCPPoseRotvec(self):
        return self.rtde_r.getActualTCPPose()

class CameraCalib:
    def __init__(self):
        self.robot = None
        self.initRobotTCP = np.array(INITIAL_TCP)
        self.initRobotPose = np.array(INITIAL_ROBOT_POSE)
        self.initTargetPose = np.array(INITIAL_TARGET_POSE)

    def bringupRobot(self, robot_ip):
        self.robot = RobotContoller(robot_ip)
        self.robot.set_tcp(self.initRobotTCP)
        self.robot.moveL(self.initRobotPose, vel=0.05, acc=0.5)

    def generate_shooting_pose(self, N=5):
        init_x, init_y, init_z = self.initRobotPose[:3]
        xs_ = np.linspace(init_x - RANGE_X, init_x + RANGE_X, N)
        ys_ = np.linspace(init_y - RANGE_Y, init_y + RANGE_Y, N)
        for x in xs_:
            for y in ys_:
                theta = (np.random.rand() - 0.5) * 2 * np.radians(90-RANGE_THETA)
                LookAtRot = self.LookAt(
                    eye=np.array([x, y, init_z]),
                    target=self.initTargetPose[:3],
                    up=R.from_rotvec([0, 0, theta]).apply(UP))
                # LookAtRot = self.create_look_at_mat(np.array([x, y, init_z]), self.initTargetPose[:3], R.from_rotvec([0, 0, theta]).apply(UP))
                LookAtRotVec = LookAtRot.as_rotvec()
                z = (np.random.rand() - 0.5) * 2 * RANGE_Z + init_z
                yield [x, y, z] + LookAtRotVec.tolist()
            ys_ = ys_[::-1]

    def LookAt(self, eye, target, up):
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        # If forward and up vectors are parallel, right vector is zero
        # Fix by perturbing up vector a bit
        if np.linalg.norm(right) < 0.001:
            epsilon = np.array([0.001, 0, 0])
            right = np.cross(forward, up + epsilon)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        LookAtRot = R.from_matrix([
            [-right[0], up[0], forward[0]],
            [-right[1], up[1], forward[1]],
            [-right[2], up[2], forward[2]]])
        return LookAtRot

    def create_translation_mat(self, x: float, y: float, z: float):
        trans_mat = np.eye(4)
        trans_mat[:, 3] = [x, y, z, 1]

        return trans_mat


    def create_scale_mat(self, x: float, y: float, z: float):
        scale_mat = np.eye(4)

        scale_mat[0, 0] = x
        scale_mat[1, 1] = y
        scale_mat[2, 2] = z

        return scale_mat

    def create_look_at_mat(self, eye, target, up=[0.0, 1.0, 0.0]):
        # Open3D world coordinate system: right-handed coordinate system (Y up, same as OpenGL)
        # Open3D camera coordinate system: right-handed coordinate system (Y down, Z forward, same as OpenCV) 
        # https://github.com/intel-isl/Open3D/issues/1347

        eye = np.array(eye, dtype=np.float32, copy=True)
        target = np.array(target, dtype=np.float32, copy=True)
        up = np.array(up, dtype=np.float32, copy=True)

        z = eye - target
        z = z / np.linalg.norm(z)

        x = np.cross(up, z)
        x = x / np.linalg.norm(x)

        y = np.cross(z, x)
        y = y / np.linalg.norm(y)

        rotate_mat = np.array([
            [x[0], x[1], x[2], 0.0],
            [y[0], y[1], y[2], 0.0],
            [z[0], z[1], z[2], 0.0],
            [0, 0, 0, 1]
        ])

        trans_mat = self.create_translation_mat(-eye[0], -eye[1], -eye[2])

        scale_mat = self.create_scale_mat(1, -1, -1)

        tmp = np.dot(rotate_mat, trans_mat)
        tmp = np.dot(scale_mat, tmp)
        tmp = R.from_matrix(tmp[:3,:3])
        return tmp

    def calibMove(self, N):
        before = time.time()
        crop_settings = {
            "crop_size": CROP_SIZE,
            "crop_origin_x": CROP_CENTER[0] - int(CROP_SIZE/2),
            "crop_origin_y": CROP_CENTER[1] - int(CROP_SIZE/2)}
        # rs = RealSense(WIDTH, HEIGHT, CLIPPING_DISTANCE, [crop_settings] * 2)
        # cap = cv2.VideoCapture(0)
        cap = EasyPySpin.VideoCapture(0)
        time.sleep(1)
        # チェッカーボードの交点の座標の指定 (X, Y, Z=0)
        pattern_points = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
        pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
        pattern_points *= SQUARE_SIZE / 1000  # [mm] -> [m]
        # cv2.calibrateCamera()の返り値 rvecs, tvecs や cv2.calibrateHandEye()の
        # 返り値は引数と同じ単位系になる。今回は、[m] に統一して入力
        objpoints, imgpoints = [], []
        b_R_g, b_t_g = [], []
        shooting_pose_generator = self.generate_shooting_pose(N)
        for _ in range(N * N):
            shooting_pose = next(shooting_pose_generator)
            print(shooting_pose)
            self.robot.moveL(shooting_pose, vel=0.12, acc=0.8)
            time.sleep(0.5)
            # color_images, _, _, _ = rs.get_image(crop=True)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = color_images[0]
            _, gray = cap.read()
            gray= cv2.resize(gray, (640, 480))
            # チェスボードのコーナーを検出
            ret, corner = cv2.findChessboardCorners(gray, PATTERN_SIZE)
            # コーナーがあれば
            if ret == True:
                print("Detected corner: {}/{}".format(len(objpoints) + 1, N * N))
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), term)
                imgpoints.append(corner.reshape(-1, 2))
                objpoints.append(pattern_points)
                b_T_g = self.get_b_T_g()
                b_R_g.append(b_T_g[3:])
                b_t_g.append(b_T_g[:3])
            cv2.imshow('image', gray)
            if cv2.waitKey(100) == 27:
                break
        self.robot.moveL(self.initRobotPose, vel=0.05, acc=0.5)
        print("Calculating camera parameters...")
        # 内部パラメータを計算
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # 計算結果を保存
        np.save("mtx", mtx) # カメラ行列
        np.save("dist", dist.ravel()) # 歪みパラメータ
        # 計算結果を表示
        print("RMS = ", np.round(ret, 1), "[pixel]")
        print("Intrinsic parameters ([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) [pixel]:\n", mtx)
        print("Distortion parameters (k1, k2, p1, p2, k3):", dist.ravel())
        c_R_t = np.hstack(rvecs).T
        c_t_t = np.hstack(tvecs).T
        # print("c_t_t [mm]:", np.round(c_t_t * 1000, 1))
        # print("c_R_t [deg]:", np.round(np.degrees(c_R_t), 1))
        b_R_g = np.array(b_R_g)
        b_t_g = np.array(b_t_g)
        # print("b_t_g [mm]:", np.round(b_t_g * 1000, 1))
        # print("b_R_g [deg]:", np.round(np.degrees(b_R_g), 1))
        # 外部パラメータを計算
        g_R_c, g_t_c = cv2.calibrateHandEye(b_R_g, b_t_g, c_R_t, c_t_t)
        g_R_c_rotvec = R.from_matrix(g_R_c).as_rotvec()
        # 計算結果を保存
        np.save("g_t_c", g_t_c.ravel())
        np.save("g_R_c", g_R_c_rotvec)
        # 計算結果を表示
        print("g_t_c [mm]:", np.round(g_t_c.ravel() * 1000, 1))
        print("g_R_c [deg]:", np.degrees(g_R_c_rotvec))
        # キャリブレにかかった時間
        print("Elapsed time: {} [sec]".format(int(time.time() - before)))

    def print_curr_pose(self):
        curr_pose = np.array(self.robot.getTCPPoseRotvec())
        curr_pos = curr_pose[:3]
        curr_rot = curr_pose[3:]
        print(np.round(curr_pos * 1000, 1), np.round(np.degrees(curr_rot), 1))

    def get_b_T_g(self):
        self.robot.set_tcp([0.0] * 6)
        b_T_g = np.array(self.robot.getTCPPoseRotvec())
        self.robot.set_tcp(self.initRobotTCP)
        return b_T_g
    
    def detectChess(self):
       
        # cap = cv2.VideoCapture(0)
        cap = EasyPySpin.VideoCapture(0)
        time.sleep(1)
        # チェッカーボードの交点の座標の指定 (X, Y, Z=0)
        pattern_points = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
        pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
        pattern_points *= SQUARE_SIZE / 1000  # [mm] -> [m]
        # cv2.calibrateCamera()の返り値 rvecs, tvecs や cv2.calibrateHandEye()の
        # 返り値は引数と同じ単位系になる。今回は、[m] に統一して入力
        objpoints, imgpoints = [], []
        b_R_g, b_t_g = [], []
        print("camera")
        while True:
            before = time.time()
            # _, img = cap.read()
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, gray = cap.read()
            print(gray.shape)
            # gray = gray[1824:1824+480, 2736:2736+640]
            gray= cv2.resize(gray, (640, 480))
            print("read")
            # チェスボードのコーナーを検出
            ret, corner = cv2.findChessboardCorners(gray, PATTERN_SIZE)
            # コーナーがあれば
            if ret == True:
                print("Detected corner: {}/{}".format(len(objpoints) + 1, N * N))
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), term)
                # imgpoints.append(corner.reshape(-1, 2))
                # objpoints.append(pattern_points)
                # b_T_g = self.get_b_T_g()
                # b_R_g.append(b_T_g[3:])
                # b_t_g.append(b_T_g[:3])
            cv2.imshow("frame_calib", gray)
            if cv2.waitKey(10) == 27:
                break
            


if __name__ == "__main__":
    cc = CameraCalib()
    cc.bringupRobot(ROBOT_IP)
    cc.calibMove(N)
    # cc.detectChess()