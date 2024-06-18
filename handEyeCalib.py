import time
import cv2
import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=3, suppress=True)

from charuco.charuco import ChArucoModule

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

# ロボットの姿勢変更に関する設定
RANGE_X, RANGE_Y, RANGE_Z = 0.005, 0.005, 0.0001  # ロボットが動き回る範囲 [+/- m]
RANGE_THETA = 10  # カメラの光軸周り角度の変更範囲 [+/- deg]

# ロボットのフランジから見たカメラの姿勢
# 画像の下方向がy+, 画像の右方向がx+
# おおよその値を入れておき、cv2.calibrateHandEye()で正確な値を求める。
# INITIAL_TCP = [0.0, -0.1, 0.0, 0.0, 0.0, 0.0]
INITIAL_TCP = [0.00, 0.05, 0.3185, 0.0, 0.0, 0.0]

# ベース座標系とTCP座標系の対応
UP = [0, 1, 0]

# ロボットのホームポジション
# INITIAL_ROBOT_POSE = [0.6, 0.0, 0.5, 2.22, -2.22, 0.0]
INITIAL_ROBOT_POSE = [0.01512, -0.535, 0.013, 0.0, 3.14, 0.0]

# ターゲットのチェスボードのベース座標系におけるおおよその中心位置
# INITIAL_TARGET_POSE = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0]
INITIAL_TARGET_POSE = [0.01512, -0.535, 0.0, 0.0, 0.0, 0.0]

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

class HandEyeCalib:
    def __init__(self):
        self.robot = None
        self.initRobotTCP = np.array(INITIAL_TCP)
        self.initRobotPose = np.array(INITIAL_ROBOT_POSE)
        self.initTargetPose = np.array(INITIAL_TARGET_POSE)
        self.charuco = ChArucoModule(camera_id=0)

    def bringupRobot(self, robot_ip="192.168.11.30"):
        self.robot = RobotContoller(robot_ip)
        self.robot.set_tcp(self.initRobotTCP)
        self.robot.moveL(self.initRobotPose, vel=0.05, acc=0.5)
        # self.robot.moveL(self.initRobotPose, vel=1.05, acc=1.5)

    def generate_shooting_pose(self, N=5):
        init_x, init_y, init_z = self.initRobotPose[:3]
        xs_ = np.linspace(init_x - RANGE_X, init_x + RANGE_X, N)
        ys_ = np.linspace(init_y - RANGE_Y, init_y + RANGE_Y, N)
        for x in xs_:
            for y in ys_:
                theta = (np.random.rand() - 0.5) * 2 * np.radians(RANGE_THETA)
                LookAtRot = self.LookAt(
                    eye=np.array([x, y, init_z]),
                    target=self.initTargetPose[:3],
                    up=R.from_rotvec([0, 0, theta]).apply(UP))
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

    def calibMove(self, N, verpose=False):
        before = time.time()
        tvecs, rvecs = [], []
        b_R_g, b_t_g = [], []
        shooting_pose_generator = self.generate_shooting_pose(N)
        for _ in range(N * N):
            shooting_pose = next(shooting_pose_generator)
            print(shooting_pose)
            self.robot.moveL(shooting_pose, vel=0.02, acc=0.2)
            # self.robot.moveL(shooting_pose, vel=1.02, acc=1.2)
            time.sleep(0.5)
            cam_T_center, _, color_image = self.charuco.get_board_pose()
            if np.all(cam_T_center != None):
                print("cam_T_charuco_center:",
                    np.round(cam_T_center[:3] * 1000, 1), "[mm]",
                    np.round(np.degrees(cam_T_center[3:]), 1), "[deg]")
                color_image = cv2.drawFrameAxes(
                    color_image, self.charuco.cameraMatrix, self.charuco.distCoeffs,
                    cam_T_center[3:], cam_T_center[:3], self.charuco.axis_length, thickness=4)
                tvecs.append(cam_T_center[:3])
                rvecs.append(cam_T_center[3:])
                b_T_g = self.get_b_T_g()
                b_R_g.append(b_T_g[3:])
                b_t_g.append(b_T_g[:3])
            cv2.imshow('image', color_image)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break
        self.robot.moveL(self.initRobotPose, vel=0.05, acc=0.5)
        c_R_t = np.array(rvecs)
        c_t_t = np.array(tvecs)
        b_R_g = np.array(b_R_g)
        b_t_g = np.array(b_t_g)
        if verpose:
            print("c_t_t [mm]:", np.round(c_t_t * 1000, 1))
            print("c_R_t [deg]:", np.round(np.degrees(c_R_t), 1))
            print("b_t_g [mm]:", np.round(b_t_g * 1000, 1))
            print("b_R_g [deg]:", np.round(np.degrees(b_R_g), 1))
        # 外部パラメータを計算
        g_R_c, g_t_c = cv2.calibrateHandEye(b_R_g, b_t_g, c_R_t, c_t_t)
        g_R_c_rotvec = R.from_matrix(g_R_c).as_rotvec()
        # 計算結果を保存
        np.save("g_t_c", g_t_c.ravel())
        np.save("g_R_c", g_R_c_rotvec)
        # 計算結果を表示
        print("g_t_c [mm]:", np.round(g_t_c.ravel() * 1000, 1))
        print("g_R_c [deg]:", np.round(np.degrees(g_R_c_rotvec), 1))
        # キャリブレにかかった時間
        print("Elapsed time: {} [sec]".format(int(time.time() - before)))

    def print_curr_pose(self):
        curr_pose = np.array(self.robot.get_pose())
        curr_pos = curr_pose[:3]
        curr_rot = curr_pose[3:]
        print(np.round(curr_pos * 1000, 1), np.round(np.degrees(curr_rot), 1))

    def get_b_T_g(self):
        self.robot.set_tcp([0.0] * 6)
        b_T_g = np.array(self.robot.get_pose())
        self.robot.set_tcp(self.initRobotTCP)
        return b_T_g

if __name__ == "__main__":
    robot_ip="192.168.11.30"
    hec = HandEyeCalib()
    hec.bringupRobot(robot_ip)
    N = 5  # 姿勢を変えて N * N 枚の画像を撮影
    hec.calibMove(N, verpose=False)

"""
g_t_c [mm]: [ -9.5 -98.9 -13.7]
g_R_c [deg]: [-0.2 -0.1 -1. ]
"""