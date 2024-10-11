import os, time, argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
from scipy.spatial.transform import Rotation as R
from scipy.optimize import curve_fit 
from collections import deque

from ur_utils import URController
from charuco import ChArucoModule
from pose_utils import *

class camera_caribration:
    def __init__(self, **kwargs):
        """
        tcp: Camera pose from robot flange.
        up:  Rough direction of Y axis of TCP (=camera) coordinates in base coordinates.
             Y axis of camera coordinates is downward in images.
        """
        self.robot_ip = kwargs.get("robot_ip")
        self.home_pose = kwargs.get("home_pose")
        self.tcp = kwargs.get("tcp")
        self.up = kwargs.get("up")
        self.charuco_z = kwargs.get("charuco_z")
        self.depth_to_dist = kwargs.get("depth_to_dist")
        self.device = kwargs.get("device")
        self.N = kwargs.get("N")
        self.min_r = kwargs.get("min_r")
        self.max_r = kwargs.get("max_r")
        self.max_theta = np.radians(kwargs.get("max_theta"))
        self.z_escape = kwargs.get("z_escape")
        self.rot_temperature = kwargs.get("rot_temperature")
        self.random = kwargs.get("random")
        self.use_robot = kwargs.get("use_robot")
        self.use_charuco = kwargs.get("use_charuco")
        self.auto_data_collection = kwargs.get("auto_data_collection")
        self.charuco = ChArucoModule(
            squares_x=12, squares_y=9, square_length=0.06, marker_length=0.045,
            aruco_dict="5X5_100", save_charuco_image=False, camera_id=0)
        for device_name in self.charuco.cam.device_name:
            assert self.device in device_name
        if self.use_robot:
            self.robot = URController(self.robot_ip, self.home_pose, self.tcp)
            # Register camera pose from robot base
            base_T_cam = self.robot.get_pose()
            self.charuco.register_pose(base_T_cam[:3], base_T_cam[3:], "base", "home_cam")
        else:
            self.charuco.register_pose(self.home_pose[:3], self.home_pose[3:], "base", "home_cam")
        # base_T_charuco_center
        self.charuco_pose = self.get_charuco_pose()

    def get_charuco_pose(self):
        for _ in range(20):
            self.charuco.cam.get_image(crop=False, get_mask=False)
        if self.use_charuco:
            depth_list = deque(maxlen=100)
            charuco_list = deque(maxlen=100)
        else:
            tvec = np.array([0, 0, self.charuco_z])
            rvec = np.array([0, 1, 0]) * np.radians(180)
            cam_T_center = np.hstack([tvec, rvec])
            self.charuco.register_pose(tvec, rvec, "home_cam", "center")
        print("Set the target object and press a key.")
        print("    [s] start data collection.")
        print("    [q] quit data collection.")
        cv2.namedWindow("image")
        cv2.moveWindow("image", 200, 200)
        while True:
            if self.use_charuco:
                cam_T_center, corners, charucoCorners, color_image, depth_image = self.charuco.get_board_pose()
                if np.any(cam_T_center != None):
                    self.charuco.register_pose(cam_T_center[:3], cam_T_center[3:], "home_cam", "center")
                    depth_list.append(depth_image)
                    charuco_list.append(cam_T_center)
                color = color_image
            else:
                color_images, _, _, _ = self.charuco.cam.get_image(crop=False, get_mask=False)
                color = color_images[0]
            height = color.shape[0]
            width = color.shape[1]
            cv2.line(color, (int(width/2), 0), (int(width/2), height), (255, 0, 0), 1)
            cv2.line(color, (0, int(height/2)), (width, int(height/2)), (255, 0, 0), 1)
            if np.any(cam_T_center != None):
                self.charuco.draw_coord(color, cam_T_center[:3], cam_T_center[3:], size=4)
            multiple = 2
            color = cv2.resize(color, (int(width * multiple), int(height * multiple)))
            cv2.imshow("image", color[::-1, ::-1])
            key = cv2.waitKey(1)
            if key & 0xFF == ord("s"):
                if self.use_charuco:
                    self.calculate_depth_to_dist(charuco_list, depth_list, charucoCorners)
                if np.any(cam_T_center != None):
                    break
                else:
                    print("ChAruco pose is not detected. Please adjust ChAruco board position.")
            elif key & 0xFF == ord("q"):
                if self.use_charuco:
                    self.calculate_depth_to_dist(charuco_list, depth_list, charucoCorners)
                cv2.destroyAllWindows()
                self.close()
        cv2.destroyAllWindows()
        charuco_pose = self.charuco.get_pose("base", "center")
        # print("base_T_charuco:", charuco_pose)
        # print("base_T_cam:", self.charuco.get_pose("base", "home_cam"))
        # print("cam_T_charuco:", self.charuco.get_pose("home_cam", "center"))
        return charuco_pose

    def calculate_depth_to_dist(self, charuco_list, depth_list, charucoCorners):
        # RealSenseではARマーカで測定した距離とDepthセンサで測定した距離が一致しないことがある。
        # あらかじめChArucoボードの中心点までの距離を測っておき補正する。
        charuco_mean = np.array(charuco_list).mean(axis=0)
        charuco_trans = charuco_mean[:3]
        charuco_rot = rotvec_to_euler(charuco_mean[3:], seq="XYZ", degrees=True)
        charuco_dist = np.linalg.norm(charuco_mean[:3])
        print("ChAruco pose:", np.round(charuco_trans, 3), "[m],", np.round(charuco_rot, 1), "[deg]")
        depth_mean = np.array(depth_list).mean(axis=0)
        height, width = depth_mean.shape[0], depth_mean.shape[1]
        depth_mean_center = depth_mean[int(height/2)-5:int(height/2)+5, int(width/2)-5:int(width/2)+5].mean()
        depth_to_dist = charuco_dist - depth_mean_center
        print("Depth at center", np.round(depth_mean_center, 3))
        print("charuco_z:", np.round(charuco_trans[2], 3))
        print("depth_to_dist:", np.round(depth_to_dist, 3))
        # Fit and show charuco board tilting.
        x, y, z = [], [], []
        for corner in charucoCorners:
            corner_x = int(corner[0, 0])
            corner_y = int(corner[0, 1])
            corner_z = depth_mean[corner_y, corner_x]
            if np.abs(depth_mean_center - corner_z) < 0.01:
                x.append(corner_x)
                y.append(corner_y)
                z.append(corner_z)
        # Perform curve fitting 
        def func(xy, a, b, c):
            x, y = xy 
            return a + b * x + c * y
        popt, pcov = curve_fit(func, (x, y), z)
        a, b, c = popt
        print("Fitted curve: {:.3f} {:+.6f}*x {:+.6f}*y".format(a, b, c))
        # Create 3D plot of the data points and the fitted curve.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, color="blue")
        x_range = np.linspace(0, width-1, width)
        y_range = np.linspace(0, height-1, height)
        X, Y = np.meshgrid(x_range, y_range)
        Z = func((X, Y), *popt)
        print("ChAruco board max height:", np.round(Z.max() - depth_mean_center, 3))
        print("ChAruco board min height:", np.round(Z.min() - depth_mean_center, 3))
        ax.plot_surface(X, Y, Z, color="red", alpha=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.savefig("charuco_board_tilting.png")
        plt.clf()
        plt.cla()

    def get_equidistribution_points(self):
        # https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
        self.theta, self.phi = [], []
        self.N_count = 0
        a = 4 * np.pi * self.max_r ** 2 / self.N
        d = np.sqrt(a)
        M_theta = int(np.pi / d)
        d_theta = np.pi / M_theta
        d_phi = a / d_theta
        points = []
        for m in range(M_theta):
            theta = np.pi * (m + 0.5) / M_theta
            theta_original = theta
            if theta > self.max_theta:
                break
            M_phi = int(2 * np.pi * np.sin(theta) / d_phi)
            for n in range(M_phi):
                phi = 2 * np.pi * n / M_phi
                if self.random:
                    theta = theta_original + np.random.rand() * self.rot_temperature * np.pi / M_theta
                    phi += 2 * np.pi * np.random.rand() * self.rot_temperature / M_phi
                if theta < self.max_theta:
                    r = np.random.uniform(self.min_r, self.max_r)
                    points.append(self.spherical_to_Cartesian(theta, phi, r))
                    self.theta.append(theta)
                    self.phi.append(phi)
                    self.N_count += 1
        print("N:", self.N, "N_count:", self.N_count)
        return np.array(points)

    def spherical_to_Cartesian(self, theta, phi, r):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    def generate_shooting_poses(self, points):
        tvecs, rvecs = [], []
        for p in points:
            LookAtRot = self.LookAt(eye=p, target=np.zeros(3), up=self.up)
            tvecs.append(p)
            rvecs.append(LookAtRot.as_rotvec())
        return tvecs, rvecs

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

    def get_base_T_eye(self):
        points = self.get_equidistribution_points()
        tvecs, rvecs = self.generate_shooting_poses(points)
        base_T_eye = []
        for tvec, rvec in zip(tvecs, rvecs):
            tvec[2] += self.z_escape
            self.charuco.register_pose(tvec, rvec, "center", "eye")
            base_T_eye.append(self.charuco.get_pose("base", "eye"))
        self.charuco.remove_pose("center", "eye")
        return base_T_eye

    def visualize_poses(self, poses):
        charuco = pytr.transform_from(
            pyrot.matrix_from_compact_axis_angle(self.charuco_pose[3:]), self.charuco_pose[:3])
        ax = pytr.plot_transform(A2B=charuco, s=0.05, name="charuco")
        for i, p in enumerate(poses):
            eye = pytr.transform_from(
                pyrot.matrix_from_compact_axis_angle(p[3:]), p[:3])
            pytr.plot_transform(ax, A2B=eye, s=0.02)
        ax.view_init(elev=30, azim=120)
        ax.set_xlim(self.charuco_pose[0] - self.max_r, self.charuco_pose[0] + self.max_r)
        ax.set_ylim(self.charuco_pose[1] - self.max_r, self.charuco_pose[1] + self.max_r)
        ax.set_zlim(self.charuco_pose[2] - self.max_r + self.z_escape, self.charuco_pose[2] + self.max_r + self.z_escape)
        plt.savefig("shooting_poses.png")
        plt.clf()
        plt.cla()

    def visualize_coordinates(self, origin="base"):
        ax = self.visualize_coord_one(origin, "center")
        ax = self.visualize_coord_one(origin, "cam", ax)
        ax = self.visualize_coord_one(origin, "base", ax)
        ax.view_init(elev=30, azim=120)
        L = 0.8
        ax.set_xlim(0, L)
        ax.set_ylim(-L/2, L/2)
        ax.set_zlim(0, L)
        plt.savefig("{}_coordinates.png".format(origin))
        plt.clf()
        plt.cla()

    def visualize_coord_one(self, source, target, ax=None):
        pose = self.charuco.get_pose(source, target)
        # print("{}_T_{}".format(source, target), pose[:3] * 1000, np.degrees(pose[3:]))
        transform = pytr.transform_from(
            pyrot.matrix_from_compact_axis_angle(pose[3:]), pose[:3])
        if ax is None:
            return pytr.plot_transform(A2B=transform, s=0.1, name=target)
        else:
            return pytr.plot_transform(ax, A2B=transform, s=0.1, name=target)

    def move_and_shoot(self, poses):
        assert self.use_robot, "Set --use_robot argument if you want to control a robot."
        self.save_dir = self.charuco.make_save_directory()
        self.charuco.cam.save_camera_param(self.save_dir)
        # For hand-eye calibration
        c_R_t, c_t_t, b_R_g, b_t_g = [], [], [], []
        cv2.namedWindow("image")
        cv2.moveWindow("image", 200, 200)
        for i, p in enumerate(poses):
            # Go to next pose
            trans = p[:3] * 1000
            euler = rotvec_to_euler(p[3:], seq="XYZ", degrees=True)
            print("Go to next pose({}/{}): {}, {}, theta={}, phi={}".format(
                i, self.N_count, trans.astype("int32"), euler.astype("int32"),
                np.degrees(self.theta[i]).astype("int32"), np.degrees(self.phi[i]).astype("int32")))
            self.robot.moveJ_IK2(p, vel=0.3, acc=1.0)
            if self.auto_data_collection:
                time.sleep(1)
            else:
                print("    [s] save this data.")
                print("    [q] quit data collection.")
            while True:
                if self.use_charuco:
                    cam_T_center, _, _, color_image, depth_image = self.charuco.get_board_pose(draw_corners=False)
                    if not np.any(cam_T_center != None):
                        raise ValueError
                else:
                    color_images, depth_images, _, _ = self.charuco.cam.get_image(crop=False, get_mask=False)
                    color_image, depth_image = color_images[0], depth_images[0]
                # Get camera pose
                cam_in_ob, ob_in_cam = self.get_camera_pose()
                self.visualize_coordinates(origin="base")
                # Show image
                color = color_image.copy()
                self.charuco.draw_coord(color, ob_in_cam[:3], ob_in_cam[3:], size=4)
                multiple = 2
                height = color.shape[0]
                width = color.shape[1]
                color = cv2.resize(color, (int(width * multiple), int(height * multiple)))
                cv2.imshow("image", color[::-1, ::-1])
                key = cv2.waitKey(1)
                if key & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    self.close()
                elif key & 0xFF == ord("s") or self.auto_data_collection:
                    depth_image += self.depth_to_dist
                    self.charuco.save_data(color_image, depth_image, cam_in_ob, self.save_dir, i)
                    c_R_t.append(ob_in_cam[3:])
                    c_t_t.append(ob_in_cam[:3])
                    ee_in_base = self.robot.get_ee_pose()
                    b_R_g.append(ee_in_base[3:])
                    b_t_g.append(ee_in_base[:3])
                    break
        cv2.destroyAllWindows()
        return np.array(c_R_t), np.array(c_t_t), np.array(b_R_g), np.array(b_t_g)

    def hand_eye_calibration(self, c_R_t, c_t_t, b_R_g, b_t_g):
        assert self.use_charuco, "Set --use_charuco argument for hand-eye calibration."
        initial_g_R_c = np.array(self.tcp[3:])
        initial_g_t_c = np.array(self.tcp[:3])
        g_R_c, g_t_c = cv2.calibrateHandEye(
            R_gripper2base=b_R_g,
            t_gripper2base=b_t_g,
            R_target2cam=c_R_t,
            t_target2cam=c_t_t,
            R_cam2gripper=initial_g_R_c,
            t_cam2gripper=initial_g_t_c,
            method=cv2.CALIB_HAND_EYE_TSAI)
        g_R_c_rotvec = R.from_matrix(g_R_c).as_rotvec()
        np.savetxt(os.path.join(self.save_dir, "g_t_c_ur5e.txt"), g_t_c.ravel())
        np.savetxt(os.path.join(self.save_dir, "g_R_c_ur5e.txt"), g_R_c_rotvec)
        print("g_t_c [mm]:", np.round(g_t_c.ravel() * 1000, 1))
        print("g_R_c [deg]:", np.round(np.degrees(g_R_c_rotvec), 1))
        print("tcp:", np.round(np.hstack([g_t_c.ravel(), g_R_c_rotvec]), 3))

    def get_camera_pose(self):
        if self.use_charuco:
            cam_in_ob = self.charuco.get_matrix("center", "cam")  # Transform matrix
            ob_in_cam = self.charuco.get_pose("cam", "center")  # tvec + rvec
        else:
            base_T_eye = self.robot.get_pose()
            self.charuco.register_pose(base_T_eye[:3], base_T_eye[3:], "base", "cam")
            cam_in_ob = self.charuco.get_matrix("center", "cam")  # Transform matrix
            ob_in_cam = self.charuco.get_pose("cam", "center")  # tvec + rvec
        return cam_in_ob, ob_in_cam

    def close(self):
        if self.use_robot:
            self.robot.moveJ_IK2(self.home_pose, vel=0.1, acc=1.0)
            self.robot.stop()
        self.charuco.close()
        print("Finish data collection.")
        os._exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=str, default="usb_cam")
    parser.add_argument("--N", "-n", type=int, default=10)
    parser.add_argument("--min_r", "-mir", type=float, default=0.20)
    parser.add_argument("--max_r", "-mar", type=float, default=0.20)
    parser.add_argument("--max_theta", "-mat", type=float, default=70)
    parser.add_argument("--z_escape", "-z", type=float, default=0.1)
    parser.add_argument("--rot_temperature", "-rt", type=float, default=0.5)
    parser.add_argument("--random", "-ra", action="store_true")
    parser.add_argument("--use_robot", "-ur", action="store_true")
    parser.add_argument("--use_charuco", "-uc", action="store_true")
    parser.add_argument("--auto_data_collection", "-a", action="store_true")
    args = parser.parse_args()
    kwargs = vars(args)

    kwargs["robot_ip"] = "192.168.56.2"
    if args.device == "D405":
        kwargs["home_pose"] = [0.55, 0.0, 0.3, 2.22, -2.22, 0.0]
        kwargs["tcp"] = [-0.01, -0.1, -0.01, 0.0, 0.0, 0.0]
        kwargs["up"] = [0, 0, 1]
        kwargs["charuco_z"] = 0.311
        kwargs["depth_to_dist"] = 0.03
    elif args.device == "D435":
        kwargs["home_pose"] = [0.55, 0.0, 0.4, 2.22, -2.22, 0.0]
        kwargs["tcp"] = [-0.032, -0.045, 0.025, 0.038, 0.013, 0.018]
        kwargs["up"] = [0, 0, 1]
        kwargs["charuco_z"] = 0.417
        kwargs["depth_to_dist"] = 0.003
    elif args.device == "usb_cam":
        kwargs["home_pose"] = [0.55, 0.0, 0.4, 2.22, -2.22, 0.0]
        kwargs["tcp"] = [-0.032, -0.045, 0.025, 0.038, 0.013, 0.018]
        kwargs["up"] = [0, 0, 1]
        kwargs["charuco_z"] = 0.417
        kwargs["depth_to_dist"] = 0.003

    cc = camera_caribration(**kwargs)
    if args.device == "D405" or args.device == "D435":
        cc.charuco.set_camera_param()
        base_T_eye = cc.get_base_T_eye()
        cc.visualize_poses(base_T_eye)
        hand_eye_param = cc.move_and_shoot(base_T_eye)
        if args.use_charuco:
            cc.hand_eye_calibration(*hand_eye_param)
        cc.close()
    elif args.device == "usb_cam":
        pass
    elif args.device == "flir":
        pass
    else:
        raise NotImplementedError
    
    

""" 参考URL
cv2.calibrateHandEye()
https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
カメラの外部パラメータ行列を最小二乗法で推定する
https://dev.classmethod.jp/articles/estimate-camera-external-parameter-matrix-with-least-squares/
cv2.calibrateCamera()関数の解説
https://kamino.hatenablog.com/entry/opencv_calibrate_camera
座標系の定義
https://stackoverflow.com/questions/67072289/eye-in-hand-calibration-opencv
LookAt関数の実装例
https://github.com/stemkoski/three.py/blob/master/three.py/mathutils/MatrixFactory.py#L104
"""