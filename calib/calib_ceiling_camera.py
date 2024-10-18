import json
import numpy as np
import cv2
import time
from scipy.spatial.transform import Rotation as R

import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
from pytransform3d.transform_manager import TransformManager

from rs_utils import RealSense

class CalibCeilingCamera:
    def __init__(self):
        self.aruco = cv2.aruco
        self.aruco_dict = self.aruco.getPredefinedDictionary(self.aruco.DICT_6X6_250)        
        self.arucoParams = self.aruco.DetectorParameters()

        self.marker_length = 0.04

        json_path = "./marker.json"
        with open(json_path) as f:
            self.b_t_t = json.load(f) 
        self.preprocess()

        self.camera_id = 0
        # cam_settings = {"width": 640, "height": 480, "fps": 30, "clipping_distance": 1.0}
        self.cam = RealSense()
        self.cameraMatrix = self.cam.cameraMatrix_woCrop[self.camera_id]
        self.distCoeffs = self.cam.distCoeffs[self.camera_id]

    def preprocess(self):
        marker_04 = self.b_t_t["04"]
        marker_09 = self.b_t_t["09"]
        self.b_t_t["00"] = [marker_04[0]+0.05, marker_04[1]-0.05, marker_04[2]]
        self.b_t_t["01"] = [marker_04[0]-0.05, marker_04[1]-0.05, marker_04[2]]
        self.b_t_t["02"] = [marker_04[0]+0.05, marker_04[1]+0.05, marker_04[2]]
        self.b_t_t["03"] = [marker_04[0]-0.05, marker_04[1]+0.05, marker_04[2]]

        self.b_t_t["05"] = [marker_09[0]+0.05, marker_09[1]-0.05, marker_09[2]]
        self.b_t_t["06"] = [marker_09[0]-0.05, marker_09[1]-0.05, marker_09[2]]
        self.b_t_t["07"] = [marker_09[0]+0.05, marker_09[1]+0.05, marker_09[2]]
        self.b_t_t["08"] = [marker_09[0]-0.05, marker_09[1]+0.05, marker_09[2]]    
    
    def readARMarker(self):

        c_R_t, c_t_t, b_R_t, b_t_t = [], [], [], []
        s_time = time.time()
        while time.time()-s_time<5:
            color_images, depth_images, _, _, _ = self.cam.get_image(crop=False, get_mask=False)
            color_image = color_images[self.camera_id]
            self.color_image_original = color_image.copy()
            depth_image = depth_images[self.camera_id]
            self.depth_image_original = depth_image.copy()

            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            # Detect markers
            corners, ids, rejectedImgPoints = self.aruco.detectMarkers(
                gray_image, self.aruco_dict, parameters=self.arucoParams)
            
            c_R_t, c_t_t, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.cameraMatrix, self.distCoeffs)
            if ids is not None and len(ids) ==5:
                b_R_t, b_t_t = [], []
                for i, id in enumerate(ids):
                    # print(id)
                    b_t_t.append(self.b_t_t["{:02}".format(id[0])])
                    b_R_t.append(R.from_euler("XYZ", [0, 0, 180], degrees=True).as_rotvec())
                    cv2.drawFrameAxes(color_image, self.cameraMatrix, self.distCoeffs, c_R_t[i], c_t_t[i], 0.1)

            cv2.imshow('org', color_image)

            # Escキーで終了
            key = cv2.waitKey(1)
            if key == 27: # ESC
                break

        return np.squeeze(np.array(c_R_t)), np.squeeze(np.array(c_t_t)), np.array(b_R_t), np.array(b_t_t)

    def force_calib(self):

        color_images, depth_images, _, _, frames = self.cam.get_image(crop=False, get_mask=False)
        color_image = color_images[self.camera_id]
        self.color_image_original = color_image.copy()
        depth_image = depth_images[self.camera_id]
        self.depth_image_original = depth_image.copy()

        # depth_frame = frames.get_depth_frame()

        s_time = time.time()
        while time.time()-s_time<5:
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            # Detect markers
            corners, ids, rejectedImgPoints = self.aruco.detectMarkers(
                gray_image, self.aruco_dict, parameters=self.arucoParams)
            
            if ids is not None and len(ids) ==5:
                break


        ratio = []
        center_id = 4
        for i,c in zip(ids.ravel(), corners):
            if i == center_id:
                continue
            diff_X = abs(self.b_t_t["{:02}".format(i)][0] - self.b_t_t["{:02}".format(center_id)][0])
            diff_Y = abs(self.b_t_t["{:02}".format(i)][1] - self.b_t_t["{:02}".format(center_id)][1])
            diff_x = abs(c[0].mean(axis=0)[0] - corners[ids.ravel().tolist().index(center_id)][0].mean(axis=0)[0])
            diff_y = abs(c[0].mean(axis=0)[1] - corners[ids.ravel().tolist().index(center_id)][0].mean(axis=0)[1])

            ratio.append([diff_X/diff_x, diff_Y/diff_y])
        ratio = np.average(ratio, axis=0)
        print("ratio:", ratio)
        np.save("ratio.npy", ratio)

        # check
        target_id = 3
        diff_x = corners[ids.ravel().tolist().index(target_id)][0].mean(axis=0)[0]-corners[ids.ravel().tolist().index(center_id)][0].mean(axis=0)[0]
        diff_y = corners[ids.ravel().tolist().index(target_id)][0].mean(axis=0)[1]-corners[ids.ravel().tolist().index(center_id)][0].mean(axis=0)[1]
        X = self.b_t_t["{:02}".format(center_id)][0] + diff_x*ratio[0] 
        Y = self.b_t_t["{:02}".format(center_id)][1] - diff_y*ratio[1]
        print("recall X:{}, Y:{}".format(X, Y))
        print("actual X:{}, Y:{}".format(self.b_t_t["{:02}".format(target_id)][0], self.b_t_t["{:02}".format(target_id)][1]))

        return ratio

    def calibration(self):
        self.tm = TransformManager()

        c_R_t, c_t_t, b_R_t, b_t_t = self.readARMarker()
        # print(c_R_t, c_t_t, b_R_t, b_t_t)
        self.register_pose(c_t_t[0], c_R_t[0], "cam", "target")
        self.register_pose(b_t_t[0], b_R_t[0], "base", "target")
        cam_in_base = self.get_pose("base", "cam")
        print("camera_pose", cam_in_base)
        np.save("camera_pose.npy",cam_in_base)
        return cam_in_base
    
    def register_pose(self, tvec, rvec, source, target):
        source_T_target = pytr.transform_from(
            pyrot.matrix_from_compact_axis_angle(rvec), tvec)
        self.tm.add_transform(target, source, source_T_target)

    def remove_pose(self, source, target):
        self.tm.remove_transform(target, source)

    def get_pose(self, source, target):
        return self.transform_matrix_to_vector(self.tm.get_transform(target, source))
    
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

if __name__ == "__main__":
    calib = CalibCeilingCamera()
    calib.calibration()
    calib.force_calib()
    calib.cam.close()

   