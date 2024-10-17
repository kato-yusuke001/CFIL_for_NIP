import json
import numpy as np
import cv2
import time
from scipy.spatial.transform import Rotation as R

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

    def force_calib(self):
        marker_04 = self.b_t_t["04"]
        marker_09 = self.b_t_t["09"]

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

        np.save("ratio.npy", ratio)

        # check
        diff_x = corners[ids.ravel().tolist().index(0)][0].mean(axis=0)[0]-corners[ids.ravel().tolist().index(center_id)][0].mean(axis=0)[0]
        diff_y = corners[ids.ravel().tolist().index(0)][0].mean(axis=0)[1]-corners[ids.ravel().tolist().index(center_id)][0].mean(axis=0)[1]
        X = diff_x*ratio[0] + self.b_t_t["{:02}".format(center_id)][0]
        Y = diff_y*ratio[1] + self.b_t_t["{:02}".format(center_id)][1]
        print("recall X:{}, Y:{}".format(X, Y))
        print("actual X:{}, Y:{}".format(self.b_t_t["00"][0], self.b_t_t["00"][1]))

        return ratio



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

    # def readARMarker(self):

    #     c_t_t, c_R_t = {}, {}
    #     raw_c_t_t = {}
    #     s_time = time.time()
    #     while time.time()-s_time<10:
    #         color_images, depth_images, _, _, frames = self.cam.get_image(crop=False, get_mask=False)
    #         color_image = color_images[self.camera_id]
    #         self.color_image_original = color_image.copy()
    #         depth_image = depth_images[self.camera_id]
    #         self.depth_image_original = depth_image.copy()

    #         depth_frame = frames.get_depth_frame()

    #         gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    #         # Detect markers
    #         corners, ids, rejectedImgPoints = self.aruco.detectMarkers(
    #             gray_image, self.aruco_dict, parameters=self.arucoParams)
            
            
    #         for i,id in enumerate(ids):
    #             point = np.average(corners[i][0], axis=0)
    #             depth = depth_frame.get_distance(point[0], point[1])
    #             point = np.append(point,depth)
    #             if depth!=0:
    #                 x=point[0]
    #                 y=point[1]
    #                 z=point[2]
    #                 x,y,z=self.cam.get_point_from_pixel(x, y, z)
    #                 if "{:02}".format(id[0]) in raw_c_t_t.keys():
    #                     raw_c_t_t["{:02}".format(id[0])].append([x, y, z])
    #                 else:
    #                     raw_c_t_t["{:02}".format(id[0])] = [[x, y, z]]
                    
    #     for key in raw_c_t_t.keys():
    #         c_t_t[key] = np.average(raw_c_t_t[key], axis=0)
    #         c_R_t[key] = [0, 0, 0]

    #     print(c_t_t)        
    
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
            if ids is not None and len(ids) ==10:
                b_R_t, b_t_t = [], []
                for i in range( ids.size ):
                    b_t_t.append(self.b_t_t["{:02}".format(ids[i][0])])
                    b_R_t.append(R.from_euler("XYZ", [0, 0, 180], degrees=True).as_rotvec())
                    cv2.drawFrameAxes(color_image, self.cameraMatrix, self.distCoeffs, c_R_t[i], c_t_t[i], 0.1)

            cv2.imshow('org', color_image)

            # Escキーで終了
            key = cv2.waitKey(1)
            if key == 27: # ESC
                break

        return np.squeeze(np.array(c_R_t)), np.squeeze(np.array(c_t_t)), np.array(b_R_t), np.array(b_t_t)

    def calibration(self):
        c_R_t, c_t_t, b_R_t, b_t_t = self.readARMarker()
        print(c_R_t.shape, c_t_t.shape, b_R_t.shape, b_t_t.shape)
        g_R_c, g_t_c = cv2.calibrateHandEye(
            R_gripper2base=b_R_t,
            t_gripper2base=b_t_t,
            R_target2cam=c_R_t,
            t_target2cam=c_t_t,
            method=cv2.CALIB_HAND_EYE_PARK)

        print("g_R_c:", g_R_c)
        print("g_t_c:", g_t_c)

        np.save("camera_pose.npy",np.hstack([g_t_c.ravel(), g_R_c.ravel()]))
        return g_R_c, g_t_c

if __name__ == "__main__":
    calib = CalibCeilingCamera()
    # calib.calibration()
    calib.force_calib()
    calib.cam.close()

   