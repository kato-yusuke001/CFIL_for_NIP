# coding: utf-8
import pyrealsense2 as rs
import numpy as np
import cv2

import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
from pytransform3d.transform_manager import TransformManager

from scipy.spatial.transform import Rotation as R

def register_pose(tvec, rvec, source, target):
        source_T_target = pytr.transform_from(
            pyrot.matrix_from_compact_axis_angle(rvec), tvec)
        tm.add_transform(target, source, source_T_target)

def remove_pose(source, target):
    tm.remove_transform(target, source)

def get_pose(source, target):
    return transform_matrix_to_vector(tm.get_transform(target, source))

def transform_matrix_to_vector(transform):
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


aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Start streaming")
pipeline.start(config)

cv2.namedWindow('RealsenseImage', cv2.WINDOW_AUTOSIZE)

while True:

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, dictionary)

    if len(corners)!=0:
        id = ids[0]
        point = np.average(corners[0][0], axis=0)
        depth = depth_frame.get_distance(point[0], point[1])
        point = np.append(point,depth)
        if depth!=0:
            x=point[0]
            y=point[1]
            z=point[2]
            x,y,z=rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], z)
            print("point:",x,y,z)
  
    aruco.drawDetectedMarkers(color_image, corners, ids, (0,255,0))

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((color_image, depth_colormap))

    cv2.imshow("RealsenseImage",images)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

b_t_t = {}
marker_04 = b_t_t["04"] = [-0.180, -0.350, -0.015]
b_t_t["00"] = [marker_04[0]+0.05, marker_04[1]-0.05, marker_04[2]]
b_t_t["01"] = [marker_04[0]-0.05, marker_04[1]-0.05, marker_04[2]]
b_t_t["02"] = [marker_04[0]+0.05, marker_04[1]+0.05, marker_04[2]]
b_t_t["03"] = [marker_04[0]-0.05, marker_04[1]+0.05, marker_04[2]]

print("{:02}".format(id[0]))
tm = TransformManager()
register_pose([x,y,z], R.from_euler("XYZ", [0, 180, 0], degrees=True).as_rotvec(), "cam", "target")
register_pose(b_t_t["{:02}".format(id[0])], R.from_euler("XYZ", [0, 0, 180], degrees=True).as_rotvec(), "base", "target")

cam_in_base = get_pose("base", "cam")
print(cam_in_base)
np.save("camera_pose.npy",cam_in_base)
