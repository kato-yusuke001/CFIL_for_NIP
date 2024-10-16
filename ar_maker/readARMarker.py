# coding: utf-8
import pyrealsense2 as rs
import numpy as np
import cv2

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