import sys
import msvcrt
# import tty
# import select
import cv2
import numpy as np

from scipy.spatial.transform import Rotation

timeout = 0.1
pre_key = None

def getch(): 
    if msvcrt.kbhit():
        return msvcrt.getwch()
    else:
        return ""

def transform(re, rb):
    re = np.array(re)
    rb = np.array(rb)
    rot_re = Rotation.from_rotvec(np.array(re[3:6]))
    R_re = rot_re.as_matrix()
    T_re = np.r_[np.c_[R_re, np.array([re[:3]]).T], np.array([[0, 0, 0, 1]]) ]
    T_er = np.r_[np.c_[R_re.T, -np.dot(R_re.T, np.array([re[:3]]).T)], np.array([[0, 0, 0, 1]]) ]
    
    rot_rb = Rotation.from_rotvec(np.array(rb[3:6]))
    R_rb = rot_rb.as_matrix()
    T_rb = np.r_[np.c_[R_rb, np.array([rb[:3]]).T], np.array([[0, 0, 0, 1]]) ]
      
    T_eb = np.dot(T_er, T_rb)
    # print(T_eb)
    pose = T_eb[:-1,-1]
    R_eb = T_eb[:3,:3]
    rot = Rotation.from_matrix(R_eb)
    rpy = rot.as_rotvec()
    return np.r_[pose, rpy] 

def reverse_transform(re, eb):
    re = np.array(re)
    eb = np.array(eb)
    
    rot_re = Rotation.from_rotvec(np.array(re[3:6]))
    R_re = rot_re.as_matrix()
    T_re = np.r_[np.c_[R_re, np.array([re[:3]]).T], np.array([[0, 0, 0, 1]]) ]
    
    rot_eb = Rotation.from_rotvec(np.array(eb[3:6]))
    R_eb = rot_eb.as_matrix()
    T_eb = np.r_[np.c_[R_eb, np.array([eb[:3]]).T], np.array([[0, 0, 0, 1]]) ]
    
    T_rb = np.dot(T_re, T_eb)    
    position_world = T_rb[:-1,-1]
    R_rb = T_rb[:3,:3]
    rot = Rotation.from_matrix(R_rb)
    rpy_world = rot.as_rotvec()
    
    pose_from_world = np.r_[position_world, rpy_world]
    
    return pose_from_world

def quat2rotvec(pose_quat):
    pose = pose_quat[:3]
    quat = pose_quat[3:]
    
    assert len(quat) == 4, "len(quat) must be 4" 

    rot = Rotation.from_quat(quat)
    rotvec = rot.as_rotvec()
    pose_rotvec = np.r_[pose, rotvec]

    return pose_rotvec

def rotvec2quat(pose_rotvec):
    pose = pose_rotvec[:3]
    rotvec = pose_rotvec[3:]
    
    assert len(rotvec) == 3, "len(rotvec) must be 3" 

    rot = Rotation.from_rotvvec(rotvec)
    quat = rot.as_quat()
    quat = [quat[3], quat[0], quat[1], quat[2]]    # w, x, y, z
    pose_quat = np.r_[pose, quat]
    return pose_quat

def rotate(pose, angles, order="xyz"):
    rot = Rotation.from_euler(order, angles)
    pose_rotvec = Rotation.from_rotvec(pose[3:])

    result_rot = rot * pose_rotvec
    result_rotvec = result_rot.as_rotvec()
    
    result_pose = np.r_[pose[:3], result_rotvec]

    return result_pose

def save_img(img, name="img"):    
    # Image.fromarray(img).save(name+'.png')
    cv2.imwrite(name+'.png', img)

def bgr_extraction(img, bgrLower, bgrUpper, extract_mode="inclusive", white_mask=False):
    img_mask = cv2.inRange(img, bgrLower, bgrUpper)
    if extract_mode == "inclusive":
        result = cv2.bitwise_and(img, img, mask=img_mask)
    elif extract_mode == "exclusive":
        img_mask_not = cv2.bitwise_not(img_mask)
        result = cv2.bitwise_and(img, img, mask=img_mask_not)
    if white_mask:
        result[np.all(result==[0,0,0], axis=2)] = [255,255,255]
        
    return result

def hsv_extraction(img, hsvLower, hsvUpper, extract_mode="inclusive", white_mask=False):
    img_mask = cv2.inRange(img, hsvLower, hsvUpper)
    if extract_mode == "inclusive":
        result = cv2.bitwise_and(img, img, mask=img_mask)
    elif extract_mode == "exclusive":
        img_mask_not = cv2.bitwise_not(img_mask)
        result = cv2.bitwise_and(img, img, mask=img_mask_not)
    if white_mask:
        result[np.all(result==[0,0,0], axis=2)] = [255,255,255]
        
    return result

def wait_press_key():
    print("press y to continue")
    while True:
        key = msvcrt.getch()
        if key == "y":
            break
        elif key == "q":
            quit()