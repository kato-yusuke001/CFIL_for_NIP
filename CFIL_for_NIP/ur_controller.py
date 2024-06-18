from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
import os
import sys
import numpy as np
import time
import argparse
import json
import cv2
import copy


from scipy.spatial.transform import Rotation

import utils
import cv2


sys.path.append('../')
from vacuum_gripper import VacuumGripper

class URController():
    def __init__(self, config_file="configs/robot_config.json"):
        # self.robot_ip = "localhost"

        self.home_joint_angles = np.deg2rad([-90., -90, -90, -90, 90, 0.])
        self.bottleneck_pose = None
        self.vel = 0.5
        self.acc = 0.5
        self.blend = 0.0025
        
        self.load_config_file(config_file)

        self.cam = None
        
        self.rtde_c = RTDEControlInterface(self.robot_ip)
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)
        self.io = RTDEIOInterface(self.robot_ip)

        self.set_tcp(self.tcp)
        
        self.feature = self.rtde_c.FEATURE_BASE
        
        # activate gripper
        self.gripper = None
        self.gripper_activate()
        
        self.mode = "init"

        self.manager = None

        self.camera_setup()
        
        self.initialize()


    def __del__(self):
        if self.cam is not None:
            self.cam.release()
        
    def initialize(self):
        print("move to home")
        self.moveJ(self.home_joint_angles)
        self.home_pose = self.get_pose()
        self.mode = "init"
        
    def gripper_activate(self):
        self.gripper = VacuumGripper(_io=self.io, _rtde_r=self.rtde_r)

    def camera_setup(self):
        if self.use_camera == "spin":
            print("use camera: spin")
            from camera_modules import SpinCamera
            self.cam = SpinCamera()
        elif self.use_camera == "cv2":
            print("use camera: cv2")
            self.cam = cv2.VideoCapture(0)
        else:
            self.cam = None
        
    def load_config_file(self, config_file):
        json_file = open(config_file, "r")
        json_dict = json.load(json_file)

        self.robot_ip = json_dict["robot_ip"]    
        self.home_joint_angles = np.deg2rad(json_dict["home_joint_angles"])
    
        self.tcp = json_dict["tcp"]

        self.clopping = json_dict["image_clopping"]
        self.clopping_w = json_dict["clopping_info"]["width"]
        self.clopping_h = json_dict["clopping_info"]["height"]
        self.use_camera = json_dict["camera"]

    def get_img(self):
        ret, img = self.cam.read()
        img = cv2.resize(img, (640, 480))
        if self.clopping:
            img = img[self.clopping_h[0]:self.clopping_h[1], self.clopping_w[0]:self.clopping_w[1]]
        return img  
    
    def stop(self):
        self.rtde_c.stopL(10)
        
    def set_tcp(self, point):
        self.rtde_c.setTcp(point)
        
    def reset_start(self, pose=None):
        if pose is None:
            pose = self.home_pose
        self.moveJ_IK(pose, speed=3, acceleration=2.8)
        img = self.get_img()
        
        return img, self.get_pose()

    def moveL(self, pose, speed=0.20,  acceleration=1.2, _asynchronous=False):
        args = {"speed":speed, "acceleration":acceleration, "asynchronous":_asynchronous}
        self.rtde_c.moveL(pose, **args)
    
    def moveJ(self, joints, speed=1.05,  acceleration=1.4, _asynchronous=False):
        args = {"speed":speed, "acceleration":acceleration, "asynchronous":_asynchronous}
        self.rtde_c.moveJ(joints, **args)

    def moveJ_IK(self, pose, speed=1.05,  acceleration=1.4, _asynchronous=False):
        args = {"speed":speed, "acceleration":acceleration, "asynchronous":_asynchronous}
        self.rtde_c.moveJ_IK(pose, **args)

        
    def check_arrival(self, target_pose):
        target_position = np.array(target_pose[:3])
        target_rotvec = np.array(target_pose[3:])

        current_pose = self.get_pose()
        current_position = np.array(current_pose[:3])
        current_rotvec = np.array(current_pose[3:])
        dist = np.linalg.norm(target_position-current_position)
        if(dist < 0.001):
            return True
        else:
            return False

    # rotation vector
    def get_pose(self):
        return np.array(self.rtde_r.getActualTCPPose())
    
    # quaternion
    def get_poseQ(self):
        pose = self.rtde_r.getActualTCPPose()
        r1 = Rotation.from_rotvec(pose[3:])
        quat = r1.as_quat()
        quat = [quat[3], quat[0], quat[1], quat[2]]    # w, x, y, z
        return np.append(pose[:3], quat)
    
    def teaching_mode(self, switch):
        if switch:
            self.rtde_c.teachMode()
        else:
            self.rtde_c.endTeachMode()
    
    def suction_on(self):
        if self.gripper is None:
            print("GRIPPER IS NOT ACTIVATE")
            return
        self.gripper.grip_on()
        # time.sleep(1.0)
    
    def suction_off(self):
        if self.gripper is None:
            print("GRIPPER IS NOT ACTIVATE")
            return
        self.gripper.grip_off()        
        time.sleep(1.0)

    def is_suction(self):
        pressure = self.gripper.getPressure()
        if pressure > 2.5:
            return True
        else:
            return False

    def drop_until_sution(self, limit=0.05):
        action = [0.0]*6
        action[2] = -0.01
        self.suction_on()
        while(not self.is_suction()):
            
            self.rtde_c.jogStart(action, self.feature)
            pose = self.get_pose()
            if(pose[2]<limit):
                self.rtde_c.jogStop()
                return False
        self.rtde_c.jogStop()
        return True
        
    def keyboard_ctrl(self):        
        action = [0.0]*6
        
        unit = 0.05
        unit_angle = np.deg2rad(5.0)
        
        traj = []
        s_time = time.time()
        while True:
            key = utils.getch()
            # key = readchar.readkey()
            if key == "q":
                quit()
            if key == "w": #  +x
                print("KEY w")
                action[1] = unit
            elif key == "s": # -x
                print("KEY s")
                action[1] = -unit
            elif key == "d": # +y
                print("KEY d")
                action[0] = unit
            elif key == "a": # -y
                print("KEY a")
                action[0] = -unit
            elif key == "k": # +z
                print("KEY k")
                action[2] = unit*0.5
            elif key == "m": # -z
                print("KEY m")
                action[2] = -unit*0.5
            elif key == "j": # +yaw
                print("KEY j")
                action[5] = unit_angle
            elif key == "l": # -yaw
                print("KEY l")
                action[5] = -unit_angle
            elif key == "g": # suction on
                self.suction_on()
            elif key == "b": # suction off
                self.suction_off()
            elif key == "p":
                self.drop_until_sution()
            elif key == "i":
                print(self.get_pose())
            elif key == "c":
                if self.cam is not None:
                    print("capture img")
                    cv2.imwrite("frame.png", self.get_img())
                    print("captured img")
            elif key == "r":
                print("KEY r")
                traj = []
                if self.bottleneck_pose is None:
                    self.rtde_c.jogStop()
                    time.sleep(1.0)
                    self.moveJ(self.home_joint_angles)
                
                else:
                    self.rtde_c.jogStop()
                    time.sleep(1.0)
                    self.moveL(self.bottleneck_pose)
                    s_time = time.time()
           
            elif key == "y":
                if self.mode == "init":
                    print("set bottleneck pose")
                    self.bottleneck_pose = self.get_pose()
                    traj = []
                    self.mode = "teaching"
                    print("set mode teaching")
                    s_time = time.time()
                elif self.mode == "teaching":
                    self.rtde_c.jogStop()
                    self.mode = "init"
                    return self.bottleneck_pose, traj
            else:
                action = [0.0]*6
                
            print(action,  key)
            
            self.rtde_c.jogStart(action, self.feature)
            
            if time.time()-s_time > 0.01:
                pose = self.rtde_r.getActualTCPPose()
                # traj.append(self.rtde_r.getActualTCPPose())
                traj.append(np.append(pose, [self.vel, self.acc, self.blend]))
                s_time = time.time()
            
    def direct_teaching(self):
        print("press any key to start direct teaching")
        # sys.stdin.readline()
        
        
        traj = []
        s_time = time.time()
        self.rtde_c.teachMode()
        print("start direct teaching mode")
        while True:
            key = utils.getch()
            if key == "q":
                break
            elif key == "r":
                print("KEY r")
                traj = []
                self.mode = "init"
                print("set mode init")
            elif key == "g": # suction on
                self.suction_on()
            elif key == "b": # suction off
                self.suction_off()
            elif key == "y":
                if self.mode == "init":
                    print("set bottleneck pose")
                    self.bottleneck_pose = self.get_pose()
                    traj = []
                    self.mode = "teaching"
                    print("set mode teaching")
                    s_time = time.time()
                elif self.mode == "teaching":
                    self.rtde_c.stopScript()
                    self.mode = "init"
                    return self.bottleneck_pose, traj
            
            if time.time()-s_time > 0.01:
                pose = self.rtde_r.getActualTCPPose()
                # traj.append(self.rtde_r.getActualTCPPose())
                traj.append(np.append(pose, [self.vel, self.acc, self.blend]))
                s_time = time.time()
        
        self.rtde_c.endTeachMode()
        return traj
            
    def play_demo(self, traj):
        print("start demo")
        traj[-1][-1] = 0 
        self.rtde_c.moveL(traj)
        print("end demo")

    def reverse_play_demo(self, traj):
        reverse_traj = copy.deepcopy(traj)
        reverse_traj.reverse()
        self.play_demo(reverse_traj)
        
   
    def convertTCP(self, traj):
        previous_pose = traj[0][:6]
        traj_tcp = []
        for t in traj:
            pose = t[:6]
            pose_tcp = utils.transform(previous_pose, pose)
            traj_tcp.append(np.append(pose_tcp, t[6:]))
            previous_pose = pose
            
        return traj_tcp
    
    def play_demo_tcp(self, traj_tcp):
        previous_pose = self.get_pose()
        
        traj = []
        for t_tcp in traj_tcp:
            pose_tcp = t_tcp[:6]
            pose = utils.reverse_transform(previous_pose, pose_tcp)
            traj.append(np.append(pose, t_tcp[6:]))
            previous_pose = pose
        
        self.play_demo(traj)
        
    def reverse_demo_tcp(self, trajectory_tcp):
        previous_pose = self.get_pose()
        
        traj = []
        for t_tcp in reversed(trajectory_tcp):
            pose_tcp = -1*t_tcp[:6]
            pose = utils.reverse_transform(previous_pose, pose_tcp)
            traj.append(np.append(pose, t_tcp[6:]))
            previous_pose = pose
        
        self.play_demo(traj)
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_teaching", "-d", action="store_true")
    parser.add_argument("--config", "-c ", type=str, default="configs/robot_config.json")

    args = parser.parse_args()
    
    ur = URController(config_file=args.config)

    bottleneck_pose, recorded_traj = ur.keyboard_ctrl()
    
    if bottleneck_pose is not None and len(recorded_traj) > 0:
        print(recorded_traj[0])
        ur.moveL(bottleneck_pose)
        ur.play_demo(recorded_traj)
        
        ur.moveL(bottleneck_pose)
        traj_tcp = ur.convertTCP(recorded_traj)
        ur.play_demo_tcp(traj_tcp)

        ur.reverse_demo_tcp(traj_tcp)
        
