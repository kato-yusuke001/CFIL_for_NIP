import time
import numpy as np
import rtde_control
import rtde_receive
from pose_utils import *

class URController:
    def __init__(self, robot_ip, home_pose, tcp_pose):
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.tcp_pose = tcp_pose
        self.set_tcp(self.tcp_pose)
        self.home_pose = home_pose
        self.go_to_home()

    def go_to_home(self):
        self.moveL(self.home_pose, vel=0.1, acc=1.0)

    def move_upward(self, speed=0.01, wait=1.0):
        self.move_dxyz([0, 0, speed], wait)

    def move_downward(self, speed=0.01, wait=1.0):
        self.move_dxyz([0, 0, -speed], wait)
    
    def stop(self):
        self.pause()
        self.rtde_c.stopScript()

    def pause(self):
        self.rtde_c.speedStop(a=0.5)

    def set_tcp(self, pose):
        self.rtde_c.setTcp(pose)

    def moveL(self, pose, vel=0.1, acc=0.1, asynchronous=False):
        self.rtde_c.moveL(pose, vel, acc, asynchronous)
    
    def moveJ(self, joints, vel=0.2, acc=0.2, asynchronous=False):
        self.rtde_c.moveJ(joints, vel, acc, asynchronous)

    def moveJ_IK(self, pose, vel=0.1, acc=0.1, asynchronous=False):
        self.rtde_c.moveJ_IK(pose, vel, acc, asynchronous)

    def moveJ_IK2(self, pose, vel=0.1, acc=0.1, asynchronous=False):
        joints = self.rtde_c.getInverseKinematics(pose)
        # print("Joints    :", np.degrees(joints).astype("int32"))
        current_joints = self.get_joint()
        # Check wrist3 limit
        joint_limit = False
        if joints[-1] > np.radians(360):
            joints[-1] -= np.radians(360)
            joint_limit = True
        elif joints[-1] <= -np.radians(360):
            joints[-1] += np.radians(360)
            joint_limit = True
        if joint_limit:
            current_joints[-1] = joints[-1]
            self.moveJ(current_joints, vel=1.0, acc=1.0, asynchronous=False)
            # print("New Joints:", np.degrees(joints).astype("int32"))
        self.moveJ(joints, vel, acc, asynchronous)

    def speedL(self, speed, acc=0.25):
        self.rtde_c.speedL(speed, acc, time=0.0)

    def jogStart(self, speed, frame="base"):
        if frame == "base":
            feature = self.rtde_c.FEATURE_BASE
        elif frame == "tool":
            feature = self.rtde_c.FEATURE_TOOL
        self.rtde_c.jogStart(speed, feature)

    def move_dxyz(self, dxyz, wait=3.0):
        action = np.array(dxyz).tolist() + [0.0] * 3
        self.rtde_c.jogStart(action, self.rtde_c.FEATURE_BASE)
        time.sleep(wait)
        self.rtde_c.jogStop()

    def get_pose(self):
        return self.rtde_r.getActualTCPPose()

    def get_joint(self):
        return self.rtde_r.getActualQ()

    # Get pose of end-effector mount
    def get_ee_pose(self):
        self.set_tcp([0.0] * 6)  # reset TCP to end-effector mount
        ee_pose = self.get_pose()
        self.set_tcp(self.tcp_pose)  # restore TCP
        return ee_pose

    def get_pose_euler(self, seq="xyz", degrees=False):
        current_pose = self.rtde_r.getActualTCPPose()
        euler = rotvec_to_euler(current_pose[3:], seq, degrees)
        return np.hstack([current_pose[:3], euler])

    def get_speed(self):
        return self.rtde_r.getActualTCPSpeed()

    def get_poseQ(self):
        pose = self.rtde_r.getActualTCPPose()
        return np.append(pose[:3], rotvec_to_quat(pose[3:]))

if __name__ == '__main__':
    robot_ip = "192.168.56.2"
    home = [0.55, 0.0, 0.3, 2.22, -2.22, 0.0]
    tcp = [-0.00945058, -0.09887839, -0.01367221, -0.00306723, -0.00102922, -0.01781922]
    uc = URController(robot_ip, home, tcp)
    uc.stop()