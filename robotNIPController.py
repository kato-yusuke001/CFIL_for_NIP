import sys
sys.path.append("C:\\Users\\4039423\\Desktop\\N.I.P._ver.7.4.0.0\\binary\\python\\CFIL_for_NIP")
#print(sys.path)
# print(sys.version)
# print(sys.version_info)
from pathlib import Path
# print(Path.cwd())
# print(help("python"))
import os
print(os.getcwd())
# import torch

import abc
import logging
import time
import numpy as np

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiq_epick import RobotiqEpick
from robotiq_rtu import RobotiqRtu
from vacuum_gripper import VacuumGripper
from rtde_io import RTDEIOInterface
from dashboard_client import DashboardClient

from scipy.spatial.transform import Rotation

import json

from CFIL_for_NIP.agent import CFIL_ABN


# パラメータ設定
LOGFILENAME = "robotNIPController"
LOGFILEROOT = "logs"
LOGLEVEL = logging.DEBUG

# ログ設定
os.makedirs(LOGFILEROOT, exist_ok=True)
log_file_path = LOGFILEROOT + "/" + LOGFILENAME + ".log"
log_format = "%(asctime)s %(levelname)s:%(message)s"
logging.basicConfig(filename=log_file_path, level=LOGLEVEL, format=log_format)

# ロボット設定
print(os.getcwd())
# ROBOTPARAMS = os.path.dirname(__file__) + "/robot_config_kato.json"
ROBOTPARAMS = os.path.dirname(__file__) + "/robot_config_ew.json"
# ROBOTPARAMS = os.path.dirname(__file__) + "/robot_config_sim.json"
json_file = open(ROBOTPARAMS, "r")
json_dict = json.load(json_file)
ROBOT_IP = json_dict["robot_ip"]
USE_GRIPPER = json_dict["use_gripper"]
TCP = json_dict["tcp"]
print(ROBOT_IP)

rtde_c = None
rtde_r = None
gripper = None
dashboard = None
io = None

cfil = None

class RobotClient(metaclass=abc.ABCMeta):
    # 初期化処理
    def initialize(self, directory):
        os.chdir(path = directory)

    # 終了処理
    def terminate(self):
        pass

    @abc.abstractmethod
    def execute(self, solution):
        pass


# 通信開始
class RobotConnect(RobotClient):
    def execute(self, solution):
        global rtde_c, rtde_r, gripper, dashboard, io
        try:
            dashboard = DashboardClient(ROBOT_IP)
            dashboard.connect()
            print("dashboard connected")
            # if(dashboard.isConnected()):
            #     dashboard.powerOn()
            #     dashboard.brakeRelease()
            #     dashboard.unlockProtectiveStop()
            if(rtde_c is None): rtde_c = RTDEControlInterface(ROBOT_IP)
            print("rtde_c connected")
            if(rtde_r is None): rtde_r = RTDEReceiveInterface(ROBOT_IP)
            print("rtde_r connected")
            if(io is None): io = RTDEIOInterface(ROBOT_IP)
            print("io connected")
            gripper = None
            if USE_GRIPPER == "socket": # EPcikをURのフランジに接続するとき
                gripper = RobotiqEpick()
                gripper.connect(ROBOT_IP, 63352)
                gripper.activate()
            elif USE_GRIPPER == "rtu": # EPickをCOMポートでPCに接続するとき
                if(gripper is None):
                    gripper = RobotiqRtu()
                    gripper.connect("COM3")
                    gripper.activate()
            elif USE_GRIPPER == "ejector": # 圧縮空気を使ってエジェクターから真空吸着するとき
                gripper = VacuumGripper(_io=io, _rtde_r=rtde_r)
            else:
                pass
            print("gripper connected")
            print("Robot Connection is established !")
            set_variable(solution, "Server_Connect", 1)
            return solution.judge_pass()
        except Exception as e:
            # 全ての種類のエラーを取得
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# 通信終了
class RobotDisconnect(RobotClient):
    def execute(self, solution):
        global rtde_c, rtde_r, gripper, dashboard, io
        try:
            if rtde_c is not None:
                rtde_c.disconnect()
            if rtde_r is not None:
                rtde_r.disconnect()
            if io is not None:
                io.disconnect()
            if USE_GRIPPER:
                gripper.disconnect()
            return solution.judge_pass()
        
        except Exception as e:
            print(type(e), e)
            return solution.judge_fail()
        
# ロボットの軌道とブレーキのリリースをリクエスト
class WakeupRobot(RobotClient):
    # 実行
    def execute(self, solution):
        global rtde_c, rtde_r, dashboard, io, gripper
        try:
            print("Safety Mode", rtde_r.getSafetyMode())
            print("Robot Status", rtde_r.getRobotStatus())
            if(rtde_r.getRobotStatus()==3):
                dashboard.closeSafetyPopup()
                print(" Status Normal")
                return solution.judge_pass()
            print("wakeup robot ...")
            if(dashboard.isConnected()):
                dashboard.closeSafetyPopup()
                
                if(rtde_r.getSafetyMode() == 9):
                    print(" Status Failed")
                    dashboard.restartSafety()
                    time.sleep(3)
                    dashboard.powerOn()
                    time.sleep(3)
                    dashboard.brakeRelease()
                    dashboard.unlockProtectiveStop()
                    rtde_c.disconnect()
                    rtde_r.disconnect()
                    gripper.disconnect()
                    # io.disconnect()
                    rtde_c.reconnect()
                    rtde_r.reconnect()
                    # io.reconnect()
                elif(rtde_r.getSafetyMode() == 7):
                    print(" Status Emergency Button still pressed")
                    return solution.judge_fail()
                else:
                    print(" Status Standby")
                    dashboard.powerOn()
                    dashboard.brakeRelease()
                    dashboard.unlockProtectiveStop()
                    io.setConfigurableDigitalOut(0,1)
                    io.setConfigurableDigitalOut(1,1)
                    # io.setConfigurableDigitalOut(0,0)
                    # io.setConfigurableDigitalOut(1,0)
                    rtde_c.disconnect()
                    rtde_r.disconnect()
                    # io.disconnect()
                    #if USE_GRIPPER: gripper.disconnect()
                    rtde_c.reconnect()
                    rtde_r.reconnect()
                    # io.reconnect()
                    #if USE_GRIPPER: gripper.reconnect()
                    print("Robot status", rtde_r.getRobotStatus())
                s_time = time.time()
                while(rtde_r.getRobotStatus()!=3):
                    if(time.time()-s_time > 30):
                        print("Failed to wakeup robot")
                        set_variable(solution, "Servo_On", 0)
                        return solution.judge_fail()
                    print(rtde_r.getRobotStatus(), rtde_r.getSafetyMode())
                    rtde_c.disconnect()
                    rtde_r.disconnect()
                    # io.disconnect()
                    rtde_c.reconnect()
                    rtde_r.reconnect()
                    # io.reconnect()
                print("Completed to wakeup robot")
                set_variable(solution, "Servo_On", 1)
                return solution.judge_pass()
            else:
                print("Failed to wakeup robot")
                set_variable(solution, "Servo_On", 0)
                return solution.judge_fail()

        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()
        
# ロボットのサーボ状態の取得
class GetServoStatus(RobotClient):
    # 実行
    def execute(self, solution):
        global rtde_c, rtde_r, dashboard, io, gripper
        try:
            print("Safety Mode", rtde_r.getSafetyMode())
            print("Robot Status", rtde_r.getRobotStatus())
            if(rtde_r.getRobotStatus()==3):
                dashboard.closeSafetyPopup()
                print(" Status Normal")
                return solution.judge_pass()
            print("wakeup robot ...")
            if(dashboard.isConnected()):
                dashboard.closeSafetyPopup()
                
                if(rtde_r.getSafetyMode() == 9):
                    print(" Status Failed")
                    return solution.judge_fail()
                elif(rtde_r.getSafetyMode() == 7):
                    print(" Status Emergency Button still pressed")
                    return solution.judge_fail()
                else:
                    if(rtde_r.getRobotStatus()==1):
                        rtde_c.disconnect()
                        rtde_r.disconnect()
                        rtde_c.reconnect()
                        rtde_r.reconnect()
                        return solution.judge_success()    
                    else:
                        print(" Servo off")
                        return solution.judge_fail()

            else:
                print("Failed to get servo status")
                set_variable(solution, "Servo_On", 0)
                return solution.judge_fail()

        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()
        
# 自己位置のリクエスト
class GetCurrentPose(RobotClient):
    def execute(self, solution):
        global rtde_r
        try:                       
            current_pose = rtde_r.getActualTCPPose()
            print("current pose: ", current_pose)
            set_variable(solution, "current_robot_pose", current_pose)
            return solution.judge_pass()
         
        except Exception as e:
            # 全ての種類のエラーを取得
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# ロボットの手先位置(TCP, tool center point)を設定する
class SetTCP(RobotClient):
    # 実行
    def execute(self, solution):
        global rtde_c
        try:
            # 設定したいtcpを取得
            variable_id_tcp = solution.get_variable_id("tcp")
            tcp = list(solution.get_variable(variable_id_tcp).values())
            
            # tcpの設定
            ret = rtde_c.setTcp(tcp)
            if(ret):
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える  

        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# MoveLリクエスト
class MoveL(RobotClient):
    # 実行
    def execute(self, solution):
        global rtde_c
        try:
            # NIP配列変数から移動先の姿勢を取得
            variable_id_dst = solution.get_variable_id("dst")
            if variable_id_dst == 0:
                print("Variable named dst did not found")
                logging.error("Variable named dst did not found")
                return solution.judge_fail()
            dst = list(solution.get_variable(variable_id_dst).values())
            if(sum(dst)==0):
                print("moveL failed, dst is all 0")
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

            dst = self.checkLimit(dst) 
            # ret = rtde_c.moveL(dst, speed=0.15, acceleration=0.5)
            ret = rtde_c.moveL(dst, speed=1.15, acceleration=1.5)
            if(ret):
                print("moveL success: ", dst)
                # set_variable(solution, "Server_Connect", 1)
                return solution.judge_pass()
            else:
                print("moveL failed")
                set_variable(solution, "Server_Connect", 0)
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える
        
        except Exception as e:
            # 全ての種類のエラーを取得
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

    def checkLimit(self, dst):
        if(dst[2]< -0.005):
            dst[2] = -0.005
        if(dst[2]> 0.15):
            dst[2] = 0.15
        return dst

class Vac_On(RobotClient):
    def execute(self, solution):
        global gripper
        if USE_GRIPPER:
            try:
                gripper.grip_off()
                gripper.grip_on()
                time.sleep(1)
                print("Vacuum on done")
                logging.info("Vacuum on done")
                set_variable(solution, "Pressure", gripper.getPressure())
                return solution.judge_pass()
            
            except Exception as e:
                print(type(e), e)
                logging.error("{} : {}".format(type(e), e))
                return solution.judge_fail()
        else:
            return solution.judge_pass()

class Vac_Off(RobotClient):
    def execute(self, solution):
        global gripper
        if USE_GRIPPER:
            try:
                gripper.grip_off()
                print("Vacuum off done")
                logging.info("Vacuum off done")
                return solution.judge_pass()
            
            except Exception as e:
                print(type(e), e)
                logging.error("{} : {}".format(type(e), e))
                return solution.judge_fail()

        else:
            return solution.judge_pass()

class GetPressure(RobotClient):
    def execute(self, solution):
        global gripper
        if USE_GRIPPER:
            pressure = gripper
            set_variable(solution, "Pressure", pressure)
            return solution.judge_pass()
        else:
            return solution.judge_fail()

class CheckPressure(RobotClient):
    def execute(self, solution):
        global gripper
        if USE_GRIPPER:
            pressure = gripper.getPressure()
            if(pressure < 80):
                return solution.judge_pass()
            else:
                return solution.judge_fail()
        else:
            return solution.judge_pass()

def makeDict(value):
    if isinstance(value, list):
        return dict(zip(np.arange(1,len(value)+1), value))
    elif isinstance(value, int):
        return {0:value}
    elif isinstance(value, float):
        return {0:value}
    else:
        return NotImplementedError

def set_variable(solution, variable_name, value):
    variable_id = solution.get_variable_id(variable_name)
    solution.set_variable(variable_id, makeDict(value))
    return solution

def get_variable(solution, variable_name):
    variable_id = solution.get_variable_id(variable_name)
    val = list(solution.get_variable(variable_id).values())
    return val 

#############################################################################################################################################
#############################################################################################################################################
#CFIL
#############################################################################################################################################
#############################################################################################################################################
from CFIL_for_NIP.memory import ApproachMemory
memory_size = 5e4
# device = "cuda" if torch.cuda.is_available() else "cpu"  
device = "cpu"  
approach_memory = ApproachMemory(memory_size, device)

class Convert_c_T_r(RobotClient):
    def execute(self, solution):
        try:

            mtx = np.load(os.path.dirname(__file__) + "/mtx.npy")
            g_t_c = np.load(os.path.dirname(__file__) + "/g_t_c.npy")
            g_R_c = np.load(os.path.dirname(__file__) + "/g_R_c.npy")
            print(g_t_c, g_R_c)
            g_R_c = Rotation.from_rotvec(g_R_c)
            gTc = np.r_[np.c_[g_R_c.as_matrix(), g_t_c], np.array([[0, 0, 0, 1]])]

            target_s = np.array(get_variable(solution, "target_in_camera"))

            rtde_c.setTcp(np.concatenate([g_t_c, [0,0,0]], 0))
            target_s_ = target_s*rtde_r.getActualTCPPose()[2]
            print(target_s_)
            rtde_c.setTcp(TCP)
            target_c = np.dot(np.linalg.inv(mtx), target_s_)

            R = np.asarray([[1,0,0], [0,1,0], [0,0,1]])

            print(R.shape, target_c.shape)
            cTt = np.r_[np.c_[R, target_c.T], np.array([[0, 0, 0, 1]])]

            robot_pose = rtde_r.getActualTCPPose()
            rot_bTg = Rotation.from_rotvec(robot_pose[3:])
            bTg = np.r_[np.c_[rot_bTg.as_matrix(), np.array(robot_pose[:3]).T], np.array([[0, 0, 0, 1]])]

            bTt = np.dot(np.dot(bTg, gTc),cTt)
            print(bTt)

            target = bTt[:-1,-1]
            target_in_robotbase = np.concatenate([target[:3], robot_pose[3:]], 0)
            print(target_in_robotbase)
            set_variable(solution, "target_in_robotbase", target_in_robotbase.tolist())

            return solution.judge_pass()          
        
        except Exception as e:
                print(type(e), e)
                logging.error("{} : {}".format(type(e), e))
                return solution.judge_fail()

class LoadTrainedModel(RobotClient):
    def execute(self, solution):
        import torch
        
        global cfil
        try:
            cfil = CFIL_ABN()
            cfil.load_for_test()
            return solution.judge_pass() 
        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

class CFILExecute(RobotClient):
    def execute(self, solution):
        global cfil
        try:
            if cfil is None:
                print("CFIL model is not prepared")
                return solution.judge_fail()
            
            image_id = solution.get_image_id("image")
            image = solution.get_image(image_id)

            output = cfil.approach_test_from_image(image)
            return solution.judge_pass() 
        
        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

class LoadImage(RobotClient):
    def execute(self, solution):
        
        try:
            
            image_id = solution.get_image_id("image")
            image = solution.get_image(image_id)
            print(image)
            return solution.judge_pass() 
        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()
        
class ExecuteCollectData(RobotClient):
    def execute(self, solution):
        try:
            global rtde_r, approach_memory
            current_pose = rtde_r.getActualTCPPose()
            print("current pose: ", current_pose)
            image_id = solution.get_image_id("image")
            if image_id==0:
                print("Image ID not found")
                return solution.judge_fail()
            image = solution.get_image(image_id)

            if approach_memory.initialize is False:
                approach_memory.initial_settings(image, current_pose) # 画像と姿勢のサイズの初期設定

            approach_memory.append(image, current_pose)

            return solution.judge_pass()
        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()
    
class SaveCollectData(RobotClient):
    def execute(self, solution):
        try:
            global approach_memory
            path = os.getcwd()
            approach_memory.save_joblib(os.path.join(path, "CFIL_for_NIP", "approach_memory"))
            return solution.judge_pass()
        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()
