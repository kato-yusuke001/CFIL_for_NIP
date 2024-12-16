import sys
sys.dont_write_bytecode = True # キャッシュを生成させない
nip_path = "C:\\Users\\4039423\\Desktop\\N.I.P._ver.7.4.0.0\\binary\\python\\CFIL_for_NIP"

if not nip_path in sys.path:
    sys.path.append(nip_path)

import os
print(os.getcwd())
# import torch

import abc
# import logging
import time
import numpy as np

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiq_epick import RobotiqEpick
# from robotiq_rtu import RobotiqRtu
from vacuum_gripper import VacuumGripper
from rtde_io import RTDEIOInterface
from dashboard_client import DashboardClient

from scipy.spatial.transform import Rotation

import json

from logger import setup_logger

# # パラメータ設定
# LOGFILENAME = "robotNIPController"
# LOGFILEROOT = "logs"
# LOGLEVEL = logging.INFO

# # ログ設定
# os.makedirs(LOGFILEROOT, exist_ok=True)
# log_file_path = LOGFILEROOT + "/" + LOGFILENAME + ".log"
# log_format = "%(asctime)s %(levelname)s:%(message)s"
# logging.basicConfig(filename=log_file_path, level=LOGLEVEL, format=log_format)

logging = setup_logger("robotNIPController", "robotNIPController")

# ロボット設定
print(os.getcwd())
# ROBOTPARAMS = os.path.dirname(__file__) + "/robot_config_kato.json"
ROBOTPARAMS = os.path.dirname(__file__) + "/robot_config_ew.json"
# ROBOTPARAMS = os.path.dirname(__file__) + "/robot_config_sim.json"
# ROBOTPARAMS = os.path.dirname(__file__) + "/robot_config_daic.json"
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

TIMEOUT = 15

def log_info_output(log_message):
    print(log_message)
    logging.info(log_message)

def log_error_output(log_message):
    print(log_message)
    logging.error(log_message)


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


# ロボットとPCの接続を確立
class RobotConnect(RobotClient):
    def execute(self, solution):
        global rtde_c, rtde_r, gripper, dashboard, io
        try:
            dashboard = DashboardClient(ROBOT_IP)
            dashboard.connect()
            log_info_output("dashboard connected")
           
            if(rtde_r is None): rtde_r = RTDEReceiveInterface(ROBOT_IP)
            log_info_output("rtde_r connected")
            log_info_output("Safety Mode: {}".format(rtde_r.getSafetyMode()))
            log_info_output("Robot Status: {}".format(rtde_r.getRobotStatus()))
            log_info_output("Robot Mode: {}".format(rtde_r.getRobotMode()))
            if(dashboard.isConnected()):
                dashboard.closeSafetyPopup()
                if(rtde_r.getSafetyMode() == 2 or rtde_r.getSafetyMode() == 8 or rtde_r.getSafetyMode() == 9):
                    dashboard.restartSafety()
                s_time = time.time()
                while(rtde_r.getRobotMode() != 5 and rtde_r.getRobotMode() != 7):
                    dashboard.powerOn()
                    log_info_output(" Power on ...")
                    time.sleep(1)
                    if(time.time()-s_time > TIMEOUT):
                        log_error_output("Failed to power on robot")
                        return solution.judge_fail()
                # dashboard.brakeRelease()
                # s_time = time.time()
                # while(rtde_r.getRobotMode() != 7):
                #     log_info_output(" break release ...")
                #     time.sleep(1)
                #     if(time.time()-s_time > TIMEOUT):
                #         log_error_output("Failed to release brake")
                #         return solution.judge_fail()
                # dashboard.stop()

            # if(rtde_c is None): rtde_c = RTDEControlInterface(ROBOT_IP)
            # log_info_output("rtde_c connected")
            
            if(gripper is not None):
                pass
            elif USE_GRIPPER == "socket": # EPcikをURのフランジに接続するとき
                gripper = RobotiqEpick()
                gripper.connect(ROBOT_IP, 63352)
                gripper.activate()
            # elif USE_GRIPPER == "rtu": # EPickをCOMポートでPCに接続するとき
            #     gripper = RobotiqRtu()
            #     gripper.connect("COM3")
            #     gripper.activate()
            elif USE_GRIPPER == "ejector": # 圧縮空気を使ってエジェクターから真空吸着するとき
                if(io is None): io = RTDEIOInterface(ROBOT_IP)
                log_info_output("io connected")
                gripper = VacuumGripper(_io=io, _rtde_r=rtde_r)
            else:
                pass
            log_info_output("gripper connected")
            log_info_output("Robot Connection is established !")
            set_variable(solution, "Server_Connect", 1)
            return solution.judge_pass()
        except Exception as e:
            # 全ての種類のエラーを取得
            log_error_output("Robot Connection is failed !")
            log_error_output("{} : {}".format(type(e), e))
            dashboard.disconnect()
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
            log_error_output("{} : {}".format(type(e), e))
            return solution.judge_fail()
        
# ロボットの軌道とブレーキのリリースをリクエスト
class WakeupRobot(RobotClient):
    # 実行
    def execute(self, solution):
        global rtde_c, rtde_r, dashboard, io, gripper
        try:
            # robotの状態確認
            log_info_output("Safety Mode: {}".format(rtde_r.getSafetyMode()))
            log_info_output("Robot Status: {}".format(rtde_r.getRobotStatus()))
            log_info_output("Robot Mode: {}".format(rtde_r.getRobotMode()))
            # if(rtde_r.getRobotStatus()==3):
            if(rtde_r.getRobotMode()==7): # Robot mode is running
                
                if(rtde_c is None): 
                    rtde_c = RTDEControlInterface(ROBOT_IP)
                log_info_output("Status is Normal: Robot is running")
                return solution.judge_pass()
            
            else:
                dashboard.closeSafetyPopup()
                log_info_output("wakeup robot ...")
                if(dashboard.isConnected()):
                    dashboard.closeSafetyPopup()
                    if(rtde_r.getSafetyMode() == 2 or rtde_r.getSafetyMode() == 8 or rtde_r.getSafetyMode() == 9):
                        dashboard.restartSafety()
                    s_time = time.time()
                    while(rtde_r.getRobotMode() != 5 and rtde_r.getRobotMode() != 7):
                        dashboard.powerOn()
                        log_info_output(" Power on ...")
                        time.sleep(1)
                        if(time.time()-s_time > TIMEOUT):
                            log_error_output("Failed to power on robot")
                            return solution.judge_fail()
                    dashboard.brakeRelease()
                    s_time = time.time()
                    while(rtde_r.getRobotMode() != 7):
                        log_info_output(" break release ...")
                        time.sleep(1)
                        if(time.time()-s_time > TIMEOUT):
                            log_error_output("Failed to release brake")
                            return solution.judge_fail()
                    dashboard.stop()
                    if(rtde_c is None): 
                        rtde_c = RTDEControlInterface(ROBOT_IP)
                    else:
                        rtde_c.disconnect()
                        rtde_c = RTDEControlInterface(ROBOT_IP)
                    log_info_output("rtde_c connected")

        except Exception as e:
            log_error_output("{} : {}".format(type(e), e))
            return solution.judge_fail()
        
# ロボットのサーボ状態の取得
class GetServoStatus(RobotClient):
    # 実行
    def execute(self, solution):
        global rtde_c, rtde_r, dashboard
        try:
            log_info_output("Safety Mode: {}".format(rtde_r.getSafetyMode()))
            log_info_output("Robot Status: {}".format(rtde_r.getRobotStatus()))
            log_info_output("Robot Mode: {}".format(rtde_r.getRobotMode()))
            if(rtde_r.getRobotMode()==7): # Robot mode is running
                if(rtde_c is None): 
                    rtde_c = RTDEControlInterface(ROBOT_IP)
                    log_info_output("rtde_c connected")
                log_info_output(" Status is Normal: Robot is running")
                set_variable(solution, "Servo_On", 1)
                return solution.judge_pass()
            else:
                log_info_output("Sero is off")
                
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
            log_info_output("GetCurrentPose[m] : {}".format(current_pose))

            set_variable(solution, "current_robot_pose", current_pose)
            return solution.judge_pass()
         
        except Exception as e:
            # 全ての種類のエラーを取得
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# 自己位置のリクエスト (mm)
class GetCurrentPose_mm(RobotClient):
    def execute(self, solution):
        global rtde_r
        try:                       
            current_pose = rtde_r.getActualTCPPose()
            current_pose[0] *= 1000
            current_pose[1] *= 1000
            current_pose[2] *= 1000
            log_info_output("GetCurrentPose[mm] : {}".format(current_pose))
            set_variable(solution, "current_robot_pose", current_pose)
            return solution.judge_pass()
         
        except Exception as e:
            # 全ての種類のエラーを取得
            log_error_output("{} : {}".format(type(e), e))
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
            log_error_output("{} : {}".format(type(e), e))
            return solution.judge_fail()

class SetTCP_mm(RobotClient):
    # 実行
    def execute(self, solution):
        global rtde_c
        try:
            # 設定したいtcpを取得
            variable_id_tcp = solution.get_variable_id("tcp")
            tcp = list(solution.get_variable(variable_id_tcp).values())
            tcp[0] /= 1000.0
            tcp[1] /= 1000.0
            tcp[2] /= 1000.0
            # tcpの設定
            ret = rtde_c.setTcp(tcp)
            if(ret):
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える  

        except Exception as e:
            log_error_output("{} : {}".format(type(e), e))
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
                log_error_output("Variable named dst did not found")
                return solution.judge_fail()
            dst = list(solution.get_variable(variable_id_dst).values())
            log_info_output("MoveL dst: {}".format(dst))
            if(sum(dst)==0):
                log_error_output("moveL failed, dst is all 0")
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

            dst = self.checkLimit(dst) 
            # ret = rtde_c.moveL(dst, speed=0.15, acceleration=0.5)
            ret = rtde_c.moveL(dst, speed=1.15, acceleration=1.5)
            if(ret):
                log_info_output("moveL success: {}".format(dst))
                # set_variable(solution, "Server_Connect", 1)
                return solution.judge_pass()
            else:
                log_error_output("moveL failed")
                set_variable(solution, "Server_Connect", 0)
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える
        
        except Exception as e:
            # 全ての種類のエラーを取得
            log_error_output("{} : {}".format(type(e), e))
            return solution.judge_fail()

    def checkLimit(self, dst):
        if(dst[2]< -0.005):
            dst[2] = -0.005
        if(dst[2]> 0.15):
            dst[2] = 0.15
        return dst

# MoveLリクエスト (mm)
class MoveL_mm(RobotClient):
    # 実行
    def execute(self, solution):
        global rtde_c
        try:
            # NIP配列変数から移動先の姿勢を取得
            variable_id_dst = solution.get_variable_id("dst")
            if variable_id_dst == 0:
                log_error_output("Variable named dst did not found")
                return solution.judge_fail()
            dst = list(solution.get_variable(variable_id_dst).values())
            log_info_output("MoveL mm dst: {}".format(dst))
            if(sum(dst)==0):
                log_error_output("moveL failed, dst is all 0")
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える
            dst[0] /= 1000.0
            dst[1] /= 1000.0
            dst[2] /= 1000.0
            dst = self.checkLimit(dst) 
            # ret = rtde_c.moveL(dst, speed=0.15, acceleration=0.5)
            ret = rtde_c.moveL(dst, speed=1.15, acceleration=1.5)
            if(ret):
                log_info_output("moveL(mm) success: {}".format(dst))
                # set_variable(solution, "Server_Connect", 1)
                return solution.judge_pass()
            else:
                log_error_output("moveL(mm) failed")
                set_variable(solution, "Server_Connect", 0)
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える
        
        except Exception as e:
            # 全ての種類のエラーを取得
            log_error_output("{} : {}".format(type(e), e))
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
                log_info_output("Vacuum on done")
                set_variable(solution, "Pressure", gripper.getPressure())
                return solution.judge_pass()
            
            except Exception as e:
                log_error_output("{} : {}".format(type(e), e))
                return solution.judge_fail()
        else:
            return solution.judge_pass()

class Vac_Off(RobotClient):
    def execute(self, solution):
        global gripper
        if USE_GRIPPER:
            try:
                gripper.grip_off()
                log_info_output("Vacuum off done")
                return solution.judge_pass()
            
            except Exception as e:
                log_error_output("{} : {}".format(type(e), e))
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

class Rotate(RobotClient):
    def execute(self, solution):
        global rtde_r
        try:
            base_pose = get_variable(solution, "robot_initial_pose")
            angles = get_variable(solution, "rotate_angles")
            angles = [np.deg2rad(angle) for angle in angles]
            goal_pose = rotate(base_pose, angles)
            set_variable(solution, "target_rotVec", goal_pose[3:])
            return solution.judge_pass()
        except Exception as e:
            log_error_output("{} : {}".format(type(e), e))
            return solution.judge_fail()

class LoadJson(RobotClient):
    def execute(self, solution):
        try:
            json_file = get_variable(solution, "json_file")
            json_file = json_file[0]
            # print(json_file)
            json_dict = json.load(open(json_file, "r"))
            log_info_output(json_dict)
            set_variable_string(solution, "train_data_file", json_dict["train_data_file"])
            return solution.judge_pass()
        except Exception as e:
            log_error_output("{} : {}".format(type(e), e))
            return solution.judge_fail()


def makeDict(value):
    if isinstance(value, list):
        return dict(zip(np.arange(1,len(value)+1), value))
    if isinstance(value, np.ndarray):
        return dict(zip(np.arange(1,len(value)+1), value.tolist()))
    elif isinstance(value, int):
        return {0:value}
    elif isinstance(value, float):
        return {0:value}
    else:
        return NotImplementedError

def set_variable(solution, variable_name, value):
    variable_id = solution.get_variable_id(variable_name)
    if variable_id == 0:
        log_error_output("Variable named {} did not found".format(variable_name))
        return solution.judge_fail()
    solution.set_variable(variable_id, makeDict(value))
    return solution

def get_variable(solution, variable_name):
    variable_id = solution.get_variable_id(variable_name)
    if variable_id == 0:
        log_error_output("Variable named {} did not found".format(variable_name))
        return solution.judge_fail()
    val = list(solution.get_variable(variable_id).values())
    return val 

def set_variable_string(solution, variable_string_name, value_string):
    variable_id = solution.get_variable_id(variable_string_name)
    if variable_id == 0:
        log_error_output("Variable named {} did not found".format(variable_string_name))
        return solution.judge_fail()
    solution.set_variable_string(variable_id, value_string)
    return solution


def rotate(pose, angles, order="xyz"):
    rot = Rotation.from_euler(order, angles)
    pose_rotvec = Rotation.from_rotvec(pose[3:])

    result_rot = rot * pose_rotvec
    result_rotvec = result_rot.as_rotvec()
    
    result_pose = np.r_[pose[:3], result_rotvec]

    return result_pose