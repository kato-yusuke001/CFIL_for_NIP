import os
import time
import json
import logging
import requests
import copy
import numpy as np
import abc

from scipy.spatial.transform import Rotation

# パラメータ設定
LOGFILENAME = "robotClient"
LOGFILEROOT = "logs"
LOGLEVEL = logging.DEBUG
PORT = 5000

PROXY = ""

# ログ設定
os.makedirs(LOGFILEROOT, exist_ok=True)
log_file_path = LOGFILEROOT + "/" + LOGFILENAME + ".log"
log_format = "%(asctime)s %(levelname)s:%(message)s"
logging.basicConfig(filename=log_file_path, level=LOGLEVEL, format=log_format)

# ロボット設定
print(os.getcwd())
print('basename:    ', os.path.basename(__file__))
print('dirname:     ', os.path.dirname(__file__))
# ROBOTPARAMS = os.path.dirname(__file__) + r"\robot_config_kato.json"
ROBOTPARAMS = os.path.dirname(__file__) + r"\robot_config_ew.json"
json_file = open(ROBOTPARAMS, "r")
json_dict = json.load(json_file)
robot_ip = json_dict["robot_ip"]
HOST = json_dict["server_ip"]

print("robot ip:", robot_ip, "HOST:",HOST)


# Flask設定
flask_root = "http://" + HOST + ":" + str(PORT) + "/"

def request_post(solution, _act="", _data="", _value=None):
    act = _act
    try:
        if(_value is not None):
            prompt = {"target_ip":robot_ip, _data:str(_value)}
        else:
            prompt = {"target_ip":robot_ip,}
        print(act, prompt)
        res = requests.post(url=flask_root+act, data=prompt, proxies={"http":PROXY})
        print("Action Request: {} & {}, Response: {} {}".format(act, prompt, res.status_code, res.text))
        # print(res, res.text, res.status_code)
        logging.debug("Action Request: {} & {}, Response: {}".format(act, prompt, res))
        return res
    
    except Exception as e: #接続エラー時。（サーバー側との接続が出来ないときなど）
        set_variable(solution, "Server_Connect", 0)
        print("error info: ", e)
        return solution.judge_fail()
    
def check_res(solution, res):
    if(res.status_code==200): # サーバーとの通信はできてる
        set_variable(solution, "Server_Connect", 1)
        # print(res.text, res.text == "Failed")
        if(res.text == "Failed"): # ロボットに与えた指示が失敗
            set_variable(solution, "Error_urscript", 1)
            return False
        else:
            set_variable(solution, "Error_urscript", 0)
            return True
    else:
        set_variable(solution, "Server_Connect", 0)
        return False

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

def rotate(pose, angles, order="xyz"):
    rot = Rotation.from_euler(order, angles)
    pose_rotvec = Rotation.from_rotvec(pose[3:])

    result_rot = rot * pose_rotvec
    result_rotvec = result_rot.as_rotvec()
    
    # print(pose[:3], result_rotvec)
    result_pose = np.r_[pose[:3], result_rotvec]

    return result_pose

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

# 吸着ONリクエスト
class vac_on(RobotClient):
    def execute(self, solution):
        try:
            res = request_post(solution, _act="vac_on")
            if(check_res(solution, res)):
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# 吸着OFFリクエスト
class vac_off(RobotClient):
    def execute(self, solution):
        try:
            res = request_post(solution, _act="vac_off")
            if(check_res(res)):
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# 吸着圧の取得リクエスト
class getPressure(RobotClient):
    def execute(self, solution):
        try:                       
            res = request_post(solution, _act="get_pressure")

            if(check_res(solution, res)):
                p = eval(res.text)
                # print(res)
                variable_id_pressure = solution.get_variable_id("pressure")
                pressure = {0:p}
                solution.set_variable(variable_id_pressure, pressure)
                print("pressure", pressure, type(pressure), variable_id_pressure)            
                
                
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える
        
        except Exception as e:
            # 全ての種類のエラーを取得
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# MoveLリクエスト
class moveL(RobotClient):
    # 実行
    def execute(self, solution):
        try:
            # NIP配列変数から移動先の姿勢を取得
            variable_id_dst = solution.get_variable_id("dst")
            if variable_id_dst == 0:
                print("Variable named dst did not found")
                logging.error("Variable named dst did not found")
                return solution.judge_fail()
            dst = list(solution.get_variable(variable_id_dst).values())

            res = request_post(solution, _act="move_L", _data="dst", _value=dst)
            if(check_res(solution, res)):
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える
        
        except Exception as e:
            # 全ての種類のエラーを取得
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# MoveJリクエスト
class moveJ(RobotClient):
    # 実行
    def execute(self, solution):
        try:
            # NIP配列変数から移動先の姿勢を取得
            variable_id_dst_angles = solution.get_variable_id("dst_angles")
            print(solution.get_variable(variable_id_dst_angles).keys())
            dst_angles = list(solution.get_variable(variable_id_dst_angles).values())
            if variable_id_dst_angles == 0:
                print("Variable named dst did not found")
                logging.error("Variable named dst did not found")
                return solution.judge_fail()
            
            res = request_post(solution, _act="moveJ", _data="dst_angles", _value=dst_angles)
            if(check_res(solution, res)):
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える
        
        except Exception as e:
            # 全ての種類のエラーを取得
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# 自己位置のリクエスト
class GetCurrentPose(RobotClient):
    def execute(self, solution):
        try:                       
            res = request_post(solution, _act="get_current_pose")

            if(check_res(solution, res)):
                cp = eval(res.text)
                set_variable(solution, "current_robot_pose", cp)
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える
        
        except Exception as e:
            # 全ての種類のエラーを取得
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()


# 自己角度位置のリクエスト
class GetCurrentAngles(RobotClient):
    def execute(self, solution):
        try:                       
            res = request_post(solution, _act="get_current_angles")

            if(check_res(solution, res)):
                ca = eval(res.text)
                set_variable(solution, "current_robot_pose", ca)
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える
        
        except Exception as e:
            # 全ての種類のエラーを取得
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# ロボットの手先位置(TCP, tool center point)を設定する
class SetTCP(RobotClient):
    # 実行
    def execute(self, solution):
        try:
            # 設定したいtcpを取得
            variable_id_tcp = solution.get_variable_id("tcp")
            tcp = list(solution.get_variable(variable_id_tcp).values())
            
            # tcpの設定をリクエスト
            res = request_post(solution, _act="set_TCP", _data="tcp_pose", _value=tcp)
            if(check_res(solution, res)):
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える  

        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# サーバプログラムとロボットの接続をリクエスト
class ConnectRobot(RobotClient):
    # 実行
    def execute(self, solution):
        try:
            res = request_post(solution, _act="connect_robot")
            if(check_res(res)):
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# サーバプログラムとロボットの接続解除をリクエスト
class DisconnectRobot(RobotClient):
    # 実行
    def execute(self, solution):
        try:
            res = request_post(solution, _act="disconnect_robot")
            if(check_res(solution, res)):
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# ロボットの軌道とブレーキのリリースをリクエスト
class WakeupRobot(RobotClient):
    # 実行
    def execute(self, solution):
        try:
            res = request_post(solution, _act="wakeup_robot")
            if(check_res(solution, res)):
                set_variable(solution, "Servo_On", 1)
                return solution.judge_pass()
            else:
                set_variable(solution, "Servo_On", 0)
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()

# safety mode の取得
class GetSafetyMode(RobotClient):
    # 実行
    def execute(self, solution):
        try:
            res = request_post(solution, _act="get_safety_mode")

            if(check_res(solution, res)):
                p = eval(res.text)
                # print(res)
                variable_id_safety_mode = solution.get_variable_id("safety_mode")
                safety_mode = {0:p}
                solution.set_variable(variable_id_safety_mode, safety_mode)
                print("safety_mode", safety_mode, type(safety_mode), variable_id_safety_mode)            
    
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            print(type(e), e)
            logging.error("{} : {}".format(type(e), e))
            return solution.judge_fail()
