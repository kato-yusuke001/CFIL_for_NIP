import os
import time
import json
from logger import setup_logger
import requests
import copy
import numpy as np
import abc

from scipy.spatial.transform import Rotation

# ロガー設定
logging = setup_logger("torchClient", "torchClient")

def log_meesage(message):
    print(message)
    logging.info(message)

def log_error(message):
    print("error: {}".format(message))
    logging.error(message)


HOST = "192.168.11.55"
PORT = 5000

PROXY = ""


# Flask設定
flask_root = "http://" + HOST + ":" + str(PORT) + "/"

log_meesage("flask root:" + flask_root)

def request_post(solution, _act="", _data="", _value=None):
    act = _act
    try:
        if(_value is not None):
            prompt = {_data:str(_value)}
        else:
            prompt = {"":None,}
        print(act, prompt)
        res = requests.post(url=flask_root+act, data=prompt, proxies={"http":PROXY})
        log_meesage("Action Request: {} & {}, Response: {}".format(act, prompt, res))
        return res
    
    except Exception as e: #接続エラー時。（サーバー側との接続が出来ないときなど）
        log_error("Error in request_post: {}".format(e))
        return solution.judge_fail()
    
def check_res(res):
    if(res.status_code==200): # サーバーとの通信はできてる
        log_meesage("{},  {}".format(res.status_code, res.text))
        if(res.text == "False"): # ロボットに与えた指示が失敗
            return False
        else:
            return True
    else:
        log_error("{},  {}".format(res.status_code, res.text))
        return False

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
        log_error("Variable named {} did not found".format(variable_name))
        return solution.judge_fail()
    solution.set_variable(variable_id, makeDict(value))
    return solution

def get_variable(solution, variable_name):
    variable_id = solution.get_variable_id(variable_name)
    if variable_id == 0:
        log_error("Variable named {} did not found".format(variable_name))
        return solution.judge_fail()
    val = list(solution.get_variable(variable_id).values())
    return val 


class NIPClient(metaclass=abc.ABCMeta):
    # 初期化処理
    def initialize(self, directory):
        os.chdir(path = directory)

    # 終了処理
    def terminate(self):
        pass

    @abc.abstractmethod
    def execute(self, solution):
        pass


# CFIL初期化リクエスト
class Initialize(NIPClient):
    def execute(self, solution):
        try:
            res = request_post(solution, _act="initialize")
            if(check_res(res)):
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in initialize: {}".format(e))
            return solution.judge_fail()
        
# CFIL学習済みモデルロードリクエスト
class LoadTrainedModel(NIPClient):
    def execute(self, solution):
        try:
            model_path = get_variable(solution, "model_path")[0]
            log_meesage("model_path: {}".format(model_path))
            res = request_post(solution, _act="loadTrainedModel", _data="model_path", _value=model_path)
            if(check_res(res)):
                log_meesage("Trained Model Loaded")
                return solution.judge_pass()
            else:
                log_error("Trained Model Loading Failed")
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in loadTrainedModel: {}".format(e))
            return solution.judge_fail()

# CFIL推論実行リクエスト
class Estimate(NIPClient):
    def execute(self, solution):
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
        try:
            image_path = get_variable(solution, "image_path")[0]
            log_meesage("model_path: {}".format(image_path))
            res = request_post(solution, _act="estimate", _data="image_path", _value=image_path)
            if(check_res(res)):
                output = eval(res.text)
                position_eb = [output[0], output[1], 0.0, 0, 0, output[2]]
                position_re = get_variable(solution, "current_robot_pose")
                position_re[0] *= 1000
                position_re[1] *= 1000
                position_re[2] *= 1000
                position_rb = reverse_transform(position_re, position_eb) 
                position_rb[0] /= 1000.
                position_rb[1] /= 1000.
                position_rb[2] /= 1000.
                set_variable(solution, "estimated_pose", position_rb)
                log_meesage("Estimation Completed {}".format(position_rb))
                return solution.judge_pass()
            else:
                log_error("Estimation Failed")
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in Estimate: {}".format(e))
            return solution.judge_fail()