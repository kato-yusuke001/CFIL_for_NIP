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
logging = setup_logger("torchClient")

def log_meesage(message):
    print(message)
    logging.info(message)

def log_error(message):
    print("error: {}".format(message))
    logging.error(message)


HOST = "192.168.0.3" #津の設定
# HOST = "10.178.64.66" #debug
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

def request_posts(solution, _act="", _data="", _value=None):
    act = _act
    try:
        if(_value is not None):
            prompt = {}
            for d, v in zip(_data, _value):
                prompt[d] = str(v)
            # prompt = {_data:str(_value)}
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

def set_variable_string(solution, variable_name, value):
    variable_id = solution.get_variable_id(variable_name)
    if variable_id == 0:
        log_error("Variable named {} did not found".format(variable_name))
        return solution.judge_fail()
    solution.set_variable_string(variable_id, value)
    return solution


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
        log_meesage(f"### {os.path.basename(__file__)}: {self.__class__.__name__} ###")
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
        log_meesage(f"### {os.path.basename(__file__)}: {self.__class__.__name__} ###")
        try:
            file_path = get_variable(solution, "file_path")[0]
            task_name = get_variable(solution, "task_name")[0]    
            epoch = get_variable(solution, "epoch")[0]           
            log_meesage("model_path: {}".format(file_path))
            res = request_posts(solution, _act="loadTrainedModel", _data=["file_path", "task_name", "epoch"], _value=[file_path, task_name, epoch])
            if(check_res(res)):
                log_meesage("Trained Model Loaded")
                return solution.judge_pass()
            else:
                log_error("Trained Model Loading Failed")
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in loadTrainedModel: {}".format(e))
            return solution.judge_fail()
        
# SAMモデルロードリクエスト
class LoadSAMModel(NIPClient):
    def execute(self, solution):
        log_meesage(f"### {os.path.basename(__file__)}: {self.__class__.__name__} ###")
        try:
            image_save_path = get_variable(solution, "image_path")[0]
            model_path = get_variable(solution, "model_path")[0]
            sam_type = get_variable(solution, "sam_type")[0]
            log_meesage("sam image_save_path: {}".format(image_save_path))
            res = request_posts(solution, _act="loadSAMModel", _data=["image_save_path", "model_path", "sam_type"], _value=[image_save_path,model_path,sam_type])
            if(check_res(res)):
                log_meesage("SAM Model Loaded")
                return solution.judge_pass()
            else:
                log_error("SAM Model Loading Failed")
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in load SAM Model: {}".format(e))
            return solution.judge_fail()

# CFIL推論実行リクエスト
class Estimate(NIPClient):
    def execute(self, solution):
        log_meesage(f"### {os.path.basename(__file__)}: {self.__class__.__name__} ###")
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
        
        def euler2rotvec(pose_euler):
            pose = pose_euler[:3]
            euler = pose_euler[3:]
            
            assert len(euler) == 3, "len(euler) must be 3" 

            rot = Rotation.from_euler("xyz", euler)
            rotvec = rot.as_rotvec()
            pose_rotvec = np.r_[pose, rotvec]

            return pose_rotvec

        try:
            image_path = get_variable(solution, "image_path")[0]
            log_meesage("model_path: {}".format(image_path))
            res = request_post(solution, _act="estimate", _data="image_path", _value=image_path)
            if(check_res(res)):
                output = eval(res.text)
                # position_eb = [output[0], output[1], output[2], output[3], output[4], output[5]]
                position_eb = [output[0], output[1], output[2], 0.0, 0.0, output[5]]
                position_eb = euler2rotvec(position_eb)
                position_re = get_variable(solution, "current_robot_pose")
                # position_re[0] *= 1000
                # position_re[1] *= 1000
                # position_re[2] *= 1000
                print(position_eb, position_re)
                position_rb = reverse_transform(position_re, position_eb) 
                # position_rb[0] /= 1000.
                # position_rb[1] /= 1000.
                # position_rb[2] /= 1000.
                set_variable(solution, "estimated_pose", position_rb)
                log_meesage("Estimation Completed {}".format(position_rb))
                return solution.judge_pass()
            else:
                log_error("Estimation Failed")
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in Estimate: {}".format(e))
            return solution.judge_fail()
        

# SAMモデルロードリクエスト
class LoadSAM_f_Model(NIPClient):
    def execute(self, solution):
        log_meesage(f"### {os.path.basename(__file__)}: {self.__class__.__name__} ###")
        try:
            image_save_path = get_variable(solution, "result_path")[0]
            file_path = get_variable(solution, "file_path")[0]
            task_name = get_variable(solution, "task_name")[0]
            log_meesage("sam image_save_path: {}".format(image_save_path))
            res = request_posts(solution, _act="loadSAM_f_Model", _data=["image_save_path", "file_path", "task_name"], _value=[image_save_path,file_path, task_name])
            if(check_res(res)):
                log_meesage("SAM_f Model Loaded")
                return solution.judge_pass()
            else:
                log_error("SAM_f Model Loading Failed")
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in load SAM-f Model: {}".format(e))
            return solution.judge_fail()

# CFIL推論実行リクエスト
class Estimate_f(NIPClient):
    def execute(self, solution):
        log_meesage(f"### {os.path.basename(__file__)}: {self.__class__.__name__} ###")
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
        
        def euler2rotvec(pose_euler):
            pose = pose_euler[:3]
            euler = pose_euler[3:]
            
            assert len(euler) == 3, "len(euler) must be 3" 

            rot = Rotation.from_euler("xyz", euler)
            rotvec = rot.as_rotvec()
            pose_rotvec = np.r_[pose, rotvec]

            return pose_rotvec
        
        def rotate(pose, angles, order="xyz"):
            rot = Rotation.from_euler(order, angles)
            pose_rotvec = Rotation.from_rotvec(pose[3:])

            result_rot = rot * pose_rotvec
            result_rotvec = result_rot.as_rotvec()
            
            result_pose = np.r_[pose[:3], result_rotvec]

            return result_pose

        try:
            image_path = get_variable(solution, "image_path")[0]
            log_meesage("image_path: {}".format(image_path))
            res = request_post(solution, _act="estimate_f", _data="image_path", _value=image_path)
            if(check_res(res)):
                output = eval(res.text)
                
                # position_eb = [output[0], output[1], 0.0, 0.0, 0.0, 0.0]
                rotate_angles = [0, 0, output[2]]
                rotate_angles = [np.deg2rad(angle) for angle in rotate_angles]
                rot = Rotation.from_euler("xyz", rotate_angles)
                rotvec = rot.as_rotvec()
                position_eb = [output[0], output[1], 0.0, rotvec[0], rotvec[1], rotvec[2]]
                
                
                position_re = get_variable(solution, "current_robot_pose")
                print(position_eb, position_re)
                position_rb = reverse_transform(position_re, position_eb)
                # rotate_angles = [0, 0, -output[2]]
                # rotate_angles = [np.deg2rad(angle) for angle in rotate_angles]
                # position_rb = rotate(position_rb, rotate_angles)
                set_variable(solution, "estimated_pose", position_rb)
                log_meesage("Estimation Completed {}".format(position_rb))
                return solution.judge_pass()
            else:
                log_error("Estimation Failed")
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in Estimate: {}".format(e))
            return solution.judge_fail()

# CFIL推論実行リクエスト
class Estimate_f_tray(NIPClient):
    def execute(self, solution):
        log_meesage(f"### {os.path.basename(__file__)}: {self.__class__.__name__} ###")
        try:
            image_path = get_variable(solution, "image_path")[0]
            log_meesage("image_path: {}".format(image_path))
            res = request_post(solution, _act="estimate_f", _data="image_path", _value=image_path)
            if(check_res(res)):
                output = eval(res.text)
                error = [0, 0, 0]
                error[0] = output[1]
                error[1] = output[0]
                error[2] = -output[2]/10.0
                set_variable(solution, "error", error)
                log_meesage("Estimation Completed {}".format(error))
                return solution.judge_pass()
            else:
                log_error("Estimation Failed")
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in Estimate: {}".format(e))
            return solution.judge_fail()

#ワーク位置検出器初期化リクエスト
class Initialize_PD(NIPClient):
    def execute(self, solution):
        log_meesage(f"### {os.path.basename(__file__)}: {self.__class__.__name__} ###")
        try:
            res = request_post(solution, _act="initialize_PD")
            if(check_res(res)):
                return solution.judge_pass()
            else:
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in Initialize_PD: {}".format(e))
            return solution.judge_fail()

# ワーク位置検出リクエスト
class DetectPositions(NIPClient):
    def execute(self, solution):
        log_meesage(f"### {os.path.basename(__file__)}: {self.__class__.__name__} ###")
        try:
            res = request_posts(solution, _act="get_positions")
            output = eval(res.text)
            set_variable(solution, "work_positions_x", output[0])
            set_variable(solution, "work_positions_y", output[1])
            set_variable(solution, "work_num", output[2])
            if(check_res(res)):
                log_meesage("DetectPositions Completed {}".format(output))
                return solution.judge_pass()
            else:
                log_error("DetectPositions Failed {}".format(output))
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in DetectPositions: {}".format(e))
            return solution.judge_fail()
        
# ワーク位置検出リクエスト(ピクセル比)
class DetectPositions_force(NIPClient):
    def execute(self, solution):
        log_meesage(f"### {os.path.basename(__file__)}: {self.__class__.__name__} ###")
        try:
            res = request_posts(solution, _act="get_positions_force")
            output = eval(res.text)
            set_variable(solution, "work_positions_x", output[0])
            set_variable(solution, "work_positions_y", output[1])
            set_variable(solution, "work_num", output[2])
            if(check_res(res)):
                log_meesage("DetectPositions Completed {}".format(output))
                return solution.judge_pass()
            else:
                log_error("DetectPositions Failed {}".format(output))
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in DetectPositions: {}".format(e))
            return solution.judge_fail()
        
# ワークのトレイ位置検出リクエスト(ピクセル比)
class DetectTrayPositions_force(NIPClient):
    def execute(self, solution):
        log_meesage(f"### {os.path.basename(__file__)}: {self.__class__.__name__} ###")
        try:
            res = request_posts(solution, _act="get_tray_position_force")
            output = eval(res.text)
            set_variable(solution, "work_positions_x", output[0])
            set_variable(solution, "work_positions_y", output[1])
            set_variable(solution, "work_num", output[2])
            if(check_res(res)):
                log_meesage("DetectTrayPositions Completed {}".format(output))
                return solution.judge_pass()
            else:
                log_error("DetectTrayPositions Failed {}".format(output))
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error("Error in DetectTrayPositions: {}".format(e))
            return solution.judge_fail()
        

# 位置補正リクエスト
class ImageRotShift(NIPClient):
    def execute(self, solution):
        s_time = time.time()
        log_meesage(f"### {os.path.basename(__file__)}: {self.__class__.__name__} ###")
        try:
            image_dir = get_variable(solution, "image_dir")[0]
            file_name = get_variable(solution, "file_name")[0]
            repeat = get_variable(solution, "repeat")[0]
            log_meesage(f"image_path: {image_dir}, file_name: {file_name}")
            res = request_posts(solution, _act="image_rot_shift", _data=["image_dir", "file_name", "repeat"], _value=[image_dir, file_name, repeat])
            if(check_res(res)):
                output = eval(res.text)
                x = output[0]
                y = output[1]
                rot_angle = output[2]
                set_variable(solution, "shift_x", x)
                set_variable(solution, "shift_y", y)
                set_variable(solution, "rot_angle", rot_angle)
                log_meesage(f"[time:{time.time()-s_time}] Image shit x:{x} y:{y}, rot_angle:{rot_angle}")
                return solution.judge_pass()
            else:
                log_error(f"{self.__class__.__name__} Failed")
                return solution.judge_fail() #　エラーコードで分けて出力できるなら、変数の未定義とでreturnを変える

        except Exception as e:
            log_error(f"Error in {self.__class__.__name__}: {e}")
            return solution.judge_fail()