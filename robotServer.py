import os
import time
import json
import logging
from flask import Flask, request
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
from dashboard_client import DashboardClient
from robotiq_epick import RobotiqEpick
from robotiq_rtu import RobotiqRtu
from vacuum_gripper import VacuumGripper
import numpy as np

# パラメータ設定
LOGFILENAME = "robotServer"
LOGFILEROOT = "logs"
LOGLEVEL = logging.DEBUG
# ROBOTPARAMS = os.path.dirname(__file__) + "/robot_config_kato.json"
ROBOTPARAMS = os.path.dirname(__file__) + "/robot_config_ew.json"
json_file = open(ROBOTPARAMS, "r")
json_dict = json.load(json_file)
robot_ip = json_dict["robot_ip"]
HOST = json_dict["server_ip"]
use_gripper = json_dict["use_gripper"]
PORT = 5000

# ログ設定
os.makedirs(LOGFILEROOT, exist_ok=True)
log_file_path = LOGFILEROOT + "/" + LOGFILENAME + ".log"
log_format = "%(asctime)s %(levelname)s:%(message)s"
logging.basicConfig(filename=log_file_path, level=LOGLEVEL, format=log_format)

# ロボット設定
global robot_clients
robot_clients = {}

# Flask設定
app = Flask(__name__)

# 吸着ON
@app.route("/vac_on", methods=["POST"])
def vac_on():
    global robot_clients
    target_ip = request.form["target_ip"]
    robot = robot_clients[target_ip]
    ret = robot.vac_on()
    print("Vacuum on request")
    logging.info("Vacuum on request")
    if(ret):
        return "Success"
    else:
        return "False"

# 吸着OFF
@app.route("/vac_off", methods=["POST"])
def vac_off():
    global robot_clients
    target_ip = request.form["target_ip"]
    robot = robot_clients[target_ip]
    ret = robot.vac_off()
    print("Vacuum off request")
    logging.info("Vacuum off request")
    if(ret):
        return "Success"
    else:
        return "False"

# get pressure
@app.route("/get_pressure", methods=["POST"])
def get_pressure():
    global robot_clients
    target_ip = request.form["target_ip"]
    robot = robot_clients[target_ip]
    ret = robot.get_pressure()
    print("get pressure", ret, type(ret))
    logging.info("get pressure request")
    return str(ret)


# Move_L
@app.route("/move_L", methods=["POST"])
def move_L():
    global robot_clients
    target_ip = request.form["target_ip"]
    robot = robot_clients[target_ip]
    dst = eval(request.form["dst"])
    ret = robot.move_L(dst)
    print("moveL to {} request".format(dst))
    logging.info("moveL to {} request".format(dst))
    if ret:
        return "Success"
    else:
        robot.stopScript()
        return "Failed"

# Move_J
@app.route("/move_J", methods=["POST"])
def move_J():
    global robot_clients
    target_ip = request.form["target_ip"]
    robot = robot_clients[target_ip]
    dst_angles_deg = eval(request.form["dst_angles"])
    dst_angles_rad = np.deg2rad(dst_angles_deg)
    ret = robot.move_J(dst_angles_rad)
    print("moveJ to {} request (rad)".format(dst_angles_rad))
    logging.info("moveJ to {} request  (rad)".format(dst_angles_rad))
    if ret:
        return "Success"
    else:
        return "False"

# get current pose
@app.route("/get_current_pose", methods=["POST"])
def get_current_pose():
    global robot_clients
    target_ip = request.form["target_ip"]
    robot = robot_clients[target_ip]
    ret = robot.get_current_pose()
    print("get current pose", ret, type(ret))
    logging.info("get current pose request")
    return str(ret)

# get current angles
@app.route("/get_current_angles", methods=["POST"])
def get_current_angles():
    global robot_clients
    target_ip = request.form["target_ip"]
    robot = robot_clients[target_ip]
    ret_rad = robot.get_current_angles()
    print("get current angles rad", ret_rad, type(ret_rad))
    ret_deg = np.rad2deg(ret_rad).tolist()
    print("get current angles deg", ret_deg, type(ret_deg))
    logging.info("get current angles request")
    return str(ret_deg)

# set new tcp pose
@app.route("/set_TCP", methods=["POST"])
def set_tcp():
    global robot_clients
    target_ip = request.form["target_ip"]
    robot = robot_clients[target_ip]
    tcp_pose = eval(request.form["tcp_pose"])
    ret = robot.set_tcp(tcp_pose)
    logging.info("set tcp pose")
    if(ret):
        return "Success"
    else:
        return "False"
# サーバプログラムとロボットの接続
@app.route("/connect_robot", methods=["POST"])
def connect():
    global robot_clients
    target_ip = request.form["target_ip"]
    robot = robot_clients[target_ip]
    ret = robot.connect()
    print("connect on request")
    logging.info("connect on request")
    if(ret):
        return "Success"
    else:
        return "False"

# サーバプログラムとロボットの接続解除
@app.route("/disconnect_robot", methods=["POST"])
def disconnect():
    global robot_clients
    target_ip = request.form["target_ip"]
    robot = robot_clients[target_ip]
    robot.disconnect()
    print("disconnect on request")
    logging.info("disconnect on request")
    return "Success"


# アームの軌道とブレーキのリリース
@app.route("/wakeup_robot", methods=["POST"])
def wakeup():
    global robot_clients
    target_ip = request.form["target_ip"]
    robot = robot_clients[target_ip]
    ret = robot.wakeup()
    print("wakeup on request")
    logging.info("wakeup on request")
    if ret:
        return "Success"
    else:
        return "False"

# get safety mode
@app.route("/get_safety_mode", methods=["POST"])
def get_safety_mode():
    global robot_clients
    target_ip = request.form["target_ip"]
    robot = robot_clients[target_ip]
    ret = robot.get_safety_mode()
    print("get safety mode", ret, type(ret))
    logging.info("get pressure safety mode")
    return str(ret)


# ロボットインスタンス生成
class Robot:
    # 初期化処理
    def __init__(self, params):

        # ロボット設定
        self.robot_ip = robot_ip
        self.use_gripper = use_gripper
        
        self.dashboard = None
        self.rtde_c = None
        self.rtde_r = None
        self.gripper = None
        self.connect()

        #コントロールボックスのデジタルI/Oの設定
        # self.valve_id = 0 #I/OのID
        # #
        # self.ctrlSolenoidValve(self.valve_id, True)
        # self.ctrlSolenoidValve(self.valve_id, False)

        print("Robot {} initialized".format(self.robot_ip))
        logging.info("Robot {} initialized".format(self.robot_ip))    

    def __del__(self):
        self.disconnect()

    # 通信開始
    def connect(self):
        # try:
        self.dashboard = DashboardClient(robot_ip)
        self.dashboard.connect()
        if(self.dashboard.isConnected()):
            self.dashboard.powerOn()
            self.dashboard.brakeRelease()
            self.dashboard.unlockProtectiveStop()
        self.rtde_c = RTDEControlInterface(self.robot_ip)
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)
        self.io = RTDEIOInterface(self.robot_ip)
        self.gripper = None
        if self.use_gripper == "socket": # EPcikをURのフランジに接続するとき
            self.gripper = RobotiqEpick()
            self.gripper.connect(self.robot_ip, 63352)
            self.gripper.activate()
        elif self.use_gripper == "rtu": # EPickをCOMポートでPCに接続するとき
            self.gripper = RobotiqRtu()
            self.gripper.connect("COM7")
            self.gripper.activate()
        elif self.use_gripper == "ejector": # 圧縮空気を使ってエジェクターから真空吸着するとき
            self.gripper = VacuumGripper(_io=self.io, _rtde_r=self.rtde_r)
        else:
            pass
    
        return True
                
        # except Exception as e:
        #     print(type(e), e)
        #     return False


    # 通信終了
    def disconnect(self):
        try:
            if self.rtde_c is not None:
                self.rtde_c.disconnect()
            if self.rtde_r is not None:
                self.rtde_r.disconnect()
            if self.use_gripper:
                self.gripper.disconnect()
            return True
        
        except Exception as e:
            print(type(e), e)
            return False

    # 吸着ON
    def vac_on(self):
        if self.use_gripper:
            self.gripper.grip_off()
            self.gripper.grip_on()
            time.sleep(1)
            print("Vacuum on done")
            logging.info("Vacuum on done")
            return self.gripper.getPressure()

    # 吸着OFF
    def vac_off(self):
        if self.use_gripper:
            self.gripper.grip_off()
            print("Vacuum off done")
            logging.info("Vacuum off done")
    
    # 吸着圧取得
    def get_pressure(self):
        if self.use_gripper:
            pressure = self.gripper.getPressure()
            print("get pressure")
            logging.info("get pressure")
            return pressure

    # Move_L
    def move_L(self, dst):
        print("moveL to {} done".format(dst))
        logging.info("moveL to {} done".format(dst))
        ret = self.rtde_c.moveL(dst, speed=0.15, acceleration=0.8)
        return ret
    
    # Move_J
    def move_J(self, dst_angle):
        print("moveJ to {} done".format(dst_angle))
        logging.info("moveJ to {} done".format(dst_angle))
        ret = self.rtde_c.moveJ(dst_angle, speed = 1.05, acceleration = 1.0)
        return ret

    def stopScript(self):
        print("Stop script")
        logging.info("Stop script")
        ret = self.rtde_c.stopScript()
        return ret
    
    # get current pose
    def get_current_pose(self):
        current_pose = self.rtde_r.getActualTCPPose()
        return current_pose

    # get current angles
    def get_current_angles(self):
        current_angles_rad = self.rtde_r.getActualQ()
        return current_angles_rad
    
    def set_tcp(self, tcp_pose):
        self.rtde_c.setTcp(tcp_pose)

    def ctrlSolenoidValve(self, valve_id, val):
        self.io.setStandardDigitalOut(valve_id,val)
    
    def wakeup(self):
        print(self.rtde_r.getSafetyMode())
        print("Robot status", self.rtde_r.getRobotStatus())
        if(self.rtde_r.getRobotStatus()==3):
            return True
        
        if(self.dashboard.isConnected()):
            self.dashboard.closeSafetyPopup()
            
            if(self.rtde_r.getSafetyMode() == 9):
                self.dashboard.restartSafety()
                time.sleep(3)
                self.dashboard.powerOn()
                time.sleep(3)
                self.dashboard.brakeRelease()
                self.dashboard.unlockProtectiveStop()
                self.disconnect()
                self.rtde_c.reconnect()
                self.rtde_r.reconnect()
            elif(self.rtde_r.getSafetyMode() == 7):
                return False
            else:
                self.dashboard.powerOn()
                self.dashboard.brakeRelease()
                self.dashboard.unlockProtectiveStop()
                self.disconnect()
                self.rtde_c.reconnect()
                self.rtde_r.reconnect()
            while(self.rtde_r.getRobotStatus()!=3):
                print(self.rtde_r.getRobotStatus())
                self.disconnect()
                self.rtde_c.reconnect()
                self.rtde_r.reconnect()
                pass
            return True
        else:
            return False
        
    def get_safety_mode(self):
        return self.rtde_r.getSafetyMode()    

if __name__ == "__main__":
    robot = Robot(ROBOTPARAMS)
    robot_clients[robot.robot_ip] = robot

    app.run(debug=False, port=PORT, host=HOST)
