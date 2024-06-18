import tkinter as tk
# from tkinter import filed
import cv2
import argparse
import numpy as np
from PIL import Image, ImageTk, ImageOps, ImageDraw  # 画像データ用

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface

from scipy.spatial.transform import Rotation

from ur_controller import URController

import os
import sys
sys.path.append('../')

from agent import CFIL_ABN, CoarseToFineImitation


class Application(tk.Frame):
    def __init__(self, master=None, no_robot = False, ur_sim=False):
        super().__init__(master)

        self.master = master
        self.click_x = None
        self.click_y = None
        self.create_widgets()
        
        if ur_sim:
            self.wakeupRobot(config_file="configs/robot_config_sim.json")
        else:
            self.wakeupRobot()

        self.camera_setup(no_robot)
        
        self.disp_image()

    def create_widgets(self):
        self.width = 640
        self.height = 480

        self.canvas = tk.Canvas(self.master, width=self.width, height=self.height)
        self.canvas.grid(row=2, column=0, rowspan=2, columnspan=12)

        #クリックで座標取得
        self.canvas.bind("<ButtonPress-1>", self.ButtonPress)

        self.master.title("教示ソフト_デモ")

        self.bt_reset = tk.Button(self.master, text="教示をリセット", command=self.reset)
        self.bt_reset.grid(row=1, column=0)

        self.bt_reset = tk.Button(self.master, text="原点へ移動", command=self.move_home)
        self.bt_reset.grid(row=1, column=1)
        self.bt_robot_move = tk.Button(self.master, text="ロボットを移動", command=self.move_robot)
        self.bt_robot_move.grid(row=1, column=10)

        self.bt_data_collection = tk.Button(self.master, text="データ収集開始", command=self.collect_approach_traj)
        self.bt_data_collection.grid(row=5, column=10)
    
    def camera_setup(self, no_robot=False):
        # WIDTH = 640, HEIGHT = 480
        if no_robot:
            self.cam = Camera()
        else:
            self.cam = self.ur

    def disp_image(self):
        '''画像をCanvasに表示する'''

        # # フレーム画像の取得
        image = self.cam.get_img()
        if image is None:
            self.disp_id = self.after(100, self.disp_image)
            
        cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # NumPyのndarrayからPillowのImageへ変換
        pil_image = Image.fromarray(cv_image)

        # 画像のアスペクト比（縦横比）を崩さずに指定したサイズ（キャンバスのサイズ）全体に画像をリサイズする
        pil_image = ImageOps.pad(pil_image, (self.width, self.height))
        if self.click_x is not None and self.click_y is not None:
            draw_image = ImageDraw.Draw(pil_image)
            draw_image.rectangle(((self.click_x-5, self.click_y+5), (self.click_x+5, self.click_y-5)), fill=(0, 0, 255), outline='white',  width=2)
            
        # PIL.ImageからPhotoImageへ変換する
        self.photo_image = ImageTk.PhotoImage(image=pil_image)

        # 画像の描画
        self.canvas.create_image(
                self.width / 2,       # 画像表示位置(Canvasの中心)
                self.height / 2,                   
                image=self.photo_image  # 表示画像データ
                )

        # disp_image()を10msec後に実行する
        self.disp_id = self.after(100, self.disp_image)
    
    def reset(self):
        self.click_x = self.click_y = None


    def ButtonPress(self, event):        
        self.click_x = event.x
        self.click_y = event.y
       
        print("clicked at ", event.x, event.y)

        image = self.cam.get_img()
        # cv2.imwrite("img.png", image)

    def wakeupRobot(self, config_file="configs/robot_config.json"):
        self.ur = URController(config_file=config_file)

    #カメラ座標をロボット座標に変換
    # https://dev.classmethod.jp/articles/convert-coords-screen-to-space/
    def convert_c_T_r(self, target_in_camera):
        mtx   = np.load("../mtx.npy")
        g_t_c = np.load("../g_t_c.npy")
        g_R_c = np.load("../g_R_c.npy")
        print(g_t_c, g_R_c)
        g_R_c = Rotation.from_rotvec(g_R_c)
        gTc = np.r_[np.c_[g_R_c.as_matrix(), g_t_c], np.array([[0, 0, 0, 1]])]

        original_pose = self.ur.get_pose()

        target_s = target_in_camera

        self.ur.rtde_c.setTcp(np.concatenate([g_t_c, [0,0,0]], 0))
        print(target_s, self.ur.rtde_r.getActualTCPPose()[2])
        target_s_ = target_s*self.ur.rtde_r.getActualTCPPose()[2]
        print(target_s_)
        self.ur.rtde_c.setTcp([0,0,0.3185,0,0,0])
        target_c = np.dot(np.linalg.inv(mtx), target_s_)

        R = np.asarray([[1,0,0], [0,1,0], [0,0,1]])

        print(R.shape, target_c.shape)
        cTt = np.r_[np.c_[R, target_c.T], np.array([[0, 0, 0, 1]])]

        robot_pose = self.ur.rtde_r.getActualTCPPose()
        rot_bTg = Rotation.from_rotvec(robot_pose[3:])
        bTg = np.r_[np.c_[rot_bTg.as_matrix(), np.array(robot_pose[:3]).T], np.array([[0, 0, 0, 1]])]

        bTt = np.dot(np.dot(bTg, gTc),cTt)
        print(bTt)

        target = bTt[:-1,-1]
        target_in_robotbase = np.concatenate([target[:3], robot_pose[3:]], 0)
        target_in_robotbase[2] = original_pose[2]
        print(target_in_robotbase)

        return target_in_robotbase

    def move_robot(self):
        if self.click_x is None or self.click_y is None:
            return 
        target_in_camera = np.array([self.click_x, self.click_y, 1])
        target_in_robotbase = self.convert_c_T_r(target_in_camera)

        self.ur.moveL(target_in_robotbase)
    
    def move_home(self):
        
        self.ur.initialize()

    def collect_approach_traj(self):
        self.cfil = CFIL_ABN()
        self.cfil.bottleneck_pose = self.ur.get_pose()
        self.cfil.setupRobot(self.ur)
        self.cfil.collect_approach_traj(num=100)
        self.cfil.save_memory()

class Camera:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
    
    def get_img(self):
        ret, frame = self.cam.read()

        return frame

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_robot", action="store_true")
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()
    
    root = tk.Tk()
    app = Application(master=root, no_robot=args.no_robot, ur_sim=args.sim)
    app.mainloop()