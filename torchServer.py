import argparse
import os
print(os.getcwd())
import logging
import cv2
import numpy as np
from datetime import datetime
import time
from flask import Flask, request
import torch

import json

import pathlib

from CFIL_for_NIP.memory import ApproachMemory
from CFIL_for_NIP.network import ABN128, ABN256
from scipy.ndimage import maximum_filter
from scipy.spatial.transform import Rotation as R

# import pytransform3d.rotations as pyrot
# import pytransform3d.transformations as pytr
# from pytransform3d.transform_manager import TransformManager

# from calib.rs_utils import RealSense

from logger import setup_logger


cfil_agent = None
position_detector = None

# Flask設定
HOST = "192.168.0.3" #津の設定
# HOST = "10.178.64.66" #debug設定

PORT = 5000
app = Flask(__name__)

# ロガー設定
logging = setup_logger("torchServer")

def log_meesage(message):
    print(message)
    logging.info(message)

def log_error(message):
    print(message)
    logging.error(message)

@app.route("/initialize", methods=["POST"])
def initialise():
    global cfil_agent
    ret = cfil_agent.initialize()
    if(ret):
        log_meesage("CFIL Agent Initialized")
        return "Success"
    else:
        log_error("CFIL Agent Initialization Failed")
        return "False"
    
@app.route("/loadTrainedModel", methods=["POST"])
def loadTrainedModel():
    global cfil_agent
    file_path = request.form["file_path"]
    task_name = request.form["task_name"]
    epoch = request.form["epoch"]
    ret = cfil_agent.loadTrainedModel(file_path, task_name, epoch)
    if(ret):
        log_meesage("Trained CFIL Model Loaded")
        return "Success"
    else:
        log_error("Trained CFIL Model Loading Failed")
        return "False"

@app.route("/loadSAMModel", methods=["POST"])
def loadSAMModel():
    global cfil_agent
    image_save_path = request.form["image_save_path"]
    model_path = request.form["model_path"]
    sam_type = request.form["sam_type"]
    ret = cfil_agent.loadSAMModel(image_save_path, model_path, sam_type)
    if(ret):
        log_meesage("SAM Model Loaded")
        return "Success"
    else:
        log_error("SAM Model Loading Failed")
        return "False"
    
@app.route("/estimate", methods=["POST"])
def Estimate():
    global cfil_agent
    image_path = request.form["image_path"]
    ret = str(cfil_agent.estimate_from_image(image_path))
    log_meesage("Estimation Completed {}".format(ret))
    return ret

@app.route("/loadSAM_f_Model", methods=["POST"])
def loadSAM_f_Model():
    global cfil_agent
    image_save_path = request.form["image_save_path"]
    file_path = request.form["file_path"]
    task_name = request.form["task_name"]
    ret = cfil_agent.loadSAM_f_Model(image_save_path, file_path, task_name)
    if(ret):
        log_meesage("SAM_f Model Loaded")
        return "Success"
    else:
        log_error("SAM_f Model Loading Failed")
        return "False"
    
@app.route("/estimate_f", methods=["POST"])
def Estimate_f():
    global cfil_agent
    image_path = request.form["image_path"]
    ret = str(cfil_agent.estimate_from_image_f(image_path))
    log_meesage("Estimation Completed {}".format(ret))
    return ret

###################################################################
# Position Detector
###################################################################
# @app.route("/initialize_PD", methods=["POST"])
# def initialise_pd():
#     global cfil_agent
#     ret = cfil_agent.initialize_positionDetector()
#     if(ret):
#         log_meesage("Position Detector Initialized")
#         return "Success"
#     else:
#         log_error("Position Detector Initialization Failed")
#         return "False"

@app.route("/get_tray_position_force", methods=["POST"])
def get_tray_position_force():
    global cfil_agent
    ret = str(cfil_agent.get_tray_position_force())
    log_meesage("Tray Positions Detected {}".format(ret))
    return ret

@app.route("/get_positions_force", methods=["POST"])
def get_positions_force():
    global cfil_agent
    ret = str(cfil_agent.get_positions_force())
    log_meesage("Positions Detected {}".format(ret))
    return ret

@app.route("/get_positions", methods=["POST"])
def get_positions():
    global cfil_agent
    ret = str(cfil_agent.get_positions())
    log_meesage("Positions Detected {}".format(ret))
    return ret

@app.route("/image_rot_shift", methods=["POST"])
def image_rot_shift():
    global cfil_agent
    image_dir = request.form["image_dir"]
    file_name = request.form["file_name"]
    repeat = request.form["repeat"]
    ret = cfil_agent.image_rot_shift(image_dir, file_name, float(repeat))
    log_meesage(f"Image Rot Shifted: x={ret[0]}, y={ret[1]}, rot_angle={ret[2]}")
    return str(ret)

class Agent:
    def __init__(self):
        pass

    def initialize(self):
        try:
            self.memory_size = 5e4
            self.image_size = 256

            self.device = "cuda" if torch.cuda.is_available() else "cpu" 
            log_meesage("Device: {}".format(self.device))
            if self.image_size == 128:
                self.cfil = ABN128(output_size=3)
            elif self.image_size == 256:
                self.cfil = ABN256(output_size=3)
            self.cfil.to(self.device)
            log_meesage("CFIL Model Initialized")

            self.approach_memory = ApproachMemory(self.memory_size, self.device)
            log_meesage("Approach Memory Initialized")

            self.use_sam = False

            self.train_data_file = None

            self.task_name = None

            return True
        except Exception as e:
            log_error("{} : {}".format(type(e), e))
            return False
             
    def loadTrainedModel(self, file_path=None, task_name=None, epoch=None):
        try:
            model_path = None
            if self.train_data_file is None:
                print(file_path, task_name, f"approach_model_{epoch}.pth")
                model_path = os.path.join(file_path, task_name, "abn", f"approach_model_{epoch}.pth")
            else:
                model_path = os.path.join(*["CFIL_for_NIP", "train_data", self.train_data_file, "abn", f"approach_model_{epoch}.pth"])

            log_meesage(model_path)
            self.cfil.load_state_dict(torch.load(model_path, map_location=self.device))
            
            return True
        except Exception as e:
            log_error(f"train_data_file: {self.train_data_file}, file_path: {file_path}, task_name: {task_name}, epoch: {epoch}")
            log_error("{} : {}".format(type(e), e))
            return False
        
    def loadSAMModel(self,image_path="", model_path="", sam_type="vit_b"):
        try:
            from perSam import PerSAM
            print(image_path, model_path)
            if self.train_data_file is None:
                self.output_path = os.path.join(*image_path.split("\\")[:-2], "output_images")
            else:
                self.output_path = os.path.join("CFIL_for_NIP", "train_data", self.train_data_file, "test", format(datetime.date.today(), '%Y%m%d'), "output_images")
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            self.per_sam = PerSAM(
                        # annotation_path="sam\\ref", 
                        annotation_path=os.path.join(model_path, "ref"), 
                        # output_path=os.path.join(image_path, "masked_images"))
                        output_path=self.output_path)
            
            self.per_sam.loadSAM(sam_type=sam_type)
            self.use_sam = True
            return True
        except Exception as e:
            log_error("{} : {}".format(type(e), e))
            return False

    def estimate_from_image(self, image_path):
        image = cv2.imread(image_path+".jpg")
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        image_num = image_path.split("\\")[-1].split("_")[0]
        if self.use_sam:
            masks, best_idx, topk_xy, topk_label = self.per_sam.executePerSAM(image, show_heatmap=False)
            image = self.per_sam.save_masked_image(masks[best_idx], image, image_num+"_mask.jpg")
            # self.per_sam.save_heatmap(image_num+"_similarity.jpg")
            # image = self.per_sam.save_masked_image(masks[best_idx], image, image_path.split("\\")[-1]+".jpg")
            # self.per_sam.save_heatmap(image_path.split("\\")[-1]+"_similarity.jpg")


        image = np.transpose(image, [2, 0, 1])
        image_tensor = torch.ByteTensor(image).to(self.device).float() / 255.
        image_tensor = torch.unsqueeze(image_tensor, 0)
        self.cfil.eval()
        with torch.no_grad():
            # appraoch
            output_tensor, _, att = self.cfil(image_tensor)
            output = output_tensor.to('cpu').detach().numpy().copy()
            self.save_attention_fig(image_tensor, att, image_num)

        return output[0].tolist()
    
    def loadSAM_f_Model(self, result_path, file_path, task_name):
        try:
            from perSam import PerSAM
            print(f"result_path: {result_path}, file_path: {file_path}, task_name: {task_name}")
            self.task_name = task_name
            self.output_path = os.path.join(file_path, task_name, "test", format(datetime.today(), '%Y%m%d'), "output_images")
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            self.per_sam = PerSAM(
                        # annotation_path="sam\\ref", 
                        annotation_path=os.path.join(file_path, "ref"), 
                        output_path=os.path.join(result_path, "masked_images"))
            if os.path.exists(os.path.join(file_path, "weight.npy")):
                wight_np = np.load(os.path.join(file_path, "weight.npy"))
                self.per_sam.loadSAM_f(weight=wight_np)
            else:
                weight_np = self.per_sam.loadSAM_f()
                np.save(os.path.join(file_path, "weight.npy"), weight_np)
            self.use_sam = True
            return True
        except Exception as e:
            log_error("{} : {}".format(type(e), e))
            return False
        
    def estimate_from_image_f(self, image_path):
        try:
            image = cv2.imread(image_path+".jpg")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, None,fx=0.1,fy=0.1)
            if self.use_sam:
                heatmap = False
                masks, best_idx, topk_xy, topk_label = self.per_sam.executePerSAM_f(image, show_heatmap=heatmap)
                image = self.per_sam.save_masked_image(masks[best_idx], image, image_path.split("\\")[-1]+".jpg")
                if heatmap:
                    self.per_sam.save_heatmap(image_path.split("\\")[-1]+"_similarity.jpg")
                if "mask_image_only" in self.task_name:
                    image = np.zeros((masks[best_idx].shape[0], masks[best_idx].shape[1], 3), dtype=np.uint8)
                    image[masks[best_idx], :] = np.array([[0, 0, 128]])

            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            image = np.transpose(image, [2, 0, 1])
            image_tensor = torch.ByteTensor(image).to(self.device).float() / 255.
            image_tensor = torch.unsqueeze(image_tensor, 0)
            self.cfil.eval()
            with torch.no_grad():
                # appraoch
                output_tensor, _, att = self.cfil(image_tensor)
                output = output_tensor.to('cpu').detach().numpy().copy()
                self.save_attention_fig(image_tensor, att, image_path.split("\\")[-1])
            return output[0].tolist()
    
        except Exception as e:
            log_error("{} : {}".format(type(e), e))

            return False    

    def save_attention_fig(self, inputs, attention, image_num):
        def min_max(x, axis=None):
            min = x.min(axis=axis, keepdims=True)
            max = x.max(axis=axis, keepdims=True)
            result = (x-min)/(max-min)
            return result
        try:
            c_att = attention.data.cpu()
            c_att = c_att.numpy()
            d_inputs = inputs.data.cpu()
            d_inputs = d_inputs.numpy()
            in_b, in_c, in_y, in_x = inputs.shape
            for item_img, item_att in zip(d_inputs, c_att):
                v_img = item_img.transpose((1,2,0))* 255
                resize_att = cv2.resize(item_att[0], (in_x, in_y))
                resize_att = min_max(resize_att)* 255
                vis_map = cv2.cvtColor(resize_att, cv2.COLOR_GRAY2BGR)
                jet_map = cv2.applyColorMap(vis_map.astype(np.uint8), cv2.COLORMAP_JET)
                v_img = v_img.astype(np.uint8)
                jet_map = cv2.addWeighted(v_img, 0.5, jet_map, 0.5, 0)
                cv2.imwrite(os.path.join(self.output_path, image_num + "_attntion.jpg"), cv2.vconcat([v_img, jet_map]))

        except Exception as e:
            log_error("{} : {}".format(type(e), e))
            return False

###############################################################
# Position Detector
###############################################################

    # def initialize_positionDetector(self):
    #     try:
    #         self.camera_id = 0
    #         self.tm = TransformManager()
    #         self.cam = None
        
    #         # nishikadoma
    #         # crop_settings = [{"crop_size_x": 240, "crop_size_y": 240, "crop_center_x": 320, "crop_center_y": 240}]
    #         # crop_settings = [{'crop_size_x': 273, 'crop_size_y': 212, 'crop_center_x': 338, 'crop_center_y': 212}]
    #         # crop_settings = [{'crop_size_x': 251, 'crop_size_y': 194, 'crop_center_x': 337, 'crop_center_y': 210}]

    #         # with open("calib\\camera_info\\crop_settings.json", "r") as f:
    #         #     crop_json = json.load(f)
    #         #     crop_settings = crop_json["crop_settings"]
            
    #         with open(self.crop_settings_path, "r") as f:
    #             crop_json = json.load(f)
    #             crop_settings = crop_json["crop_settings"]

    #         # crop_settings = [{'crop_size_x': 264, 'crop_size_y': 191, 'crop_center_x': 338, 'crop_center_y': 212}]
    #         # D405  tsu
    #         # crop_settings = [{"crop_size": 260, "crop_center_x": 350, "crop_center_y": 240}]
    #         log_meesage(crop_settings)
    #         self.cam = RealSense(crop_settings=crop_settings)

    #         # self.ratio = np.load("calib\\camera_info\\ratio.npy")
    #         self.ratio = np.load(self.ratio_path)
    #         # self.center_pixels = [self.cam.crop_settings[self.camera_id]["crop_center_x"], self.cam.crop_settings[self.camera_id]["crop_center_y"]]
    #         # diff_center = crop_settings[self.camera_id]["crop_center_x"] - 320 
    #         # self.center_pixels = [crop_settings[self.camera_id]["crop_size_x"]//2 - diff_center, (crop_settings[self.camera_id]["crop_size_y"])//2]
    #         self.center_pixels = [crop_settings[self.camera_id]["crop_size_x"]//2, crop_settings[self.camera_id]["crop_size_y"]//2]
            
    #         log_meesage(f"center pixels {self.center_pixels}")

    #         #nishikadoma
    #         # self.center_position = [-0.0809, -0.470]
    #         # self.center_position = [-0.010, -0.535]
    #         # self.center_position = [-0.03, -0.5]
    #         # self.center_position = [0.004998829998830001, -0.4519421305709614]
    #         self.center_position = self.center_position


    #         #tsu
    #         # self.center_position = [-0.065, -0.470]

    #         self.camera_pose = np.load("calib/camera_info/camera_pose.npy")
            
    #         self.register_pose(self.camera_pose[:3], self.camera_pose[3:], "base", "cam")

    #         self.per_sam.loadPositionDetector()

    #         return True

    #     except Exception as e:
    #         log_error("{} : {}".format(type(e), e))
    #         return False

    def get_tray_position_force(self):
        # 点p0に一番近い点を取得
        def func_search_neighbourhood(p0, ps):
            L = np.array([])
            for i in range(ps.shape[0]):
                norm = np.sqrt( (ps[i][0] - p0[0])*(ps[i][0] - p0[0]) +
                                (ps[i][1] - p0[1])*(ps[i][1] - p0[1]) )
                L = np.append(L, norm)
            return np.argmin(L) ,ps[np.argmin(L)], np.min(L)


        positions = self.get_positions_force()
        positions_X = positions[0]
        positions_Y = positions[1]
        N = positions[2]
        tray = np.load('../python_scripts/coordinate.npz')
        coordinate1 = tray['tray1']
        coordinate2 = tray['tray2']
        tray_position = []
        tray_num = []
        for x,y in zip(positions_X, positions_Y):
            target_coor =  np.array([x,y])
            idx1, _ , minL1 = func_search_neighbourhood(target_coor, coordinate1)
            idx2, _ , minL2 = func_search_neighbourhood(target_coor, coordinate2)
            if(minL1 < minL2):
                tray_position.append(idx1)
                tray_num.append(1)
            else:
                tray_position.append(idx2)
                tray_num.append(2)
        
        return [tray_position, tray_num, len(tray_position)]
        
    def get_positions_force(self):
        color_images, depth_images, _, _ = self.cam.get_image(crop=True)
        # peaks_pixels = self.per_sam.getPeaks(color_images[0], filter_size=20, order=0.65, save_sim=True)
        peaks_pixels = self.per_sam.getObjects(color_images[0], filter_size=20, order=0.7, save_sim=True)
        positions_X = []
        positions_Y = []
        # xy の順番に注意
        for i in range(len(peaks_pixels[0])):
            X = self.center_position[0] + (peaks_pixels[1][i] - self.center_pixels[0])*self.ratio[0]
            Y = self.center_position[1] - (peaks_pixels[0][i] - self.center_pixels[1])*self.ratio[1] 
            positions_X.append(X*1000) # m -> mm
            positions_Y.append(Y*1000) # m -> mm

        return [positions_X, positions_Y, len(positions_X)]
    
    def get_positions(self):
        color_images, depth_images, _, _ = self.cam.get_image(crop=True)
        frames = self.cam.get_frames()
        peaks_pixels = self.per_sam.getPeaks(color_images[0], filter_size=100, order=0.7, save_sim=True)
        positions_X = []
        positions_Y = []
        # xy の順番に注意
        for i in range(len(peaks_pixels[0])):
            depth_frame = frames[0].as_depth_frame()
            depth = depth_frame.get_distance(peaks_pixels[1][i], peaks_pixels[0][i])
            tvec = self.cam.get_point_from_pixel(peaks_pixels[1][i], peaks_pixels[0][i], depth)
            rvec = R.from_euler("XYZ", [0, 0, 180], degrees=True).as_rotvec()
            # print(tvec, rvec)
            self.register_pose(tvec, rvec, "cam", "target")
            pose = self.get_pose("base", "target")
            positions_X.append(pose[0])
            positions_Y.append(pose[1])

        return [positions_X, positions_Y, len(positions_X)]
    
    # def register_pose(self, tvec, rvec, source, target):
    #     source_T_target = pytr.transform_from(
    #         pyrot.matrix_from_compact_axis_angle(rvec), tvec)
    #     self.tm.add_transform(target, source, source_T_target)

    def remove_pose(self, source, target):
        self.tm.remove_transform(target, source)

    def get_pose(self, source, target):
        return self.transform_matrix_to_vector(self.tm.get_transform(target, source))

    def get_matrix(self, source, target):
        return self.tm.get_transform(target, source)
    
    def transform_matrix_to_vector(self, transform):
        tvec = transform[:3, 3]
        rvec = R.from_matrix(transform[:3, :3]).as_rotvec()
        # 回転ベクトルの絶対値を180°以下にする。
        while True:
            dr = np.linalg.norm(rvec)
            if dr > np.radians(180):
                rvec = rvec * (dr - np.radians(360)) / dr
            else:
                break
        return np.hstack([tvec, rvec])
    
    def test_PD(self):
        self.loadSAMModel()
        self.initialize_positionDetector()
        print(self.get_positions_force())
        # self.get_positions()

    def initialize_all(self):
        self.initialize()
        log_meesage("initialized")
        self.loadJson()
        log_meesage("json loaded")
        self.loadTrainedModel(file_path=self.file_path, task_name=self.task_name, epoch=self.epoch)
        log_meesage("trained model loaded")
        self.loadSAM_f_Model(result_path=self.image_path, file_path=self.file_path, task_name=self.task_name)
        log_meesage("sam_f model loaded")
        self.initialize_positionDetector()
        log_meesage("position detector initialized")
        return True

    def loadJson(self, path="config_server.json"):
        with open(path, "r") as f:
            json_dict = json.load(f)

        self.file_path = json_dict["file_path"]

        self.task_name = json_dict["sam_f"]["task_name"]
        self.epoch = json_dict["sam_f"]["epoch"]

        self.image_path = json_dict["sam_f"]["image_path"]

        self.crop_settings_path = json_dict["position_detector"]["crop_settings_path"]
        self.ratio_path = json_dict["position_detector"]["ratio_path"]
        self.center_position = json_dict["position_detector"]["center_position"]

###############################################################
# Segmentation
###############################################################
    def contours(self, image, rate=1):
        # グレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ２値化
        retval, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 輪郭を抽出
        #   contours : [領域][Point No][0][x=0, y=1]
        #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
        #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # cv2.imwrite(os.path.join(self.output_path, "contours.jpg"), cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 10))
        # print(os.path.join(self.output_path, "contours.jpg"))
        
        # 各輪郭に対する処理
        x,y = None, None
        box = None
        points = None
        if len(contours) == 0:
            log_meesage("No contours found")
            return (None, None), None, None, None
        
        for i in range(0, len(contours)):

            # 輪郭の領域を計算
            area = cv2.contourArea(contours[i])
            cv2.imwrite(os.path.join(self.output_path, f"contours_{i}_{time.time()}.jpg"), cv2.drawContours(image.copy(), contours[i], -1, (0, 255, 0), 10))

            
            # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
            ref_mask_area = self.per_sam.getRefMaskArea()
            # log_meesage(f"Contour {i}: area={area}")
            if area < ref_mask_area*0.9 or ref_mask_area*1.1 < area:
                continue
            x, y, w, h = cv2.boundingRect(contours[i])
            box = cv2.minAreaRect(contours[i])
            points = cv2.boxPoints(box)
            break

        return (x,y), box, points, contours[i]

    def image_rot_shift(self, image_dir, file_name, repeat=1):
        rate = 0.2  # リサイズ率
        image_path = os.path.join(image_dir, file_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, None, fx=rate, fy=rate)

        results = []
        pre_contour = []
        for i in range(int(repeat)):
            # masks, best_idx, topk_xy, topk_label = self.per_sam.executePerSAM(image, show_heatmap=False)
            s_time = time.time()
            topk_xy, topk_label, sim = self.per_sam.getTopKPoints(image)
            print(f"getTopKPoints time: {time.time() - s_time:.3f} sec")
            # 前回のtopk_xyと重なっているなら除外
            inside_pre_contour = False
            for pc in pre_contour:
                if cv2.pointPolygonTest(pc, (int(topk_xy[0][0]),int(topk_xy[0][1])), False) > 0:
                    log_meesage("inside pre_contour")
                    inside_pre_contour = True
                    break

            if inside_pre_contour:
                log_meesage("Skipping PerSAM due to inside pre_contour")
                continue
            
            masks, best_idx = self.per_sam.getSAMMask(topk_xy, topk_label)
            # masks, best_idx = self.per_sam.getSAMMask2(topk_xy, topk_label, sim)
            if masks is None:
                return [[0], [0], [0]]
            print(f"getSAMMask time: {time.time() - s_time:.3f} sec")
            final_mask = masks[best_idx]
            mask_image = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
            mask_image[final_mask, :] = np.array([[0, 0, 128]])
            (x, y), box, points, contour = self.contours(mask_image, rate)
            log_meesage(f"Contour found: {x}, {y}, box: {box}, points: {points}")

            if x is None and y is None:
                log_meesage("No contours found after PerSAM")
                # return [[0], [0], [0]]
                continue           
            
            # pre_topk_xy = (topk_xy[0][0], topk_xy[0][1])
            pre_contour.append(contour)
            
            rot_angle = box[2]
            if rot_angle > 45:
                rot_angle = rot_angle - 90

            center = np.mean(points, axis=0)

            # 2. 各点を (x, y) として、左上・右上・右下・左下に分類
            for p in points:
                x, y = p
                if x < center[0] and y < center[1]:
                    break

            image[final_mask!=0] = [71,89,144]
            output_image = cv2.drawMarker(image.copy(), topk_xy[0], (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=10)
            cv2.imwrite(os.path.join(image_dir, f"{file_name}_mask_{i}.jpg"), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

            results.append([int(x), x/rate, y/rate, rot_angle])

        results.sort()
        results = np.array(results)
        shit_x = results[:,1].tolist()
        shit_y = results[:,2].tolist()
        angles = results[:,3].tolist()

        return [shit_x, shit_y, angles]
        # return [x/rate, y/rate, rot_angle]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action='store_true')
    parser.add_argument("--production", "-p", action='store_true')
    args = parser.parse_args()

    try:
        cfil_agent = Agent()
        if args.debug: # デバッグモード
            cfil_agent.test_PD()
        elif args.production: # 本番モード
            cfil_agent.initialize_all()
            app.run(debug=False, port=PORT, host=HOST)
        else:
            cfil_agent.initialize()
            app.run(debug=False, port=PORT, host=HOST)
    except Exception as e:
        log_error("{} : {}".format(type(e), e))