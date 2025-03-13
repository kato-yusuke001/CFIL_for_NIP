import argparse
import os
print(os.getcwd())
import logging
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request
import torch

import json

from CFIL_for_NIP.memory import ApproachMemory
from CFIL_for_NIP.network import ABN128, ABN256
from scipy.ndimage import maximum_filter
from scipy.spatial.transform import Rotation as R

import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
from pytransform3d.transform_manager import TransformManager


from calib.rs_utils import RealSense

from logger import setup_logger


cfil_agent = None
position_detector = None

# Flask設定
HOST = "192.168.11.3" #津の設定
# HOST = "192.168.11.54" #西門真設定
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
    ret = cfil_agent.loadSAMModel(image_save_path, model_path)
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
@app.route("/initialize_PD", methods=["POST"])
def initialise_pd():
    global cfil_agent
    ret = cfil_agent.initialize_positionDetector()
    if(ret):
        log_meesage("Position Detector Initialized")
        return "Success"
    else:
        log_error("Position Detector Initialization Failed")
        return "False"

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
        
    def loadSAMModel(self,image_path="CFIL_for_NIP\\train_data\\20241025_151158_245\\test_pd", model_path="CFIL_for_NIP\\train_data\\20241025_151158_245"):
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
            
            self.per_sam.loadSAM()
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
    
    def loadSAM_f_Model(self,image_path, file_path, task_name):
        try:
            from perSam import PerSAM
            print(f"image_path: {image_path}, file_path: {file_path}, task_name: {task_name}")
            self.task_name = task_name
            self.output_path = os.path.join(file_path, task_name, "test", format(datetime.today(), '%Y%m%d'), "output_images")
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            self.per_sam = PerSAM(
                        # annotation_path="sam\\ref", 
                        annotation_path=os.path.join(file_path, "ref"), 
                        output_path=os.path.join(image_path, "masked_images"))
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

    def initialize_positionDetector(self):
        try:
            self.camera_id = 0
            self.tm = TransformManager()
            self.cam = None
        
            # nishikadoma
            # crop_settings = [{"crop_size_x": 240, "crop_size_y": 240, "crop_center_x": 320, "crop_center_y": 240}]
            # crop_settings = [{'crop_size_x': 273, 'crop_size_y': 212, 'crop_center_x': 338, 'crop_center_y': 212}]
            # crop_settings = [{'crop_size_x': 251, 'crop_size_y': 194, 'crop_center_x': 337, 'crop_center_y': 210}]

            with open("calib/camera_info/crop_settings.json", "r") as f:
                crop_json = json.load(f)
                crop_settings = crop_json["crop_settings"]

            # crop_settings = [{'crop_size_x': 264, 'crop_size_y': 191, 'crop_center_x': 338, 'crop_center_y': 212}]
            # D405  tsu
            # crop_settings = [{"crop_size": 260, "crop_center_x": 350, "crop_center_y": 240}]
            log_meesage(crop_settings)
            self.cam = RealSense(crop_settings=crop_settings)

            self.ratio = np.load("calib/camera_info/ratio.npy")
            # self.center_pixels = [self.cam.crop_settings[self.camera_id]["crop_center_x"], self.cam.crop_settings[self.camera_id]["crop_center_y"]]
            # diff_center = crop_settings[self.camera_id]["crop_center_x"] - 320 
            # self.center_pixels = [crop_settings[self.camera_id]["crop_size_x"]//2 - diff_center, (crop_settings[self.camera_id]["crop_size_y"])//2]
            self.center_pixels = [crop_settings[self.camera_id]["crop_size_x"]//2, crop_settings[self.camera_id]["crop_size_y"]//2]
            
            log_meesage(f"center pixels {self.center_pixels}")

            #nishikadoma
            # self.center_position = [-0.0809, -0.470]
            # self.center_position = [-0.010, -0.535]
            # self.center_position = [-0.03, -0.5]
            self.center_position = [0.004998829998830001, -0.4519421305709614]


            #tsu
            # self.center_position = [-0.065, -0.470]

            self.camera_pose = np.load("calib/camera_info/camera_pose.npy")
            
            self.register_pose(self.camera_pose[:3], self.camera_pose[3:], "base", "cam")

            self.per_sam.loadPositionDetector()

            return True

        except Exception as e:
            log_error("{} : {}".format(type(e), e))
            return False
        
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
    
    def register_pose(self, tvec, rvec, source, target):
        source_T_target = pytr.transform_from(
            pyrot.matrix_from_compact_axis_angle(rvec), tvec)
        self.tm.add_transform(target, source, source_T_target)

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
        # self.loadJson()
        # self.loadSAM_f_Model()
        self.loadTrainedModel()
        # self.initialize_positionDetector()
        return True

    def loadJson(self, path="config_cfil.json"):
        with open(path, "r") as f:
            json_dict = json.load(f)

        self.train_data_file = json_dict["train_data_file"]
        return json_dict
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action='store_true')
    args = parser.parse_args()

    cfil_agent = Agent()
    if args.debug:
        cfil_agent.test_PD()
    else:
        # cfil_agent.initialize_all()
        cfil_agent.initialize()
        app.run(debug=False, port=PORT, host=HOST)