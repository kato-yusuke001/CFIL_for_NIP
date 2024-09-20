import os
print(os.getcwd())
import logging
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request
import torch

from CFIL_for_NIP.memory import ApproachMemory
from CFIL_for_NIP.network import ABN128, ABN256

from logger import setup_logger


cfil_agent = None

# Flask設定
HOST = "192.168.11.55"
PORT = 5000
app = Flask(__name__)

# ロガー設定
logging = setup_logger("torchServer", "torchServer")

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
    model_path = request.form["model_path"]
    ret = cfil_agent.loadTrainedModel(model_path)
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
    ret = cfil_agent.loadSAMModel(image_save_path)
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
    
class CFIL:
    def __init__(self):
        pass

    def initialize(self):
        try:
            self.memory_size = 5e4
            self.image_size = 256

            self.device = "cuda" if torch.cuda.is_available() else "cpu" 
            if self.image_size == 128:
                self.approach_model = ABN128()
            elif self.image_size == 256:
                self.approach_model = ABN256()
            print(self.approach_model.state_dict())
            self.approach_model.to(self.device)

            self.approach_memory = ApproachMemory(self.memory_size, self.device)

            self.use_sam = False

            # self.file_path = "CFIL_for_NIP\\train_data\\20240913_175206_764"

            
            return True
        except Exception as e:
            log_error("{} : {}".format(type(e), e))
            return False
             
    def loadTrainedModel(self, model_path):
        try:
            self.approach_model.load_state_dict(torch.load(os.path.join(model_path, "approach_model_final.pth"),map_location=self.device))
            return True
        except Exception as e:
            log_error("{} : {}".format(type(e), e))
            return False
        
    def loadSAMModel(self,image_path):
        try:
            from perSam import PerSAM
            self.per_sam = PerSAM(annotation_path="sam\\ref", 
                        output_path=os.path.join(image_path, "masked_images"))
            self.per_sam.loadSAM()
            self.use_sam = True
            return True
        except Exception as e:
            log_error("{} : {}".format(type(e), e))
            return False

    def estimate_from_image(self, image_path):
        image = cv2.imread(image_path+".jpg")
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        if self.use_sam:
            masks, best_idx, topk_xy, topk_label = self.per_sam.executePerSAM(image)
            image = self.per_sam.save_masked_image(masks[best_idx], image, image_path.split("\\")[-1]+".jpg")

        image = np.transpose(image, [2, 0, 1])
        image_tensor = torch.ByteTensor(image).to(self.device).float() / 255.
        image_tensor = torch.unsqueeze(image_tensor, 0)
        self.approach_model.eval()
        with torch.no_grad():
            # appraoch
            output_tensor, _, att = self.approach_model(image_tensor)
            output = output_tensor.to('cpu').detach().numpy().copy()
            self.save_attention_fig(image_tensor, att, image_path)

        return output[0].tolist()
    

    def save_attention_fig(self, inputs, attention, image_path):
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
            print(d_inputs, c_att)
            for item_img, item_att in zip(d_inputs, c_att):
                v_img = item_img.transpose((1,2,0))* 255
                resize_att = cv2.resize(item_att[0], (in_x, in_y))
                resize_att = min_max(resize_att)* 255
                vis_map = cv2.cvtColor(resize_att, cv2.COLOR_GRAY2BGR)
                jet_map = cv2.applyColorMap(vis_map.astype(np.uint8), cv2.COLORMAP_JET)
                v_img = v_img.astype(np.uint8)
                jet_map = cv2.addWeighted(v_img, 0.5, jet_map, 0.5, 0)
                cv2.imwrite(image_path+"_attntion.jpg", cv2.vconcat([v_img, jet_map]))

        except Exception as e:
            log_error("{} : {}".format(type(e), e))
            return False

if __name__ == "__main__":
    cfil_agent = CFIL()

    app.run(debug=False, port=PORT, host=HOST)