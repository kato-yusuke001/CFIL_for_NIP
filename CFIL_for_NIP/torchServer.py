import os
print(os.getcwd())
import logging
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request
import torch

from memory import ApproachMemory
from network import ABN128, ABN256

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
        log_meesage("Trained Model Loaded")
        return "Success"
    else:
        log_error("Trained Model Loading Failed")
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


            self.file_path = "CFIL_for_NIP\\train_data\\20240913_161619_907"

            
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

    def estimate_from_image(self, image_path):
        image = cv2.imread(image_path+".jpg")
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        # utils.save_img(image, os.path.join(self.abn_dir, "test_output"))
        image = np.transpose(image, [2, 0, 1])
        image_tensor = torch.ByteTensor(image).to(self.device).float() / 255.
        image_tensor = torch.unsqueeze(image_tensor, 0)
        self.approach_model.eval()
        time_stamp=datetime.now().strftime("%Y%m%d-%H%M%S")
        with torch.no_grad():
            # appraoch
            output_tensor, _, att = self.approach_model(image_tensor)
            output = output_tensor.to('cpu').detach().numpy().copy()

        return output[0].tolist()
    
    


if __name__ == "__main__":
    cfil_agent = CFIL()

    app.run(debug=False, port=PORT, host=HOST)