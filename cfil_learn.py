import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
import csv

import sys
sys.path.append("../")

from torch.utils.tensorboard import SummaryWriter

from CFIL_for_NIP.network import ABN128, ABN256
from CFIL_for_NIP.memory import ApproachMemory

from CFIL_for_NIP import utils

class CFILLearn():
    def __init__(self):
        memory_size = 5e4
        self.batch_size = 32
        self.image_size = 256
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.approach_memory = ApproachMemory(memory_size, self.device)
        self.csv_data = []

        self.initialize = False
        
        self.writer = None

    def loadCSV(self, file_path=""):
        data_csv_path = os.path.join(file_path, "data.csv")
        with open(data_csv_path, "rt") as f:
            header = next(csv.reader(f))
            reader = csv.reader(f)
            self.data = np.array([row for row in reader])
            # print(self.data.shape)
            self.poses = self.data[:, 0:6].astype(np.float32)
            self.image_paths = self.data[:, 6]

    def makeJobLib(self, file_path=""):
        bottleneck_csv_path = os.path.join(file_path, "bottleneck.csv") 
        with open(bottleneck_csv_path) as f:
            reader = csv.reader(f)
            bottleneck_pose = np.array([row for row in reader])[0,:-1].astype(np.float32)

        print(bottleneck_pose)

        self.loadCSV(file_path=file_path)
        for pose, image_path in zip(self.poses, self.image_paths):
            print(image_path)
            # image = cv2.imread(os.path.join("CFIL_for_NIP", image_path+".jpg"))
            image = cv2.imread(image_path+".jpg")
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            # image = image.transpose(2, 0, 1)
            # image = torch.tensor(image, dtype=torch.float32).to(self.device)
            # image = np.array(image, dtype=np.float32)

            pose_eb = utils.transform(pose, bottleneck_pose)
            # pose_eb = torch.tensor(pose_eb, dtype=torch.float32).to(self.device)

            if self.initialize == False:
                self.approach_memory.initial_settings(image, pose)
                self.initialize = True

            self.approach_memory.append(image, pose_eb)
        self.approach_memory.save_joblib(os.path.join(file_path, "approach_memory.joblib"))

    def load_joblib(self, file_path=""):
        self.approach_memory.load_joblib(os.path.join(file_path,"approach_memory.joblib"))

    def train(self, train_epochs=10000, file_path=""):
        tensorboard_dir = os.path.join(
                file_path,
                "cfil_{}".format(datetime.now().strftime("%Y%m%d-%H%M")),
            )
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
            
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

        if self.image_size == 128:
            self.approach_model = ABN128()
        elif self.image_size == 256:
            self.approach_model = ABN256()
        
        self.approach_model.to(self.device)
        self.reg_approach_criterion = nn.MSELoss()
        self.att_approach_criterion = nn.MSELoss()
        self.approach_optimizer = optim.Adam(self.approach_model.parameters(), lr=0.0001)
        # train approach
        for epoch in tqdm(range(train_epochs)):
            self.approach_model.train()
            sample = self.approach_memory.sample(self.batch_size)
            imgs = sample['images_seq']
            positions_eb = sample['positions_seq']
            self.approach_optimizer.zero_grad()
            rx, ax, att = self.approach_model(imgs)

            labels = positions_eb
            reg_loss = self.reg_approach_criterion(rx, labels)
            att_loss = self.att_approach_criterion(ax, labels)
            (reg_loss+att_loss).backward()
            # loss = self.approach_criterion(output, labels)
            # loss.backward()
            self.approach_optimizer.step()
            if epoch % 1000 == 0 and epoch > 0:
                print("epoch: {}".format(epoch) )
                print(" pos: ", positions_eb[0])
                print(" reg out_put: ", rx.detach()[0])
                print(" att out_put: ", ax.detach()[0])
            
            if epoch % 100 == 0:
                self.writer.add_scalar(
                        'loss/approach_reg', reg_loss.detach().item(), epoch)
                self.writer.add_scalar(
                        'loss/approach_att', att_loss.detach().item(), epoch)
            
            if (epoch+1) % 2000 == 0 or epoch == 0:
                time_stamp=datetime.now().strftime("%Y%m%d-%H%M%S")
                self.save_attention_fig(imgs[:10], att[:10], time_stamp, file_path, name="approach_epoch_"+str(epoch+1))  
       
        torch.save(self.approach_model.state_dict(), os.path.join(file_path, "approach_model_final.pth"))

    def min_max(self, x, axis=None):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min)
        return result

    def save_attention_fig(self, inputs, attention, time_stamp, file_path, name=""):
        c_att = attention.data.cpu()
        c_att = c_att.numpy()
        d_inputs = inputs.data.cpu()
        d_inputs = d_inputs.numpy()
        in_b, in_c, in_y, in_x = inputs.shape
        count = 0
        for item_img, item_att in zip(d_inputs, c_att):
            # v_img = ((item_img.transpose((1,2,0)) + 0.5 + [0.485, 0.456, 0.406]) * [0.229, 0.224, 0.225])* 256
            v_img = item_img.transpose((1,2,0))* 255
            # v_img = v_img[:, :, ::-1]
            resize_att = cv2.resize(item_att[0], (in_x, in_y))
            # resize_att *= 255.
            resize_att = self.min_max(resize_att)* 255
            save_dir = os.path.join(file_path, name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            vis_map = cv2.cvtColor(resize_att, cv2.COLOR_GRAY2BGR)
            jet_map = cv2.applyColorMap(vis_map.astype(np.uint8), cv2.COLORMAP_JET)
            v_img = v_img.astype(np.uint8)
            jet_map = cv2.addWeighted(v_img, 0.5, jet_map, 0.5, 0)

            cv2.imwrite(os.path.join(save_dir, 'raw_att_{}.png'.format(time_stamp)), cv2.vconcat([v_img, jet_map]))

            count += 1
        

if __name__ == "__main__":
    file_path = "CFIL_for_NIP\\train_data\\20240913_175206_764"
 
    cl = CFILLearn()
    if not os.path.exists(os.path.join(file_path, "approach_memory.joblib")):
        cl.makeJobLib(file_path=file_path)

    if os.path.exists(os.path.join(file_path, "approach_memory.joblib")):
        cl.load_joblib(file_path=file_path)
        cl.train(file_path=file_path)