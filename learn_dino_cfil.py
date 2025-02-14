import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
import csv
import json
import sys
sys.path.append("../")
from pathlib import Path

import matplotlib.pyplot as plt


from torch.utils.tensorboard import SummaryWriter

from CFIL_for_NIP.network import ABN128, ABN256
from CFIL_for_NIP.memory import ApproachMemory

from CFIL_for_NIP import utils

class LearnDINOCFIL():
    def __init__(self, 
                 memory_size=5e4, 
                 batch_size=32, 
                 image_size=256, 
                 train_epochs=10000):
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.approach_memory = ApproachMemory(memory_size, self.device)

        self.train_epochs = train_epochs
        self.csv_data = []

        self.initialize = False
        
        self.writer = None

    def loadCSV(self, file_path=""):
        data_csv_path = os.path.join(file_path, "data.csv")
        with open(data_csv_path, "rt", encoding="shift-jis") as f:
            header = next(csv.reader(f))
            reader = csv.reader(f)
            self.data = np.array([row for row in reader])
            # print(self.data.shape)
            poses = self.data[:, 0:6].astype(np.float32)
            image_paths = self.data[:, 6]
            angles = self.data[:, 7].astype(np.float32)

        return poses, image_paths, angles

    def makeJobLib(self, file_path="", num_pairs=10, load_size=224, layer='key', facet='key', bin=True, thresh=0.05):
        root_path = Path(file_path)
        bottleneck_csv_path = root_path / "bottleneck.csv" 
        with open(bottleneck_csv_path, encoding="shift-jis") as f:
            reader = csv.reader(f)
            bottleneck_pose = np.array([row for row in reader][1][:6]).astype(np.float32)

        import  dino.correspondences as dino_corr
        ref_image_path = root_path / "ref" / "masked_image.jpg"

        poses, image_paths, angles =  self.loadCSV(file_path=file_path)
        for pose, image_path in tqdm(zip(poses, image_paths)):
            # print(image_path)
            basename = image_path.split("\\")[-1]

            image_path = root_path / "image" / basename+".jpg"

            # compute point correspondences
            points1, points2, image1_pil, image2_pil = dino_corr.find_correspondences(ref_image_path, image_path,
                                                                        num_pairs, load_size, layer,
                                                                        facet, bin, thresh)
        
            # TODO:pointを学習データとして使う。DINOの結果を画像として保存する
            # saving point correspondences as images
            correspondences_path = root_path / "correspondences"
            dino_corr.save_correspondences_images(points1, points2, image1_pil, image2_pil, save_path=correspondences_path / f'{Path(ref_image_path).stem}_{Path(image_path).stem}_corresp.png')

        #     image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        #     pose_eb = utils.transform(pose, bottleneck_pose)
        #     pose_eb = self.rotvec2euler(pose_eb)
        #     # print(pose, pose_eb)

        #     if self.initialize == False:
        #         self.approach_memory.initial_settings(image, pose)
        #         self.initialize = True

        #     self.approach_memory.append(image, pose_eb)
        # self.approach_memory.save_joblib(os.path.join(file_path, "approach_memory.joblib"))

    def load_joblib(self, file_path=""):
        self.approach_memory.load_joblib(os.path.join(file_path,"approach_memory.joblib"))

    def train(self, file_path=""):
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
        self.approach_model.train()
        for epoch in tqdm(range(self.train_epochs)):
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
        
    def rotvec2euler(self, pose_rotvec):
        pose = pose_rotvec[:3]
        rotvec = pose_rotvec[3:]
        
        assert len(rotvec) == 3, "len(rotvec) must be 3" 

        rot = utils.Rotation.from_rotvec(rotvec)
        euler = rot.as_euler("xyz")
        pose_euler = np.r_[pose, euler]
        return pose_euler


if __name__ == "__main__":
    settings_file_path = "cfil_config.json"

    json_file = open(settings_file_path, "r")
    json_dict = json.load(json_file)
    file_path = os.path.join(*["CFIL_for_NIP","train_data", json_dict["train_data_file"]])
 
    cl = LearnDINOCFIL(memory_size=json_dict["memory_size"], 
                   batch_size=json_dict["batch_size"], 
                   image_size=json_dict["image_size"], 
                   train_epochs=json_dict["train_epochs"])
    
    if not os.path.exists(os.path.join(file_path, "approach_memory_dino.joblib")):
        cl.makeJobLib(file_path=file_path)

    if os.path.exists(os.path.join(file_path, "approach_memory.joblib")):
        cl.load_joblib(file_path=file_path)
        cl.train(file_path=file_path)