import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from tqdm import tqdm
import joblib
import csv

import sys
sys.path.append("../")

from CFIL_for_NIP.network import CNN, FNN, ABN
from CFIL_for_NIP.memory import ApproachMemory

from CFIL_for_NIP import utils

class CFILLearn():
    def __init__(self):
        memory_size = 5e4
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.approach_memory = ApproachMemory(memory_size, self.device)
        self.csv_data = []

        self.initialize = False

    def loadCSV(self, file_path=""):
        with open(os.path.join(file_path, "data.csv"), "rt") as f:
            header = next(csv.reader(f))
            reader = csv.reader(f)
            self.data = np.array([row for row in reader])
            print(self.data.shape)
            self.poses = self.data[:, 0:6].astype(np.float32)
            self.image_paths = self.data[:, 6]

    def makeJobLib(self, file_path=""):
        self.loadCSV(file_path=file_path)

        with open("CFIL_for_NIP/train_data/bottleneck.csv") as f:
            reader = csv.reader(f)
            bottleneck_pose = np.array([row for row in reader])[0,:-1].astype(np.float32)


        for pose, image_path in zip(self.poses, self.image_paths):
            print(image_path)
            # image = cv2.imread(os.path.join("CFIL_for_NIP", image_path+".jpg"))
            image = cv2.imread(image_path+".jpg")
            image = cv2.resize(image, (128, 128))
            # image = image.transpose(2, 0, 1)
            image = torch.tensor(image, dtype=torch.float32).to(self.device)

            pose_eb = utils.transform(pose, bottleneck_pose)
            pose_eb = torch.tensor(pose_eb, dtype=torch.float32).to(self.device)

            if self.initialize == False:
                self.approach_memory.initial_settings(image, pose)
                self.initialize = True

            self.approach_memory.append(image, pose_eb)
        self.approach_memory.save_joblib(os.path.join(file_path, "approach_memory.joblib"))

    def load_joblib(self, file_path=""):
        self.approach_memory.load_joblib(os.path.join(file_path,"approach_memory.joblib"))

    def train(self, train_epochs=10000):
        self.approach_model = ABN()
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
        
        torch.save(self.approach_model.state_dict(), os.path.join("", "approach_model_final.pth"))

        

if __name__ == "__main__":
    file_path = "CFIL_for_NIP\\train_data\\20240719150226"
 
    cl = CFILLearn()
    # cl.makeJobLib(file_path=file_path)

    cl.load_joblib(file_path=file_path)
    cl.train()