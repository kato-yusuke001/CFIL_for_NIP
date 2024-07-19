from math import degrees
from operator import pos
from unicodedata import name
from urllib.parse import ParseResultBytes
import torch
# import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
import numpy as np
import time
import cv2
from datetime import datetime
import os
import copy
import json
from tqdm import tqdm
import glob

from network import CNN, FNN, ABN
from memory import ApproachMemory

import utils


class CoarseToFineImitation():
    def __init__(self, log_dir=None, experment_config_file="configs/experiment_config.json",no_robot=False):
        if log_dir is None:
            self.log_dir = "logs/{}".format(datetime.now().strftime("%Y%m%d-%H%M"))
        else:
            self.log_dir = os.path.join("logs", log_dir)
            self.dsae_dir = os.path.join("logs/dsae", log_dir)
            if not os.path.exists(self.dsae_dir):
                os.makedirs(self.dsae_dir)
            print("log folder", self.log_dir)
        self.writer = None
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.load_config_file(experment_config_file)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.approach_memory = ApproachMemory(self.memory_size, self.device)
        self.last_inch_memory = ApproachMemory(self.memory_size, self.device)

        self.approach_model = CNN()
        self.approach_model.to(self.device)
        self.approach_criterion = nn.MSELoss()
        self.approach_optimizer = optim.Adam(self.approach_model.parameters(), lr=self.lr)

        self.last_inch_model = CNN()
        self.last_inch_model.to(self.device)
        self.last_inch_criterion = nn.MSELoss()
        self.last_inch_optimizer = optim.Adam(self.last_inch_model.parameters(), lr=self.lr)

        self.image_size = 128

        self.robot = None
        # if no_robot:
        #     pass
        # else:
        #     import ur_controller
        #     self.robot = ur_controller.URController()
        #     image, position = self.robot.reset_start()
        #     image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        #     self.approach_memory.initial_settings(image, position[:6])
        #     self.last_inch_memory.initial_settings(image, position[:6])
        self.bottleneck_pose = None
    
    def load_config_file(self, config_file):
        json_file = open(config_file, "r")
        json_dict = json.load(json_file)

        self.image_size = json_dict["image_size"]
        self.lr = json_dict["lr"]
        self.memory_size = json_dict["memory_size"]
        self.batch_size = json_dict["batch_size"]

        self.limit_x = json_dict["limit_x"]
        self.limit_y = json_dict["limit_y"]
        self.limit_z = json_dict["limit_z"]

        self.approach_range_x = json_dict["approach_range_x"]
        self.approach_range_y = json_dict["approach_range_y"]
        self.approach_z = json_dict["approach_z"]
        self.approach_range_theta = json_dict["approach_range_theta"]

        self.last_inch_range_x = json_dict["last_inch_range_x"]
        self.last_inch_range_y = json_dict["last_inch_range_y"]
        self.last_inch_z = json_dict["last_inch_z"]
        self.last_inch_range_theta = json_dict["last_inch_range_theta"]

        self.drop_limit = json_dict["drop_limit"]

    def setupRobot(self, robot):
        self.robot = robot
        image, position = self.robot.reset_start()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        self.approach_memory.initial_settings(image, position[:6])
        self.last_inch_memory.initial_settings(image, position[:6])


    def define_network_for_dsae(self, image_output_size=(128, 128), out_channels=(128, 128, 64, 32, 16)):
        from network.dsae import DeepSpatialAutoencoder, DSAE_Loss
        
        self.dsae_model = DeepSpatialAutoencoder(in_channels=3, image_output_size=image_output_size, out_channels=out_channels, normalise=True)
        self.dsae_model.to(self.device)
        self.dsae_criterion = DSAE_Loss(add_g_slow=False)
        self.dsae_optimizer = optim.Adam(self.dsae_model.parameters(), lr=0.001)

        self.approach_model = FNN()
        self.approach_model.to(self.device)
        self.approach_criterion = nn.MSELoss()
        self.approach_optimizer = optim.Adam(self.approach_model.parameters(), lr=0.0001)

        self.last_inch_model = FNN()
        self.last_inch_model.to(self.device)
        self.last_inch_criterion = nn.MSELoss()
        self.last_inch_optimizer = optim.Adam(self.last_inch_model.parameters(), lr=0.0001)
    
    def train(self, train_epochs=10000, writer=None, last_inch=True):
        if writer is None:
            tensorboard_dir = os.path.join(
                    self.log_dir,
                    "pick_{}".format(datetime.now().strftime("%Y%m%d-%H%M")),
                )
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
                
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # train approach
        for epoch in tqdm(range(train_epochs)):
            # print('epoch: {}'.format(epoch))
            self.approach_model.train()
            sample = self.approach_memory.sample(self.batch_size)
            imgs = sample['images_seq']
            positions = sample['positions_seq']
            self.approach_optimizer.zero_grad()
            output = self.approach_model(imgs)
            labels = positions
            loss = self.approach_criterion(output, labels)
            loss.backward()
            self.approach_optimizer.step()
            if epoch % 1000 == 0 and epoch > 0 and self.robot is not None:
                # self.test(epoch, last_inch=False)
                print("epoch: {}".format(epoch) )
                print(" pos: ", positions[0])
                print(" out_put: ", output.detach()[0])
            if epoch % 100 == 0:
                # print("epoch: {}\tLoss: {:.6f}".format(epoch, train_loss))
                self.writer.add_scalar(
                        'loss/approach', loss.detach().item(), epoch)
        
        torch.save(self.approach_model.state_dict(), os.path.join(self.log_dir, "approach_model_final.pth"))

        if not last_inch and self.robot is not None:
            for i in range(5):
                self.test(epoch+i+1)
            
            return 
            
        # train last inch
        for epoch in tqdm(range(train_epochs)):
            # print('epoch: {}'.format(epoch))
            self.last_inch_model.train()
            sample = self.last_inch_memory.sample(self.batch_size)
            imgs = sample['images_seq']
            positions = sample['positions_seq']
            self.last_inch_optimizer.zero_grad()
            output = self.last_inch_model(imgs)
            labels = positions
            loss_last_inch = self.last_inch_criterion(output, labels)
            loss_last_inch.backward()
            self.last_inch_optimizer.step()
            if epoch % 1000 == 0 and epoch > 0 and self.robot is not None:
                # self.test(epoch)
                print("epoch: {}".format(epoch) )
                print(" pos: ", positions[0])
                print(" out_put: ", output.detach()[0])
            if epoch % 100 == 0:
                # print("epoch: {}\tLoss: {:.6f}".format(epoch, train_loss))
                self.writer.add_scalar(
                        'loss/last_inch', loss_last_inch.detach().item(), epoch)

        torch.save(self.last_inch_model.state_dict(), os.path.join(self.log_dir, "last_inch_model_final.pth"))  
        
        if self. robot is not None:
            for i in range(5):
                self.test(epoch+i+1, suction=True)      

    def test(self, epoch, last_inch=True, suction=False):
        # set robot init position
        # ang = np.deg2rad(np.random.uniform(-15, 15))
        ang = 0
        start_pose = utils.rotate(self.bottleneck_pose, [0, 0, ang])

        start_pose[0] = self.bottleneck_pose[0] #+ np.random.uniform(-0.05, 0.05)
        start_pose[1] = self.bottleneck_pose[1] #+ np.random.uniform(-0.05, 0.05)
        start_pose[2] = self.bottleneck_pose[2] +  0.05

        
        image, position_re = self.robot.reset_start(start_pose)
        time.sleep(1.0)
        image = self.robot.get_img()
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = image[0:400, 300:700]
        # image = utils.hsv_extraction(image, self.robot.hsvLower, self.robot.hsvUpper, self.robot.extraction_mode)
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        utils.save_img(image, os.path.join(self.log_dir, "test_approach_{}".format(epoch)))
        image = np.transpose(image, [2, 0, 1])
        image = torch.ByteTensor(image).to(self.device).float() / 255.
        image = torch.unsqueeze(image, 0)
        self.approach_model.eval()
        self.last_inch_model.eval()
        with torch.no_grad():
            # appraoch
            output_tensor = self.approach_model(image)
            output = output_tensor.to('cpu').detach().numpy().copy()

            # for debug
            test_label = [self.bottleneck_pose[0], self.bottleneck_pose[1], self.bottleneck_pose[5]]
            test_label_tensor = torch.FloatTensor(test_label).to(self.device)
            loss = self.approach_criterion(output_tensor, test_label_tensor)
            if self.writer is not None:
                self.writer.add_scalar(
                            'loss/test_approach', loss.detach().item(), epoch)

            position_eb = [output[0, 0], output[0, 1], 0.01, 0, 0, output[0, 2]]
            position_rb = utils.reverse_transform(position_re, position_eb)    
            position_rb[2] = self.bottleneck_pose[2] + 0.01
            position_rb = self.check_range(position_rb)
            print("approach_position_rb", position_rb)
    
            self.robot.moveL(position_rb)
            time.sleep(1.0)
            # last inch
            if last_inch: 
                position_re = self.robot.get_pose()
                image = self.robot.get_img()
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # image = image[0:400, 300:700]
                # image = utils.hsv_extraction(image, self.robot.hsvLower, self.robot.hsvUpper, self.robot.extraction_mode)
                image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
                utils.save_img(image, os.path.join(self.log_dir, "test_last_inch_{}".format(epoch)))
                image = np.transpose(image, [2, 0, 1])
                image = torch.ByteTensor(image).to(self.device).float() / 255.
                image = torch.unsqueeze(image, 0)

                output_tensor = self.last_inch_model(image)
                output = output_tensor.to('cpu').detach().numpy().copy()

                # for debug
                loss_last_inch = self.last_inch_criterion(output_tensor, test_label_tensor)
                if self.writer is not None:
                    self.writer.add_scalar(
                            'loss/test_last_inch', loss_last_inch.detach().item(), epoch)
                position_eb = [output[0, 0], output[0, 1], 0.01, 0, 0, output[0, 2]]
                position_rb = utils.reverse_transform(position_re, position_eb)
                position_rb[2] = self.bottleneck_pose[2]
                position_rb = self.check_range(position_rb)
                print(output[0])
                print("last_inch_position_rb", position_rb)
                print("bottle_neck", self.bottleneck_pose)
                self.robot.moveL(position_rb)
            
            # play demo
            time.sleep(1)
            self.robot.play_demo_tcp(self.trajectory_tcp)
            if(suction):
                self.exec_suction()
                # pose = self.robot.get_pose()
                # # pose[2] = 0.03
                # self.robot.moveL(pose, speed=0.01)
                # self.robot.suction_on()
                # self.robot.drop_until_sution()
                # time.sleep(3.0)
                # pose = self.robot.get_pose()
                # pose[2] += 0.04
                # self.robot.moveL(pose, speed=0.05)
                # self.robot.suction_off()
                # print("press y to continue")
                # while True:
                #     key = utils.getch()
                #     if key == "y":
                #         break
                #     elif key == "q":
                #         quit()

    def train_with_dsae(self, train_epochs=10000, writer=None, last_inch=True):
        if writer is None:
            tensorboard_dir = os.path.join(
                    self.dsae_dir,
                    "with_dsae_{}".format(datetime.now().strftime("%Y%m%d-%H%M")),
                )
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
                
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # train approach
        self.dsae_model.eval()
        for epoch in tqdm(range(train_epochs)):
            # print('epoch: {}'.format(epoch))
            self.approach_model.train()
            sample = self.approach_memory.sample(self.batch_size)
            imgs = sample['images_seq']
            positions = sample['positions_seq']
            self.approach_optimizer.zero_grad()
            features = self.dsae_model.encoder(imgs)
            output = self.approach_model(features)
            labels = positions
            loss = self.approach_criterion(output, labels)
            loss.backward()
            self.approach_optimizer.step()
            if epoch % 1000 == 0 and epoch > 0 and self.robot is not None:
                # self.test(epoch, last_inch=False)
                print("epoch: {}".format(epoch) )
                print(" pos: ", positions[0])
                print(" out_put: ", output.detach()[0])
            if epoch % 100 == 0:
                # print("epoch: {}\tLoss: {:.6f}".format(epoch, train_loss))
                self.writer.add_scalar(
                        'loss/approach', loss.detach().item(), epoch)
        
        torch.save(self.approach_model.state_dict(), os.path.join(self.dsae_dir, "approach_model_with_dsae_final.pth"))

        if not last_inch and self.robot is not None:
            for i in range(5):
                self.test(epoch+i+1)
            
            return 
            
        # # train last inch
        # for epoch in tqdm(range(train_epochs)):
        #     # print('epoch: {}'.format(epoch))
        #     self.last_inch_model.train()
        #     sample = self.last_inch_memory.sample(self.batch_size)
        #     imgs = sample['images_seq']
        #     positions = sample['positions_seq']
        #     self.last_inch_optimizer.zero_grad()
        #     features = self.dsae_model.encoder(imgs)
        #     output = self.last_inch_model(features)
        #     labels = positions
        #     loss_last_inch = self.last_inch_criterion(output, labels)
        #     loss_last_inch.backward()
        #     self.last_inch_optimizer.step()
        #     if epoch % 1000 == 0 and epoch > 0 and self.robot is not None:
        #         # self.test(epoch)
        #         print("epoch: {}".format(epoch) )
        #         print(" pos: ", positions[0])
        #         print(" out_put: ", output.detach()[0])
        #     if epoch % 100 == 0:
        #         # print("epoch: {}\tLoss: {:.6f}".format(epoch, train_loss))
        #         self.writer.add_scalar(
        #                 'loss/last_inch', loss_last_inch.detach().item(), epoch)

        # torch.save(self.last_inch_model.state_dict(), os.path.join(self.dsae_dir, "last_inch_model_with_dsae_final.pth"))  
        
        # if self. robot is not None:
        #     for i in range(5):
        #         self.test(epoch+i+1, suction=True)
                
    def test_with_dsae(self, epoch, last_inch=True, suction=False):
        # set robot init position
        # ang = np.deg2rad(np.random.uniform(-15, 15))
        ang = 0
        start_pose = utils.rotate(self.bottleneck_pose, [0, 0, ang])

        start_pose[0] = self.bottleneck_pose[0] #+ np.random.uniform(-0.05, 0.05)
        start_pose[1] = self.bottleneck_pose[1] #+ np.random.uniform(-0.05, 0.05)
        start_pose[2] = self.bottleneck_pose[2] +  0.05

        
        image, position_re = self.robot.reset_start(start_pose)
        time.sleep(1.0)
        image = self.robot.get_img()
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = image[0:400, 300:700]
        # image = utils.hsv_extraction(image, self.robot.hsvLower, self.robot.hsvUpper, self.robot.extraction_mode)
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        utils.save_img(image, os.path.join(self.log_dir, "test_approach_{}".format(epoch)))
        image = np.transpose(image, [2, 0, 1])
        image = torch.ByteTensor(image).to(self.device).float() / 255.
        image = torch.unsqueeze(image, 0)
        
        self.dsae_model.eval()
        self.approach_model.eval()
        self.last_inch_model.eval()
        with torch.no_grad():
            features = self.dsae_model.encoder(image)
            # appraoch
            output_tensor = self.approach_model(features)
            output = output_tensor.to('cpu').detach().numpy().copy()

            # for debug
            test_label = [self.bottleneck_pose[0], self.bottleneck_pose[1], self.bottleneck_pose[5]]
            test_label_tensor = torch.FloatTensor(test_label).to(self.device)
            loss = self.approach_criterion(output_tensor, test_label_tensor)
            if self.writer is not None:
                self.writer.add_scalar(
                            'loss/test_approach', loss.detach().item(), epoch)

            position_eb = [output[0, 0], output[0, 1], 0.01, 0, 0, output[0, 2]]
            position_rb = utils.reverse_transform(position_re, position_eb)    
            position_rb[2] = self.bottleneck_pose[2] + 0.01
            position_rb = self.check_range(position_rb)
            print("approach_position_rb", position_rb)
    
            self.robot.moveL(position_rb)
            time.sleep(1.0)
            
            # # last inch
            # if last_inch: 
            #     position_re = self.robot.get_pose()
            #     image = self.robot.get_img()
            #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #     # image = image[0:400, 300:700]
            #     # image = utils.hsv_extraction(image, self.robot.hsvLower, self.robot.hsvUpper, self.robot.extraction_mode)
            #     image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            #     utils.save_img(image, os.path.join(self.log_dir, "test_last_inch_{}".format(epoch)))
            #     image = np.transpose(image, [2, 0, 1])
            #     image = torch.ByteTensor(image).to(self.device).float() / 255.
            #     image = torch.unsqueeze(image, 0)

            #     features = self.dsae_model.encoder(image)
            #     output_tensor = self.last_inch_model(image)
            #     output = output_tensor.to('cpu').detach().numpy().copy()

            #     # for debug
            #     loss_last_inch = self.last_inch_criterion(output_tensor, test_label_tensor)
            #     if self.writer is not None:
            #         self.writer.add_scalar(
            #                 'loss/test_last_inch', loss_last_inch.detach().item(), epoch)
            #     position_eb = [output[0, 0], output[0, 1], 0.01, 0, 0, output[0, 2]]
            #     position_rb = utils.reverse_transform(position_re, position_eb)
            #     position_rb[2] = self.bottleneck_pose[2]
            #     position_rb = self.check_range(position_rb)
            #     print(output[0])
            #     print("last_inch_position_rb", position_rb)
            #     print("bottle_neck", self.bottleneck_pose)
            #     self.robot.moveL(position_rb)
            
            # play demo
            time.sleep(1)
            self.robot.play_demo_tcp(self.trajectory_tcp)
            if(suction):
                self.exec_suction()
                # pose = self.robot.get_pose()
                # # pose[2] = 0.03
                # self.robot.moveL(pose, speed=0.01)
                # self.robot.suction_on()
                # self.robot.drop_until_sution()
                # time.sleep(3.0)
                # pose = self.robot.get_pose()
                # pose[2] += 0.04
                # self.robot.moveL(pose, speed=0.05)
                # self.robot.suction_off()
                # print("press y to continue")
                # while True:
                #     key = utils.getch()
                #     if key == "y":
                #         break
                #     elif key == "q":
                #         quit()

    def check_range(self, pos):
        # limits = [[-0.12, 0.1], [-0.55, -0.20], [0.02, 0.1]]
        limits = [self.limit_x, self.limit_y, self.limit_z]
        for i, l in enumerate(limits):
            if pos[i] < l[0]:
                pos[i] = l[0]
            if pos[i] > l[1]:
                pos[i] = l[1]
        
        return pos

    def collect_demo_traj(self):
        print("PLEASE RECORD GOAL")
        self.bottleneck_pose, recorded_traj = self.robot.keyboard_ctrl()
        np.save(os.path.join(self.log_dir, "bottleneck.npy"), np.array(self.bottleneck_pose))
        print("PLEASE RECORD ACTIONS")
        
        self.robot.reverse_play_demo(recorded_traj)
        self.robot.moveL(self.bottleneck_pose)
        bottleneck_img = self.robot.get_img()
        utils.save_img(bottleneck_img, name="bottleneck")
        
        self.robot.play_demo(recorded_traj)
        self.trajectory_tcp = self.robot.convertTCP(recorded_traj)
        
        self.robot.reverse_demo_tcp(self.trajectory_tcp)
        self.robot.moveL(self.bottleneck_pose)
        self.robot.play_demo_tcp(self.trajectory_tcp)
        
        np.save(os.path.join(self.log_dir, "trajectory_tcp.npy"), np.array(self.trajectory_tcp))
        
    def use_previous_demo_traj(self,path=None):
        if path is None:
            path = "logs"
        self.load_bottleneck(path=path)
        np.save(os.path.join(self.log_dir, "bottleneck.npy"), np.array(self.bottleneck_pose))
        self.load_trajectory(path=path)
        np.save(os.path.join(self.log_dir, "trajectory_tcp.npy"), np.array(self.trajectory_tcp))

        self.robot.moveL(self.bottleneck_pose)
        self.robot.play_demo_tcp(self.trajectory_tcp)

        print("please set object! press y to continue")
        while True:
            key = utils.getch()
            if key == "y":
                break
            elif key == "q":
                quit()

        self.robot.reverse_demo_tcp(self.trajectory_tcp)
    
    def collect_approach_traj(self, num=100, last_inch=True):
        goal_pose = copy.deepcopy(self.bottleneck_pose)
        goal_pose[2] = self.bottleneck_pose[2] + 0.05        
        self.robot.moveL(goal_pose)
        for n in tqdm(range(num)):
            # imgs, positions = self.approach_from_random_pose()
            imgs, positions = self.random_move_collection()
            self.approach_memory.append_episode(imgs, positions)
        
        if last_inch:
            for n in tqdm(range(num)):
                # imgs, positions = self.approach_from_random_pose(last_inch)
                imgs, positions = self.random_move_collection(last_inch)
                self.last_inch_memory.append_episode(imgs, positions)
              
            
                
    def approach_from_random_pose(self, last_inch=False):
                    
        origin_imgs = []
        positions = []
        # set robot init position
        ang = np.deg2rad(np.random.uniform(self.approach_range_theta[0], self.approach_range_theta[1]))
        start_pose = utils.rotate(self.bottleneck_pose, [0, 0, ang])

        if last_inch:
            start_pose[0] = self.bottleneck_pose[0] + np.random.uniform(self.last_inch_range_x[0], self.last_inch_range_x[1])
            start_pose[1] = self.bottleneck_pose[1] + np.random.uniform(self.last_inch_range_y[0], self.last_inch_range_y[1])
            start_pose[2] = self.bottleneck_pose[2] + 0.01
        else:
            start_pose[0] = self.bottleneck_pose[0] + np.random.uniform(self.approach_range_x[0], self.approach_range_x[1])
            start_pose[1] = self.bottleneck_pose[1] + np.random.uniform(self.approach_range_y[0], self.approach_range_y[1])
            start_pose[2] = self.bottleneck_pose[2] + 0.05

        image, position = self.robot.reset_start(start_pose)
        # position_eb = utils.transform(position, self.bottleneck_pose)
        # image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        origin_imgs.append(image)
        positions.append(position)

        
        s_time = time.time()
        # move to target pose
        self.robot.moveL(self.bottleneck_pose, _asynchronous=True)
        while not self.robot.check_arrival(self.bottleneck_pose):
            if time.time() - s_time > 1/30.0:
                image = self.robot.get_img()
                position = self.robot.get_pose()
                origin_imgs.append(image)
                positions.append(position)

                s_time = time.time()
        self.robot.stop()
        
        imgs = []
        positions_eb = []
        for image, position in zip(origin_imgs, positions):
            position_eb = utils.transform(position, self.bottleneck_pose)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # image = image[0:400, 300:700]
            # image = utils.hsv_extraction(image, self.robot.hsvLower, self.robot.hsvUpper, extract_mode=self.robot.extraction_mode)
            image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            imgs.append(image)
            positions_eb.append(position_eb)
            
        return np.array(imgs), np.array(positions_eb)
    
    def random_move_collection(self, last_inch=False):
                    
        origin_imgs = []
        positions = []
        # set robot init position
        speed = 0.08
        acc = 0.5
        if last_inch:
            ang = np.deg2rad(np.random.uniform(self.last_inch_range_theta[0], self.last_inch_range_theta[1]))
            goal_pose = utils.rotate(self.bottleneck_pose, [0, 0, ang])
            goal_pose[0] = self.bottleneck_pose[0] + np.random.uniform(self.last_inch_range_x[0], self.last_inch_range_x[1])
            goal_pose[1] = self.bottleneck_pose[1] + np.random.uniform(self.last_inch_range_y[0], self.last_inch_range_y[1])
            goal_pose[2] = self.bottleneck_pose[2] + self.last_inch_z
            speed = 0.01
            acc = 0.1
        else:
            ang = np.deg2rad(np.random.uniform(self.approach_range_theta[0], self.approach_range_theta[1]))
            goal_pose = utils.rotate(self.bottleneck_pose, [0, 0, ang])
            goal_pose[0] = self.bottleneck_pose[0] + np.random.uniform(self.approach_range_x[0], self.approach_range_x[1])
            goal_pose[1] = self.bottleneck_pose[1] + np.random.uniform(self.approach_range_y[0], self.approach_range_y[1])
            goal_pose[2] = self.bottleneck_pose[2] + self.approach_z
            

        current_pose = self.robot.get_pose()
        image, position = self.robot.reset_start(current_pose)
        # position_eb = utils.transform(position, self.bottleneck_pose)
        # image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        origin_imgs.append(copy.deepcopy(image))
        positions.append(copy.deepcopy(position))

        
        s_time = time.time()
        # move to target pose
        self.robot.moveL(goal_pose, speed=speed,  acceleration=0.1, _asynchronous=True)
        while not self.robot.check_arrival(goal_pose):
            if time.time() - s_time > 1/30.0:
                image = copy.deepcopy(self.robot.get_img())
                position = copy.deepcopy(self.robot.get_pose())
                origin_imgs.append(image)
                positions.append(position)

                s_time = time.time()
        self.robot.stop()
        
        imgs = []
        positions_eb = []
        for image, position in zip(origin_imgs, positions):
            position_eb = utils.transform(position, self.bottleneck_pose)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # image = image[0:400, 300:700]
            # image = utils.hsv_extraction(image, self.robot.hsvLower, self.robot.hsvUpper, self.robot.extraction_mode)
            image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            imgs.append(copy.deepcopy(image))
            positions_eb.append(copy.deepcopy(position_eb))
            
        return np.array(imgs), np.array(positions_eb)

    def exec_suction(self, random_move=False):
        # pose = self.robot.get_pose()
        # # pose[2] = 0.03
        # self.robot.moveL(pose, speed=0.01)
        self.robot.suction_on()
        ret = self.robot.drop_until_sution(limit=self.drop_limit)
        if not ret:
            self.robot.suction_off()
            pose = self.robot.get_pose()
            pose[2] += 0.01
            self.robot.moveL(pose, speed=0.1)
            print("press y to continue")
            while True:
                key = utils.getch()
                if key == "y":
                    break
                elif key == "q":
                    quit()
        time.sleep(1.0)
        pose = self.robot.get_pose()
        pose[2] += 0.03
        self.robot.moveL(pose, speed=0.1)
        if random_move:
            self.random_move_for_test()
        self.robot.suction_off()

    def random_move_for_test(self):
        # set robot init position
        ang = np.deg2rad(np.random.uniform(self.approach_range_theta[0]*2/3, self.approach_range_theta[1]*2/3))
        goal_pose = utils.rotate(self.bottleneck_pose, [0, 0, ang])
        goal_pose[0] = self.bottleneck_pose[0] + np.random.uniform(self.approach_range_x[0]+0.03, self.approach_range_x[1])
        goal_pose[1] = self.bottleneck_pose[1] + np.random.uniform(self.approach_range_y[0]-0.025, -0.1)
        goal_pose[2] = self.bottleneck_pose[2]
            
        current_pose = self.robot.get_pose()
        image, position = self.robot.reset_start(current_pose)
        
        s_time = time.time()
        # move to target pose
        self.robot.moveL(goal_pose, speed=0.5,  acceleration=0.5, _asynchronous=False)
        
        return 
 
    def load_bottleneck(self, path=None, filename="bottleneck.npy"):
        if path is None:
            path = self.log_dir
        self.bottleneck_pose = np.load(os.path.join(path, filename))

    def load_trajectory(self, path=None, filename="trajectory_tcp.npy"):
        if path is None:
            path = self.log_dir
        self.trajectory_tcp = np.load(os.path.join(path, filename))

    def load_memory(self, path=None, last_inch=True):
        if path is None:
            path = self.log_dir
        self.approach_memory.load_joblib(os.path.join(path, 'approach.joblib'))
        if last_inch:
            self.last_inch_memory.load_joblib(os.path.join(path, 'last_inch.joblib'))
       
    def save_memory(self, path=None):
        if path is None:
            path = self.log_dir
        self.approach_memory.save_joblib(os.path.join(path, 'approach.joblib'))
        self.last_inch_memory.save_joblib(os.path.join(path, 'last_inch.joblib'))
        
    def load_model(self, path=None, last_inch=True):
        if path is None:
            path = self.log_dir
        self.approach_model.load_state_dict(torch.load(os.path.join(path, "approach_model_final.pth")))
        if last_inch:
            self.last_inch_model.load_state_dict(torch.load(os.path.join(path, "last_inch_model_final.pth")))

    def load_for_test(self, last_inch=True):
        self.load_model(last_inch=last_inch)
        self.load_bottleneck()
        self.load_trajectory()

    def load_with_dsae(self, path=None, last_inch=True, test=False):
        if path is None:
            path = self.log_dir
        self.approach_memory.load_joblib(os.path.join(path, 'approach.joblib'))
        if last_inch:
            self.last_inch_memory.load_joblib(os.path.join(path, 'last_inch.joblib'))

        self.dsae_model.load_state_dict(torch.load(os.path.join(self.dsae_dir,"models/dsae_model_final.pth")))
        
        if test:
            self.approach_model.load_state_dict(torch.load(os.path.join(self.dsae_dir, "approach_model_with_dsae_final.pth")))
            if last_inch:
                self.last_inch_model.load_state_dict(torch.load(os.path.join(self.dsae_dir, "last_inch_model_with_dsae_final.pth")))
            self.load_bottleneck()
            self.load_trajectory()

class CFIL_ABN(CoarseToFineImitation):
    def __init__(self, log_dir=None, abn_dir=None, **args):
        super().__init__(log_dir=log_dir, **args)

        if log_dir is None:
            self.abn_dir = "logs/abn/{}".format(datetime.now().strftime("%Y%m%d-%H%M"))
        else:
            self.abn_dir = os.path.join("logs/abn", log_dir)
            print("abn log folder", self.abn_dir)
        self.writer = None
        if not os.path.exists(self.abn_dir):
            os.makedirs(self.abn_dir)

        self.approach_model = ABN()
        self.approach_model.to(self.device)
        self.reg_approach_criterion = nn.MSELoss()
        self.att_approach_criterion = nn.MSELoss()
        self.approach_optimizer = optim.Adam(self.approach_model.parameters(), lr=0.0001)

        self.last_inch_model = ABN()
        self.last_inch_model.to(self.device)
        self.reg_last_inch_criterion = nn.MSELoss()
        self.att_last_inch_criterion = nn.MSELoss()
        self.last_inch_optimizer = optim.Adam(self.last_inch_model.parameters(), lr=0.0001)


    def train(self, train_epochs=10000, writer=None, last_inch=False):
        if writer is None:
            tensorboard_dir = os.path.join(
                    self.abn_dir,
                    "tensorboad",
                    "abn_{}".format(datetime.now().strftime("%Y%m%d-%H%M")),
                )
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
                
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # train approach
        for epoch in tqdm(range(train_epochs)):
            # print('epoch: {}'.format(epoch))
            self.approach_model.train()
            sample = self.approach_memory.sample(self.batch_size)
            imgs = sample['images_seq']
            positions = sample['positions_seq']
            self.approach_optimizer.zero_grad()
            rx, ax, att = self.approach_model(imgs)
            
            labels = positions
            reg_loss = self.reg_approach_criterion(rx, labels)
            att_loss = self.att_approach_criterion(ax, labels)
            (reg_loss+att_loss).backward()
            self.approach_optimizer.step()
            if (epoch+1) % 1000 == 0 and epoch > 0 and self.robot is not None:
                # self.test(epoch, last_inch=False)
                print("epoch: {}".format(epoch) )
                print(" pos: ", positions[0])
                print(" reg out_put: ", rx.detach()[0])
                print(" att out_put: ", ax.detach()[0])
            if (epoch+1) % 100 == 0:
                # print("epoch: {}\tLoss: {:.6f}".format(epoch, train_loss))
                self.writer.add_scalar(
                        'loss/approach_reg', reg_loss.detach().item(), epoch)
                self.writer.add_scalar(
                        'loss/approach_att', att_loss.detach().item(), epoch)
            if (epoch+1) % 2000 == 0 or epoch == 0:
                time_stamp=datetime.now().strftime("%Y%m%d-%H%M%S")
                self.save_attention_fig(imgs[:10], att[:10], time_stamp, name="approach_epoch_"+str(epoch+1))        
        
        torch.save(self.approach_model.state_dict(), os.path.join(self.abn_dir, "abn_approach_model_final.pth"))

        if not last_inch and self.robot is not None:
            for i in range(5):
                self.test(epoch+i+1)
            
            return 
            
        # train last inch
        for epoch in tqdm(range(train_epochs)):
            # print('epoch: {}'.format(epoch))
            self.last_inch_model.train()
            sample = self.last_inch_memory.sample(self.batch_size)
            imgs = sample['images_seq']
            positions = sample['positions_seq']
            self.last_inch_optimizer.zero_grad()
            rx, ax, att = self.last_inch_model(imgs)
            labels = positions
            reg_loss_last_inch = self.reg_last_inch_criterion(rx, labels)
            att_loss_last_inch = self.att_last_inch_criterion(ax, labels)
            (reg_loss_last_inch+att_loss_last_inch).backward()
            self.last_inch_optimizer.step()
            if epoch % 1000 == 0 and epoch > 0 and self.robot is not None:
                # self.test(epoch)
                print("epoch: {}".format(epoch) )
                print(" pos: ", positions[0])
                print(" reg out_put: ", rx.detach()[0])
                print(" att out_put: ", ax.detach()[0])
            if epoch % 100 == 0:
                # print("epoch: {}\tLoss: {:.6f}".format(epoch, train_loss))
                self.writer.add_scalar(
                        'loss/last_inch_reg', reg_loss_last_inch.detach().item(), epoch)
                self.writer.add_scalar(
                        'loss/last_inch_att', att_loss_last_inch.detach().item(), epoch)
            if (epoch+1) % 2000 == 0 or epoch == 0:
                time_stamp=datetime.now().strftime("%Y%m%d-%H%M%S")
                self.save_attention_fig(imgs[:10], att[:10], time_stamp, name="last_inch_epoch_"+str(epoch+1))  

        torch.save(self.last_inch_model.state_dict(), os.path.join(self.abn_dir, "abn_last_inch_model_final.pth"))  
        
        if self. robot is not None:
            for i in range(5):
                self.test(epoch+i+1, suction=True)      

    def test(self, epoch, last_inch=True, suction=False):
        # set robot init position
        # ang = np.deg2rad(np.random.uniform(-15, 15))
        ang = 0
        start_pose = utils.rotate(self.bottleneck_pose, [0, 0, ang])

        start_pose[0] = self.bottleneck_pose[0] #+ np.random.uniform(-0.05, 0.05)
        start_pose[1] = self.bottleneck_pose[1] #+ np.random.uniform(-0.05, 0.05)
        start_pose[2] = self.bottleneck_pose[2] +  0.05

        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%H%S")

        image, position_re = self.robot.reset_start(start_pose)
        time.sleep(1.0)
        image = self.robot.get_img()
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = image[0:400, 300:700]
        # image = utils.hsv_extraction(image, self.robot.hsvLower, self.robot.hsvUpper, self.robot.extraction_mode)
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        # utils.save_img(image, os.path.join(self.abn_dir, "test_approach_{}".format(epoch)))
        save_approach_dir = os.path.join(self.abn_dir, "test_approach")
        if not os.path.exists(save_approach_dir):
            os.mkdir(save_approach_dir)
        utils.save_img(image, os.path.join(save_approach_dir, "raw_{}".format(time_stamp)))
        
        image = np.transpose(image, [2, 0, 1])
        image_tensor = torch.ByteTensor(image).to(self.device).float() / 255.
        image_tensor = torch.unsqueeze(image_tensor, 0)
        self.approach_model.eval()
        self.last_inch_model.eval()
        with torch.no_grad():
            # appraoch
            output_tensor, _, attention = self.approach_model(image_tensor)
            self.save_attention_fig(image_tensor, attention, time_stamp, "test_approach")
            output = output_tensor.to('cpu').detach().numpy().copy()

            # for debug
            test_label = [self.bottleneck_pose[0], self.bottleneck_pose[1], self.bottleneck_pose[5]]
            test_label_tensor = torch.FloatTensor(test_label).to(self.device)
            loss = self.approach_criterion(output_tensor, test_label_tensor)
            if self.writer is not None:
                self.writer.add_scalar(
                            'loss/test_approach', loss.detach().item(), epoch)

            position_eb = [output[0, 0], output[0, 1], 0.01, 0, 0, output[0, 2]]
            position_rb = utils.reverse_transform(position_re, position_eb)    
            position_rb[2] = self.bottleneck_pose[2] + 0.01
            position_rb = self.check_range(position_rb)
            print("approach_position_rb", position_rb)
    
            self.robot.moveL(position_rb)
            time.sleep(1.0)
            # last inch
            if last_inch: 
                position_re = self.robot.get_pose()
                image = self.robot.get_img()
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # image = image[0:400, 300:700]
                # image = utils.hsv_extraction(image, self.robot.hsvLower, self.robot.hsvUpper, self.robot.extraction_mode)
                image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
                # utils.save_img(image, os.path.join(self.abn_dir, "test_last_inch_{}".format(epoch)))
                save_last_inch_dir = os.path.join(self.abn_dir, "test_last_inch")
                if not os.path.exists(save_last_inch_dir):
                    os.mkdir(save_last_inch_dir)
                utils.save_img(image, os.path.join(save_last_inch_dir, "raw_{}".format(time_stamp)))
                image = np.transpose(image, [2, 0, 1])
                image_tensor = torch.ByteTensor(image).to(self.device).float() / 255.
                image_tensor = torch.unsqueeze(image_tensor, 0)

                output_tensor, _, attention = self.last_inch_model(image_tensor)
                
                self.save_attention_fig(image_tensor, attention, time_stamp, "test_last_inch")
                output = output_tensor.to('cpu').detach().numpy().copy()

                # for debug
                loss_last_inch = self.last_inch_criterion(output_tensor, test_label_tensor)
                if self.writer is not None:
                    self.writer.add_scalar(
                            'loss/test_last_inch', loss_last_inch.detach().item(), epoch)
                position_eb = [output[0, 0], output[0, 1], 0.01, 0, 0, output[0, 2]]
                position_rb = utils.reverse_transform(position_re, position_eb)
                position_rb[2] = self.bottleneck_pose[2]
                position_rb = self.check_range(position_rb)
                print(output[0])
                print("last_inch_position_rb", position_rb)
                print("bottle_neck", self.bottleneck_pose)
                self.robot.moveL(position_rb)
            
            # play demo
            time.sleep(1)
            self.robot.play_demo_tcp(self.trajectory_tcp)
            if(suction):
                self.exec_suction(random_move=True)

    def test_from_image(self, image_path):
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        # utils.save_img(image, os.path.join(self.abn_dir, "test_output"))
        image = np.transpose(image, [2, 0, 1])
        image_tensor = torch.ByteTensor(image).to(self.device).float() / 255.
        image_tensor = torch.unsqueeze(image_tensor, 0)
        self.approach_model.eval()
        self.last_inch_model.eval()
        time_stamp=datetime.now().strftime("%Y%m%d-%H%M%S")
        with torch.no_grad():
            # appraoch
            output_tensor, _, att = self.approach_model(image_tensor)
            self.save_attention_fig(image_tensor, att, time_stamp, name="output")        

    def approach_test_from_image(self, image):
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

        return output[0]
    
    def min_max(self, x, axis=None):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min)
        return result

    def save_attention_fig(self, inputs, attention, time_stamp, name=""):
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
            print(resize_att.dtype)
            save_dir = os.path.join(self.abn_dir, name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            # cv2.imwrite(os.path.join(save_dir, 'stock1.png'), v_img)
            # cv2.imwrite(os.path.join(save_dir, 'stock2.png'), resize_att)
            # v_img = cv2.imread(os.path.join(save_dir, 'stock1.png'))
            # vis_map = cv2.imread(os.path.join(save_dir, 'stock2.png'), 0)
            vis_map = cv2.cvtColor(resize_att, cv2.COLOR_GRAY2BGR)
            # jet_map = cv2.applyColorMap(vis_map.astype(np.uint8), cv2.COLORMAP_HOT)
            jet_map = cv2.applyColorMap(vis_map.astype(np.uint8), cv2.COLORMAP_JET)
            # jet_map = cv2.add(v_img, jet_map)
            v_img = v_img.astype(np.uint8)
            jet_map = cv2.addWeighted(v_img, 0.5, jet_map, 0.5, 0)

            # out_path = os.path.join(save_dir, 'attention_{0:06d}.png'.format(count))
            # cv2.imwrite(out_path, jet_map)
            # out_path = os.path.join(save_dir, 'raw_{0:06d}.png'.format(count))
            # cv2.imwrite(out_path, v_img)
            # print(v_img.shape, jet_map.shape)
            # time_stamp=datetime.now().strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(os.path.join(save_dir, 'raw_att_{}.png'.format(time_stamp)), cv2.vconcat([v_img, jet_map]))

            count += 1

    def test_from_image_folder(self, folder_path, folder_name, ext="png", no_last_inch=False):
        files = glob.glob(os.path.join(folder_path, folder_name, "*."+ext))
        if len(files) == 0:
            print(os.path.join(folder_path, folder_name, "*."+ext))
            return 
        # images = np.array([])
        images = np.empty((len(files), 128,128,3), dtype=np.uint8)
        image_names = np.array([])
        for i, image_path in enumerate(files):
            print(image_path)
            
            image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            # utils.save_img(image, os.path.join(self.abn_dir, "test_output"))            
            # images = np.append(images, image)
            images[i] = image
            basename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
            image_names = np.append(image_names, basename_without_ext)

        images = np.transpose(images, [0, 3, 1, 2])
        images_tensor = torch.ByteTensor(images).to(self.device).float() / 255.
        # images = np.append(images_tensor)
        # images_tensor = torch.unsqueeze(images_tensor, 0)
        self.approach_model.eval()
        self.last_inch_model.eval()
        with torch.no_grad():
            if no_last_inch:
                # appraoch
                output_tensor, _, attentions = self.approach_model(images_tensor)
            else:
                output_tensor, _, attentions = self.last_inch_model(images_tensor)
            # self.save_attention(images_tensor, attentions)   
            self.save_attentions(images_tensor, attentions, image_names, folder_path, name=folder_name)
    
    def save_attention(self, input_images, attentions, image_names, folder_path, name="attention_map"):
        c_att = attentions.data.cpu()
        c_att = c_att.numpy()
        d_inputs = input_images.data.cpu()
        d_inputs = d_inputs.numpy()
        in_b, in_c, in_y, in_x = input_images.shape
        count = 0
        for item_img, item_att, item_name in zip(d_inputs, c_att, image_names):
            v_img = item_img.transpose((1,2,0))* 255
            # v_img = item_img
            # v_img = v_img[:, :, ::-1]
            resize_att = cv2.resize(item_att[0], (in_x, in_y))
            resize_att = self.min_max(resize_att)* 255

            save_dir = os.path.join("remapping", name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            vis_map = cv2.cvtColor(resize_att, cv2.COLOR_GRAY2BGR)
            jet_map = cv2.applyColorMap(vis_map.astype(np.uint8), cv2.COLORMAP_JET)
            v_img = v_img.astype(np.uint8)
            jet_map = cv2.addWeighted(v_img, 0.5, jet_map, 0.5, 0)

            cv2.imwrite(os.path.join(save_dir, item_name+'.png'.format(count)), cv2.vconcat([v_img, jet_map]))

            count += 1
            
    def get_min_max(self, x, axis=None):
        min_v = x.min(axis=axis, keepdims=True)
        max_v = x.max(axis=axis, keepdims=True)
        return min_v, max_v
    
    def save_attentions(self, input_images, attentions, image_names, folder_path, name="attention_map"):
        c_att = attentions.data.cpu()
        c_att = c_att.numpy()
        d_inputs = input_images.data.cpu()
        d_inputs = d_inputs.numpy()
        in_b, in_c, in_y, in_x = input_images.shape
        attentions = []
        mins = np.array([])
        maxs = np.array([])
        for item_img, item_att, item_name in zip(d_inputs, c_att, image_names):
            v_img = item_img.transpose((1,2,0))* 255
            # v_img = item_img
            # v_img = v_img[:, :, ::-1]
            resize_att = cv2.resize(item_att[0], (in_x, in_y))
            # resize_att = self.min_max(resize_att)* 255
            min_v, max_v = self.get_min_max(resize_att)
            attentions.append(resize_att)
            mins = np.append(mins, min_v)
            maxs = np.append(maxs, max_v)

        all_min = np.array(mins).min(axis=None, keepdims=True)
        all_max = np.array(maxs).max(axis=None, keepdims=True)
        
        for item_img, item_att, item_name in zip(d_inputs, c_att, image_names):
            v_img = item_img.transpose((1,2,0))* 255
            resize_att = cv2.resize(item_att[0], (in_x, in_y))
            resize_att = 255*(resize_att-all_min)/(all_max-all_min)
            save_dir = os.path.join(self.abn_dir, "remapping_normalised", name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            vis_map = cv2.cvtColor(resize_att.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            jet_map = cv2.applyColorMap(vis_map.astype(np.uint8), cv2.COLORMAP_JET)
            v_img = v_img.astype(np.uint8)
            jet_map = cv2.addWeighted(v_img, 0.5, jet_map, 0.5, 0)

            cv2.imwrite(os.path.join(save_dir, item_name+'.png'), cv2.vconcat([v_img, jet_map]))
            print(os.path.join(save_dir, item_name+'.png'))

    def load_abn_model(self, path=None, last_inch=True):
        if path is None:
            path = self.abn_dir
        self.approach_model.load_state_dict(torch.load(os.path.join(path, "abn_approach_model_final.pth")))
        if last_inch:
            self.last_inch_model.load_state_dict(torch.load(os.path.join(path, "abn_last_inch_model_final.pth")))

    
    def load_for_test(self, last_inch=True):
        self.load_abn_model(last_inch=last_inch)
        self.load_bottleneck(path=self.abn_dir)
        self.load_trajectory(path=self.abn_dir)

    
if __name__ == "__main__":
    cfil = CoarseToFineImitation()
    
    cfil.collect_demo_traj()
    
    load = False    
    if load:
        cfil.load_memory()
    else:
        cfil.collect_approach_traj(last_inch=True)
        cfil.save_memory()
        
    cfil.train()
