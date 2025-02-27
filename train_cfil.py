import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as transforms
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
import csv
import json
import sys
sys.path.append("../")


from torch.utils.tensorboard import SummaryWriter

from CFIL_for_NIP.network import ABN128, ABN256
from CFIL_for_NIP.memory import ApproachMemory

from CFIL_for_NIP import utils

class LearnCFIL():
    def __init__(self, 
                 memory_size=5e4, 
                 batch_size=32, 
                 image_size=256, 
                 train_epochs=10000,
                 use_sam=False,
                 sam_f=False):
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.approach_memory = ApproachMemory(memory_size, self.device)

        self.use_sam = use_sam

        self.sam_f = sam_f

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

    def makeJobLib(self, file_path="", mask_image_only=False):
        bottleneck_csv_path = os.path.join(file_path, "bottleneck.csv") 
        with open(bottleneck_csv_path, encoding="shift-jis") as f:
            reader = csv.reader(f)
            # bottleneck_pose = np.array([row for row in reader])[0,:-1].astype(np.float32)
            # print([row for row in reader][1][:6])
            bottleneck_pose = np.array([row for row in reader][1][:6]).astype(np.float32)

        print(bottleneck_pose)

        if self.use_sam:
            from perSam import PerSAM
            if self.sam_f:
                per_sam = PerSAM(
                        # annotation_path="sam\\ref", 
                        annotation_path=os.path.join(file_path, "ref"), 
                        output_path=os.path.join(file_path, "masked_images_f"))
            
                per_sam.loadSAM_f()
            else:
                per_sam = PerSAM(
                        # annotation_path="sam\\ref", 
                        annotation_path=os.path.join(file_path, "ref"), 
                        output_path=os.path.join(file_path, "masked_images"))
                per_sam.loadSAM()

        poses, image_paths, angles =  self.loadCSV(file_path=file_path)
        for pose, image_path in tqdm(zip(poses, image_paths)):
            # print(image_path)
            basename = image_path.split("\\")[-1]

            image = cv2.imread(os.path.join(file_path,"image/" + basename+".jpg"))
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            
            if self.use_sam:
                if self.sam_f:
                    masks, best_idx, topk_xy, topk_label = per_sam.executePerSAM_f(image)
                else:
                    masks, best_idx, topk_xy, topk_label = per_sam.executePerSAM(image)
                image = per_sam.save_masked_image(masks[best_idx], image, image_path.split("\\")[-1]+".jpg")
                if mask_image_only:
                    image = np.zeros((masks[best_idx].shape[0], masks[best_idx].shape[1], 3), dtype=np.uint8)
                    image[masks[best_idx], :] = np.array([[0, 0, 128]])


            # image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            pose_eb = utils.transform(pose, bottleneck_pose)
            pose_eb = self.rotvec2euler(pose_eb)
            # print(pose, pose_eb)

            if self.initialize == False:
                self.approach_memory.initial_settings(image, pose)
                self.initialize = True

            self.approach_memory.append(image, pose_eb)
        return self.approach_memory
        # if self.sam_f:
        #     self.approach_memory.save_joblib(os.path.join(file_path, "approach_memory_f.joblib"))
        # else:
        #     self.approach_memory.save_joblib(os.path.join(file_path, "approach_memory.joblib"))

    def load_joblib(self, joblib_path=""):
        self.approach_memory.load_joblib(joblib_path)

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
        # self.reg_approach_criterion = nn.MSELoss()
        # self.att_approach_criterion = nn.MSELoss()
        self.reg_approach_criterion = MSE_decay()
        self.att_approach_criterion = MSE_decay()
        self.approach_optimizer = optim.Adam(self.approach_model.parameters(), lr=0.0001)
        # train approach
        self.approach_model.train()

        rng = np.random.default_rng()
        RandomInvert = transforms.RandomInvert(p=0.5)
        for epoch in tqdm(range(self.train_epochs)):
            sample = self.approach_memory.sample(self.batch_size)
            imgs = sample['images_seq']
            imgs = RandomInvert(imgs)
            positions_eb = sample['positions_seq']
            rx, ax, att = self.approach_model(imgs)

            labels = positions_eb
            reg_loss = self.reg_approach_criterion(rx, labels)
            att_loss = self.att_approach_criterion(ax, labels)

            self.approach_optimizer.zero_grad()
            (reg_loss+att_loss).backward()
            # loss = self.approach_criterion(output, labels)
            # loss.backward()
            self.approach_optimizer.step()
            if (epoch % (self.train_epochs//10) == 0 and epoch > 0) or epoch == (self.train_epochs-1):
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

class MSE_decay(torch.nn.Module):
    def __init__(self):
        super(MSE_decay, self).__init__()
        self.decay = torch.tensor([1,1,0,0,0,100]).to("cuda")
    def forward(self, pred, true):
        return torch.mean(
            torch.einsum('j,ik->ik', (self.decay, (pred - true) * (pred - true))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
    parser.add_argument('--data_dir', type=str, required=True )
    parser.add_argument('--persam_f', action='store_true')
    parser.add_argument('--mask_image_only', action='store_true')
    parser.add_argument('--make_joblib', action='store_true')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    settings_file_path = "config_cfil.json"

    json_file = open(settings_file_path, "r")
    json_dict = json.load(json_file)
    # file_path = os.path.join(*["CFIL_for_NIP","train_data", json_dict["train_data_file"]])
    file_path = os.path.join(*["CFIL_for_NIP","train_data", args.data_dir])
 
    cl = LearnCFIL(memory_size=json_dict["memory_size"], 
                   batch_size=json_dict["batch_size"], 
                   image_size=json_dict["image_size"], 
                   train_epochs=json_dict["train_epochs"],
                   use_sam=json_dict["use_sam"],
                #    sam_f=json_dict["sam_f"])
                    sam_f=args.persam_f)
    
    if json_dict["multi_data"]:
        file_paths = json_dict["train_data_files"]
        # if json_dict["use_sam"] is False:
        #     print("no use sam")
        #     for p in file_paths:
        #         file_path = os.path.join(*["CFIL_for_NIP","train_data", p])
        #         approach_memory = cl.makeJobLib(file_path=file_path)
        
        #     result_path = os.path.join(*["CFIL_for_NIP","train_data", "all_data"])
        #     cl.train(file_path=os.path.join(result_path, "no_sam"))
        
        for p in file_paths:
            file_path = os.path.join(*["CFIL_for_NIP","train_data", p])
            if args.use_persam_f:
                task_name = "persam_f"
            else:
                task_name = "persam"
            joblib_path = os.path.join(file_path, f"{task_name}.joblib")
            cl.load_joblib(joblib_path=joblib_path)
    
        result_path = os.path.join(*["CFIL_for_NIP","train_data", "all_data"])
        cl.train(file_path=os.path.join(result_path, task_name))

    else:
        task_name = "normal"
        if args.persam_f and args.mask_image_only:
            task_name = "persam_f_mask_image_only"
        elif args.persam_f:
            task_name = "persam_f"
        else:
            NotImplementedError

        print(f"task name: {task_name}")
        joblib_path = os.path.join(file_path, f"{task_name}.joblib")

        if args.make_joblib:
            print("make joblib")
            joblib = cl.makeJobLib(file_path=file_path, mask_image_only=args.mask_image_only)
            joblib.save_joblib(joblib_path)
        
        if args.train:
            cl.load_joblib(joblib_path=joblib_path)
            cl.train(file_path=os.path.join(file_path, task_name))