from per_segment_anything import sam_model_registry, SamPredictor
# from sam.predictor import SamPredictor
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sam.show import *
from scipy.ndimage import maximum_filter

# annotation_path: 参照画像が格納されているフォルダ名
# test_images_path: セグメンテーションしたい画像が格納されているフォルダ名（画像のファイル名は00.png、01.pngのような命名規則）
# output_path: セグメンテーション結果を格納するフォルダ名

class PerSAM:
    def __init__(self, annotation_path="sam\\ref", 
                 output_path="sam\\results"):
        self.annotation_path = annotation_path
        self.output_path = output_path
        self.ref_mask_area = None


    # def point_selection(self, mask_sim, topk=1):
    #     # Top-1 point selection
    #     w, h = mask_sim.shape
    #     topk_xy = mask_sim.flatten(0).topk(topk)[1]
    #     topk_x = (topk_xy // h).unsqueeze(0)
    #     topk_y = (topk_xy - topk_x * h)
    #     topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    #     topk_label = np.array([1] * topk)
    #     topk_xy = topk_xy.cpu().numpy()
            
    #     # Top-last point selection
    #     last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    #     last_x = (last_xy // h).unsqueeze(0)
    #     last_y = (last_xy - last_x * h)
    #     last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    #     last_label = np.array([0] * topk)
    #     last_xy = last_xy.cpu().numpy()
        
    #     return topk_xy, topk_label, last_xy, last_label
    
    def point_selection(self, mask_sim, topk=1):
        # Top-1 point selection
        w, h = mask_sim.shape
        topk_xy = mask_sim.flatten(0).topk(topk)[1]
        topk_x = (topk_xy // h).unsqueeze(0)
        topk_y = (topk_xy - topk_x * h)
        topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
        topk_label = np.array([1] * topk)
        topk_xy = topk_xy.cpu().numpy()
        
        return topk_xy, topk_label
    
    def point_selection2(self, mask_sim, topk=1):
        # Top-1 point selection
        w, h = mask_sim.shape
        topk_xy = mask_sim.flatten(0).topk(topk)[1]
        topk_x = (topk_xy // h).unsqueeze(0)
        topk_y = (topk_xy - topk_x * h)
        topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
        topk_label = np.array([1] * topk)
        topk_xy = topk_xy.cpu().numpy()
            
        # Top-last point selection
        last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
        last_x = (last_xy // h).unsqueeze(0)
        last_y = (last_xy - last_x * h)
        last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
        last_label = np.array([0] * topk)
        last_xy = last_xy.cpu().numpy()
        
        return topk_xy, topk_label, last_xy, last_label

    def loadSAM(self, sam_type="vit_b"):
        ref_image_path = os.path.join(self.annotation_path, 'original.png') #参照用の元画像
        ref_mask_path = os.path.join(self.annotation_path, 'mask.png') #参照用の元マスク画像
        os.makedirs(self.output_path, exist_ok=True)

        # Load images and masks
        rate = 0.2
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_image = cv2.resize(ref_image, None, fx=rate, fy=rate)
        ref_mask = cv2.imread(ref_mask_path)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
        ref_mask = cv2.resize(ref_mask, None, fx=rate, fy=rate)

        gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
        gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()

        self.ref_mask_area = np.count_nonzero(ref_mask)

        print(ref_image.shape, ref_mask.shape)
        print("======> Load SAM" )
        #学習済みモデルを指定
        if sam_type == "vit_h":
            sam_ckpt = os.path.join('sam','sam_vit_h.pth') 
        elif sam_type == "vit_l":
            sam_ckpt = os.path.join('sam','sam_vit_l.pth')
        elif sam_type == "vit_b":    
            sam_ckpt = os.path.join('sam','sam_vit_b.pth')
        elif sam_type == "vit_t":
            sam_ckpt = os.path.join('sam','mobile_sam.pt')
        
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
        sam.eval()
        self.predictor = SamPredictor(sam)
        print("======> Obtain Location Prior" )
        # Image features encoding
        ref_mask = self.predictor.set_image(ref_image, ref_mask)
        ref_feat = self.predictor.features.squeeze().permute(1, 2, 0)
        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
        ref_mask = ref_mask.squeeze()[0]
        # # Target feature extraction
        # self.target_feat = ref_feat[ref_mask > 0]
        # self.target_embedding = self.target_feat.mean(0).unsqueeze(0)
        # self.target_feat = self.target_embedding / self.target_embedding.norm(dim=-1, keepdim=True)
        # self.target_embedding = self.target_embedding.unsqueeze(0)

        # Target feature extraction(PerSam)
        self.target_feat = ref_feat[ref_mask > 0]
        self.target_embedding = self.target_feat.mean(0).unsqueeze(0).unsqueeze(0)
        self.target_feat_mean = self.target_feat.mean(0)
        self.target_feat_max = torch.max(self.target_feat, dim=0)[0]
        self.target_feat = (self.target_feat_max / 2 + self.target_feat_mean / 2).unsqueeze(0)
    
        self.weight = self.finetune_weight(ref_feat=ref_feat, ref_mask=ref_mask, ref_image=ref_image, gt_mask=gt_mask)

    # def executePerSAM(self, test_image, show_heatmap=False):             
    #     # Image feature encoding
    #     self.predictor.set_image(test_image)
    #     test_feat = self.predictor.features.squeeze()
    #     # Cosine similarity
    #     C, h, w = test_feat.shape
    #     test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    #     test_feat = test_feat.reshape(C, h * w)
    #     sim = self.target_feat @ test_feat
    #     sim = sim.reshape(1, 1, h, w)
    #     sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    #     sim = self.predictor.model.postprocess_masks(
    #                     sim,
    #                     input_size=self.predictor.input_size,
    #                     original_size=self.predictor.original_size).squeeze()
        
    #     if show_heatmap:
    #         self.heatmap = sim_to_heatmap(sim)

    #     # Positive-negative location prior
    #     # topk_xy_i, topk_label_i, last_xy_i, last_label_i = self.point_selection(sim, topk=1)
    #     # topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
    #     # topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)
    #     topk_xy, topk_label = self.point_selection(sim, topk=1)

    #     # Obtain the target guidance for cross-attention layers
    #     sim = (sim - sim.mean()) / torch.std(sim)
    #     sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
    #     attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)
    #     # First-step prediction
    #     masks, scores, logits, _ = self.predictor.predict(
    #         point_coords=topk_xy, 
    #         point_labels=topk_label, 
    #         multimask_output=True,
    #         attn_sim=attn_sim,  # Target-guided Attention
    #         target_embedding=self.target_embedding  # Target-semantic Prompting
    #     )
    #     best_idx = 0
    #     # Cascaded Post-refinement-1
    #     masks, scores, logits, _ = self.predictor.predict(
    #                 point_coords=topk_xy,
    #                 point_labels=topk_label,
    #                 mask_input=logits[best_idx: best_idx + 1, :, :], 
    #                 multimask_output=True)
    #     best_idx = np.argmax(scores)
    #     # Cascaded Post-refinement-2
    #     y, x = np.nonzero(masks[best_idx])
    #     x_min = x.min()
    #     x_max = x.max()
    #     y_min = y.min()
    #     y_max = y.max()
    #     input_box = np.array([x_min, y_min, x_max, y_max])
    #     masks, scores, logits, _ = self.predictor.predict(
    #         point_coords=topk_xy,
    #         point_labels=topk_label,
    #         box=input_box[None, :],
    #         mask_input=logits[best_idx: best_idx + 1, :, :], 
    #         multimask_output=True)
    #     best_idx = np.argmax(scores)

    #     return masks, best_idx, topk_xy, topk_label 

    def finetune_weight(self, ref_feat, ref_mask, ref_image, gt_mask):
        print("======> Obtain Location Prior" )
        # # Image features encoding
        # ref_mask = self.predictor.set_image(ref_image, ref_mask)
        # ref_feat = self.predictor.features.squeeze().permute(1, 2, 0)
        # ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
        # ref_mask = ref_mask.squeeze()[0]

        # Target feature extraction
        self.target_feat = ref_feat[ref_mask > 0]
        target_feat_mean = self.target_feat.mean(0)
        target_feat_max = torch.max(self.target_feat, dim=0)[0]
        self.target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

        # Cosine similarity
        h, w, C = ref_feat.shape
        self.target_feat = self.target_feat / self.target_feat.norm(dim=-1, keepdim=True)
        ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
        ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
        sim = self.target_feat @ ref_feat

        # if weight is not None:
        #     self.weights_np = weight
        #     self.weights = torch.tensor(weight).cuda()
        #     return None

        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.predictor.model.postprocess_masks(
                        sim,
                        input_size=self.predictor.input_size,
                        original_size=self.predictor.original_size).squeeze()

        # Positive location prior
        topk_xy, topk_label = self.point_selection_f(sim, topk=1)


        print('======> Start Training')
        lr = 1e-3
        train_epoch = 1000
        log_epoch = 200
        # Learnable mask weights
        mask_weights = Mask_Weights().cuda()
        mask_weights.train()

        optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epoch)

        for train_idx in range(train_epoch):

            # Run the decoder
            masks, scores, logits, logits_high = self.predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                multimask_output=True)
            logits_high = logits_high.flatten(1)

            # Weighted sum three-scale masks
            weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
            logits_high = logits_high * weights
            logits_high = logits_high.sum(0).unsqueeze(0)

            dice_loss = self.calculate_dice_loss(logits_high, gt_mask)
            focal_loss = self.calculate_sigmoid_focal_loss(logits_high, gt_mask)
            loss = dice_loss + focal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if train_idx % log_epoch == 0:
                print('Train Epoch: {:} / {:}'.format(train_idx, train_epoch))
                current_lr = scheduler.get_last_lr()[0]
                print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(), focal_loss.item()))


        mask_weights.eval()
        self.weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        self.weights_np = self.weights.detach().cpu().numpy()
        return self.weights_np

    def getTopKPoints(self, test_image):
        # Image feature encoding
        self.predictor.set_image(test_image)
        test_feat = self.predictor.features.squeeze()
        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = self.target_feat @ test_feat
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.predictor.model.postprocess_masks(
                        sim,
                        input_size=self.predictor.input_size,
                        original_size=self.predictor.original_size).squeeze()
        
        # topk_xy, topk_label = self.point_selection(sim, topk=1)
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = self.point_selection2(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)
        return topk_xy, topk_label, sim
    
    def getSAMMask(self, topk_xy, topk_label):
        # First-step prediction
        masks, scores, logits, logits_high = self.predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    multimask_output=True)
        # best_idx = np.argmax(scores)

             # Weighted sum three-scale masks
        logits_high = logits_high * self.weights.unsqueeze(-1)
        logit_high = logits_high.sum(0)
        mask = (logit_high > 0).detach().cpu().numpy()

        logits = logits * self.weights_np[..., None]
        logit = logits.sum(0)

         # Cascaded Post-refinement-1
        y, x = np.nonzero(mask)
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logit[None, :, :],
            multimask_output=True)
        best_idx = np.argmax(scores)

        # _input_boxs = []
        # _scores = []
        # _logits = []
        # for m,s,l in zip(masks, scores, logits):
        #     y, x = np.nonzero(m)
        #     x_min = x.min()
        #     x_max = x.max()
        #     y_min = y.min()
        #     y_max = y.max()
        #     area = np.count_nonzero(m)
        #     _input_box = np.array([x_min, y_min, x_max, y_max])
        #     print("score ", s, " input_box ", _input_box, " area ", area, " ref_area ", self.ref_mask_area) 
        #     # if area < 1e6 or 1e7 < area:
        #     if area < self.ref_mask_area*0.9 or self.ref_mask_area*1.1 < area:
        #         continue
        #     else:
        #         _input_boxs.append(_input_box)
        #         _scores.append(s)
        #         _logits.append(l)
        #         # input_box = _input_box
        # if len(_scores) > 0:
        #     best_idx = np.argmax(_scores)
        #     input_box = _input_boxs[np.argmax(_scores)]
        # else:
        #     return None, None

        # Cascaded Post-refinement-2
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :],
            multimask_output=True)
        # best_idx = np.argmax(scores)

        _scores = []
        _masks = []
        for m,s,l in zip(masks, scores, logits):
            y, x = np.nonzero(m)
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            area = np.count_nonzero(m)
            print("score ", s, " input_box ", "area ", area, " ref_area ", self.ref_mask_area) 
            # if area < 1e6 or 1e7 < area:
            if area < self.ref_mask_area*0.9 or self.ref_mask_area*1.1 < area:
                continue
            else:
                _scores.append(s)
                _masks.append(m)
        if len(_scores) > 0:
            best_idx = np.argmax(_scores)
            masks = _masks
        else:
            return None, None
        print(f"best index {best_idx}")

        return masks, best_idx
    
    def getSAMMask2(self, topk_xy, topk_label, sim):
        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

        # First-step prediction
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy, 
            point_labels=topk_label, 
            multimask_output=False,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=self.target_embedding  # Target-semantic Prompting
        )
        best_idx = 0

        # Cascaded Post-refinement-1
        masks, scores, logits, _ = self.predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    mask_input=logits[best_idx: best_idx + 1, :, :], 
                    multimask_output=True)
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :],
            multimask_output=True)
        # best_idx = np.argmax(scores)

        _scores = []
        _masks = []
        for m,s,l in zip(masks, scores, logits):
            y, x = np.nonzero(m)
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            area = np.count_nonzero(m)
            print("score ", s, " input_box ", "area ", area, " ref_area ", self.ref_mask_area) 
            # if area < 1e6 or 1e7 < area:
            if area < self.ref_mask_area*0.9 or self.ref_mask_area*1.1 < area:
                continue
            else:
                _scores.append(s)
                _masks.append(m)
        if len(_scores) > 0:
            best_idx = np.argmax(_scores)
            masks = _masks
        else:
            return None, None
        print(f"best index {best_idx}")

        return masks, best_idx
    
    def getRefMaskArea(self):
        return self.ref_mask_area
    
    def executePerSAM(self, test_image, show_heatmap=False):             
        # Image feature encoding
        self.predictor.set_image(test_image)
        test_feat = self.predictor.features.squeeze()
        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = self.target_feat @ test_feat
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.predictor.model.postprocess_masks(
                        sim,
                        input_size=self.predictor.input_size,
                        original_size=self.predictor.original_size).squeeze()
        
        if show_heatmap:
            self.heatmap = sim_to_heatmap(sim)

        # Positive-negative location prior
        # topk_xy_i, topk_label_i, last_xy_i, last_label_i = self.point_selection(sim, topk=1)
        # topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        # topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)
        topk_xy, topk_label = self.point_selection(sim, topk=1)

        # First-step prediction
        masks, scores, logits, logits_high = self.predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    multimask_output=True)
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :],
            multimask_output=True)
        best_idx = np.argmax(scores)
        # print("scores", scores)

        return masks, best_idx, topk_xy, topk_label

    def save_masked_image(self, final_mask, test_image, name, output=True):
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = test_image[final_mask, :]
        if output:
            cv2.imwrite(os.path.join(self.output_path, name), mask_colors)
        return mask_colors
    
    def save_randomback_image(self, final_mask, test_image, name):
        mask_colors = np.random.randint(0, 255, (final_mask.shape[0], final_mask.shape[1], 3))
        mask_colors[final_mask, :] = test_image[final_mask, :]
        print(os.path.join(self.output_path, name))
        cv2.imwrite(os.path.join(self.output_path, name), mask_colors)
        return mask_colors
    
    def save_randomfig_image(self, final_mask, test_image, name):
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        for i in range(5):
            # cv2.rectangle(mask_colors, np.random.randint(0,256,2).tolist(), np.random.randint(0,256,2).tolist(), np.random.randint(0,255,3).tolist())
            cv2.circle(mask_colors, np.random.randint(0,256,2).tolist(), np.random.randint(1,100), np.random.randint(0,255,3).tolist(), thickness=5)
        mask_colors[final_mask, :] = test_image[final_mask, :]
        cv2.imwrite(os.path.join(self.output_path, name), mask_colors)
        return mask_colors

    def save_masks(self, masks, best_idx, test_image, topk_xy, topk_label, test_idx):
        # Save masks
        plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        show_mask(masks[best_idx], plt.gca())
        show_points(topk_xy, topk_label, plt.gca())
        plt.title(f"Mask {best_idx}", fontsize=18)
        plt.axis('off')
        vis_mask_output_path = os.path.join(self.output_path, f'vis_mask_{test_idx}.jpg')
        with open(vis_mask_output_path, 'wb') as outfile:
            plt.savefig(outfile, format='jpg')
        final_mask = masks[best_idx]
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = np.array([[0, 0, 128]])
        mask_output_path = os.path.join(self.output_path, "mask_" + test_idx + '.png')
        cv2.imwrite(mask_output_path, mask_colors)

        mask_colors = np.zeros((test_image.shape[0], test_image.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = test_image[final_mask, :]
        cv2.imwrite(os.path.join(self.output_path, test_idx + '.png'), mask_colors)
        return mask_colors


    def testPerSAM(self, test_images_path="CFIL_for_NIP\\train_data\\20240917_182254_514\\image\\"):
        print('======> Start Testing')
        for test_idx in tqdm(range(len(os.listdir(test_images_path)))):
            test_idx = '%04d' % test_idx
            test_image_path = test_images_path + 'image_' + test_idx + '.jpg'
            test_image = cv2.imread(test_image_path)
            test_image = cv2.resize(test_image, (256,256), interpolation=cv2.INTER_CUBIC)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            masks, best_idx, topk_xy, topk_label = self.executePerSAM(test_image)
            
            self.save_masks(masks, best_idx, test_image, topk_xy, topk_label, test_idx)

    def loadSAM_f(self, weight=None, image_size=None):
        ref_image_path = os.path.join(self.annotation_path, 'original.jpg') #参照用の元画像
        ref_mask_path = os.path.join(self.annotation_path, 'mask.jpg') #参照用の元マスク画像
        os.makedirs(self.output_path, exist_ok=True)

        # Load images and masks
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_mask = cv2.imread(ref_mask_path)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
        if image_size is None:
            ref_image = cv2.resize(ref_image,None,fx=0.1,fy=0.1)
            ref_mask = cv2.resize(ref_mask,None,fx=0.1,fy=0.1)
        else:
            ref_image = cv2.resize(ref_image, image_size)
            ref_mask = cv2.resize(ref_mask, image_size)
        gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
        gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()

        print("======> Load SAM" )
        # sam_type, sam_ckpt = 'vit_h', os.path.join('sam','sam_vit_h.pth') #学習済みモデルを指定
        sam_type, sam_ckpt = 'vit_t', os.path.join('sam','mobile_sam.pt') #学習済みモデルを指定
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
        sam.eval()
        self.predictor = SamPredictor(sam)

        print("======> Obtain Location Prior" )
        # Image features encoding
        ref_mask = self.predictor.set_image(ref_image, ref_mask)
        ref_feat = self.predictor.features.squeeze().permute(1, 2, 0)
        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
        ref_mask = ref_mask.squeeze()[0]
        
        # Target feature extraction
        self.target_feat = ref_feat[ref_mask > 0]
        target_feat_mean = self.target_feat.mean(0)
        target_feat_max = torch.max(self.target_feat, dim=0)[0]
        self.target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

        # Cosine similarity
        h, w, C = ref_feat.shape
        self.target_feat = self.target_feat / self.target_feat.norm(dim=-1, keepdim=True)
        ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
        ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
        sim = self.target_feat @ ref_feat

        if weight is not None:
            self.weights_np = weight
            self.weights = torch.tensor(weight).cuda()
            return None

        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.predictor.model.postprocess_masks(
                        sim,
                        input_size=self.predictor.input_size,
                        original_size=self.predictor.original_size).squeeze()

        # Positive location prior
        topk_xy, topk_label = self.point_selection_f(sim, topk=1)


        print('======> Start Training')
        lr = 1e-3
        train_epoch = 1000
        log_epoch = 200
        # Learnable mask weights
        mask_weights = Mask_Weights().cuda()
        mask_weights.train()

        optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epoch)

        for train_idx in range(train_epoch):

            # Run the decoder
            masks, scores, logits, logits_high = self.predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                multimask_output=True)
            logits_high = logits_high.flatten(1)

            # Weighted sum three-scale masks
            weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
            logits_high = logits_high * weights
            logits_high = logits_high.sum(0).unsqueeze(0)

            dice_loss = self.calculate_dice_loss(logits_high, gt_mask)
            focal_loss = self.calculate_sigmoid_focal_loss(logits_high, gt_mask)
            loss = dice_loss + focal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if train_idx % log_epoch == 0:
                print('Train Epoch: {:} / {:}'.format(train_idx, train_epoch))
                current_lr = scheduler.get_last_lr()[0]
                print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(), focal_loss.item()))


        mask_weights.eval()
        self.weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        self.weights_np = self.weights.detach().cpu().numpy()
        return self.weights_np

    def point_selection_f(self, mask_sim, topk=1):
        # Top-1 point selection
        w, h = mask_sim.shape
        topk_xy = mask_sim.flatten(0).topk(topk)[1]
        topk_x = (topk_xy // h).unsqueeze(0)
        topk_y = (topk_xy - topk_x * h)
        topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
        topk_label = np.array([1] * topk)
        topk_xy = topk_xy.cpu().numpy()
        
        return topk_xy, topk_label


    def calculate_dice_loss(self, inputs, targets, num_masks = 1):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks


    def calculate_sigmoid_focal_loss(self, inputs, targets, num_masks = 1, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_masks
    
    def executePerSAM_f(self, test_image, show_heatmap=False):             
        # Image feature encoding
        self.predictor.set_image(test_image)
        test_feat = self.predictor.features.squeeze()
        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = self.target_feat @ test_feat
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.predictor.model.postprocess_masks(
                        sim,
                        input_size=self.predictor.input_size,
                        original_size=self.predictor.original_size).squeeze()
        
        if show_heatmap:
            self.heatmap = sim_to_heatmap(sim)
        
        # Positive-negative location prior
        topk_xy, topk_label = self.point_selection_f(sim, topk=1)
        
        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)
        
        # First-step prediction
        masks, scores, logits, logits_high = self.predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    multimask_output=True)
        
        # Weighted sum three-scale masks
        logits_high = logits_high * self.weights.unsqueeze(-1)
        logit_high = logits_high.sum(0)
        mask = (logit_high > 0).detach().cpu().numpy()

        logits = logits * self.weights_np[..., None]
        logit = logits.sum(0)


        # Cascaded Post-refinement-1
        y, x = np.nonzero(mask)
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logit[None, :, :],
            multimask_output=True)
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :],
            multimask_output=True)
        best_idx = np.argmax(scores)
        
        return masks, best_idx, topk_xy, topk_label
    
    def save_heatmap(self, name):
        cv2.imwrite(os.path.join(self.output_path, name), cv2.cvtColor(self.heatmap, cv2.COLOR_RGB2BGR))
    

    def loadPositionDetector(self):
        # Load images and masks
        print(os.path.join(self.annotation_path+"_pd", 'original.jpg'))
        ref_image_path = os.path.join(self.annotation_path+"_pd", 'original.jpg') #参照用の元画像
        ref_mask_path = os.path.join(self.annotation_path+"_pd", 'mask.jpg') #参照用の元マスク画像
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_mask = cv2.imread(ref_mask_path)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

        print("======> Obtain Location Prior" )
        # Image features encoding
        ref_mask = self.predictor.set_image(ref_image, ref_mask)
        ref_feat = self.predictor.features.squeeze().permute(1, 2, 0)
        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
        ref_mask = ref_mask.squeeze()[0]
        # # Target feature extraction
        # self.target_feat_pd = ref_feat[ref_mask > 0]
        # self.target_embedding_pd = self.target_feat_pd.mean(0).unsqueeze(0)
        # self.target_feat_pd = self.target_embedding_pd / self.target_embedding_pd.norm(dim=-1, keepdim=True)

        # Target feature extraction(PerSam)
        self.target_feat_pd = ref_feat[ref_mask > 0]
        self.target_embedding_pd = self.target_feat_pd.mean(0).unsqueeze(0).unsqueeze(0)
        self.target_feat_mean_pd = self.target_feat_pd.mean(0)
        self.target_feat_max_pd = torch.max(self.target_feat_pd, dim=0)[0]
        self.target_feat_pd = (self.target_feat_max_pd / 2 + self.target_feat_mean_pd / 2).unsqueeze(0)

    def getSimirality(self, test_image, save_sim=False):
        # Image feature encoding
        self.predictor.set_image(test_image)
        test_feat = self.predictor.features.squeeze()
        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = self.target_feat_pd @ test_feat
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.predictor.model.postprocess_masks(
                        sim,
                        input_size=self.predictor.input_size,
                        original_size=self.predictor.original_size).squeeze()
        if save_sim:
            self.heatmap = sim_to_heatmap(sim)
            self.save_heatmap("positon_detector_similarity.jpg")
        return sim
    
    def getPeaks(self, test_image, filter_size=100, order=0.7, save_sim=False):
        sim = self.getSimirality(test_image)
        peaks_index = detect_peaks(sim.cpu().detach().numpy(), order=order, filter_size=filter_size)
        if save_sim:
            self.heatmap = sim_to_heatmap(sim)
            # plt.imshow(self.heatmap)
            for i in range(len(peaks_index[0])):
                cv2.circle(self.heatmap, (peaks_index[1][i], peaks_index[0][i]), radius=3, color=(0, 0, 0), thickness=-1)
                cv2.putText(self.heatmap, 'PEAK!!!', (peaks_index[1][i], peaks_index[0][i]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)    
                # plt.scatter(peaks_index[1][i], peaks_index[0][i], color='black', s=5)
                # plt.text(peaks_index[1][i],peaks_index[0][i], 'PEAK!!!', fontsize=9)
            # plt.axis("off")
            # plt.savefig("positon_detector_similarity.jpg")
            # plt.close()
            cv2.imwrite("positon_detector_original.jpg", test_image)   
            cv2.imwrite("positon_detector_similarity.jpg", cv2.cvtColor(self.heatmap, cv2.COLOR_RGB2BGR))   
        return peaks_index
    
    def getObjects(self, image, filter_size=20, order=0.7, save_sim=False):
        sim = self.getSimirality(image)
        heatmap = sim_to_heatmap(sim, th=order)

        image_hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)
        img_H, img_S, img_V = cv2.split(image_hsv)
        _thre, image_mask = cv2.threshold(img_H, 0, 30, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) #img_binaryを輪郭抽出
        image_contours = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 2) #抽出した輪郭を緑色でimg_colorに重ね書き
        x_list = []
        y_list = []
        for i in range(0, len(contours)):
            if len(contours[i]) > 0:
                if cv2.contourArea(contours[i]) < 20 :
                    continue
                # 重心の計算
                m = cv2.moments(contours[i])
                x,y= m['m10']/m['m00'] , m['m01']/m['m00']
                # print(f"Weight Center = ({x}, {y})")
                # 座標を四捨五入
                x, y = round(x), round(y)
                x_list.append(x)
                y_list.append(y)

        peaks_index = np.array([y_list, x_list]).T
        peaks_index = sorted(peaks_index, key=lambda x: x[1])
        peaks_index = np.array(peaks_index).T

        if save_sim:
            cv2.imwrite("positon_detector_original.jpg", image)   
            cv2.imwrite("positon_detector_similarity.jpg", cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))   
            cv2.imwrite("positon_detector_Contours.jpg", image_contours)

        return peaks_index

def sim_to_heatmap(sim, th=0):
    if torch.is_tensor(sim):
        x = sim.to("cpu").detach().numpy().copy()
    else:
        x = sim.copy()
    h, w = x.shape
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x[x < th] = 0
    x = (x * 255).reshape(-1)
    cm = plt.get_cmap("jet")
    x = np.array([cm(int(np.round(xi)))[:3] for xi in x])
    return (x * 255).astype(np.uint8).reshape(h, w, 3)

class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


# ピーク検出関数
def detect_peaks(image, filter_size=3, order=0.5):
    local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image, mask=~(image == local_max))

    # 小さいピーク値を排除（最大ピーク値のorder倍のピークは排除）
    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * order))
    peaks_index = np.where((temp.mask != True))
    peaks_index =np.array(peaks_index).T
    peaks_index = sorted(peaks_index, key=lambda x: x[1])
    peaks_index =np.array(peaks_index).T
    return peaks_index

def camera_demo():
    import cv2
    perSam = PerSAM(annotation_path="sam\\ref2")
    perSam.loadSAM_f()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # カメラ画像の横幅を設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # カメラ画像の縦幅を設定

    crop_x = [200, 1220]
    crop_y = [150, 950]
    # crop_x = [400, 600]
    # crop_y = [200, 400]

    #OpenCVのタイマーの準備
    timer = cv2.TickMeter()
    timer.start()

    #各変数の初期値設定
    count = 0
    max_count =30
    fps = 0


    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (1000, 600))
        frame = frame[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
        masks, best_idx, topk_xy, topk_label = perSam.executePerSAM_f(frame, show_heatmap=False)
        frame = perSam.save_masked_image(masks[best_idx], frame, "test.jpg", output=False)
        # perSam.save_heatmap("similarity.jpg")
        if count == max_count:
            timer.stop()
            fps = max_count / timer.getTimeSec()
            print("FPS(Actual):" , '{:11.02f}'.format(fps))        
            #リセットと再スタート
            timer.reset()
            count = 0
            timer.start()
        count += 1
        cv2.imshow("Web Camera movie", frame)
        
        i = cv2.waitKey(1)
        if i == 27 or i == 13: # presss "ESC or Enter" to exit
            cv2.imwrite("test.jpg", frame)
            break


if __name__ == "__main__":
    # perSam = PerSAM()
    # perSam.loadSAM()
    # perSam.testPerSAM()

    camera_demo()