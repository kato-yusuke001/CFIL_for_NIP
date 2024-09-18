from per_segment_anything import sam_model_registry, SamPredictor
# from sam.predictor import SamPredictor
import torch
from torch.nn import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sam.show import *

# annotation_path: 参照画像が格納されているフォルダ名
# test_images_path: セグメンテーションしたい画像が格納されているフォルダ名（画像のファイル名は00.png、01.pngのような命名規則）
# output_path: セグメンテーション結果を格納するフォルダ名

class PerSAM:
    def __init__(self, annotation_path="sam\\ref", 
                 output_path="sam\\results"):
        self.annotation_path = annotation_path
        self.output_path = output_path


    def point_selection(self, mask_sim, topk=1):
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

    def loadSAM(self):
        ref_image_path = os.path.join(self.annotation_path, 'original.jpg') #参照用の元画像
        ref_mask_path = os.path.join(self.annotation_path, 'mask.jpg') #参照用の元マスク画像
        os.makedirs(self.output_path, exist_ok=True)

        # Load images and masks
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_mask = cv2.imread(ref_mask_path)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
        print(ref_image.shape, ref_mask.shape)
        print("======> Load SAM" )
        sam_type, sam_ckpt = 'vit_h', 'sam\\sam_vit_h.pth' #学習済みモデルを指定
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
        self.target_embedding = self.target_feat.mean(0).unsqueeze(0)
        self.target_feat = self.target_embedding / self.target_embedding.norm(dim=-1, keepdim=True)
        self.target_embedding = self.target_embedding.unsqueeze(0)

    def executePerSAM(self, test_image):             
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
        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = self.point_selection(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)
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
        best_idx = np.argmax(scores)

        return masks, best_idx, topk_xy, topk_label
    
    def save_masked_image(self, final_mask, test_image, name):
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = test_image[final_mask, :]
        cv2.imwrite(os.path.join(self.output_path, name), mask_colors)
        return mask_colors
    
    def save_randomback_image(self, final_mask, test_image, name):
        mask_colors = np.random.randint(0, 255, (final_mask.shape[0], final_mask.shape[1], 3))
        mask_colors[final_mask, :] = test_image[final_mask, :]
        cv2.imwrite(os.path.join(self.output_path, name), mask_colors)
        return mask_colors
    
    def save_randomfig_image(self, final_mask, test_image, name):
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        for i in range(40):
            cv2.rectangle(mask_colors, np.random.randint(0,256,2).tolist(), np.random.randint(0,256,2).tolist(), np.random.randint(0,255,3).tolist(), thickness=-1)
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

if __name__ == "__main__":
    perSam = PerSAM()
    perSam.loadSAM()
    perSam.testPerSAM()