from segment_anything import sam_model_registry, SamPredictor
import torch
from torch.nn import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm



#セグメンテーション結果の推定部の表示や保存のためのユーティリティ
def save_mask(test_idx, mask, output_path=""):
    final_mask = mask
    mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
    mask_colors[final_mask, :] = np.array([[0, 0, 128]])
    mask_output_path = os.path.join(output_path, str(test_idx) + '_mask.jpg')
    cv2.imwrite(mask_output_path, mask_colors)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', acecolor=(0,0,0,0), lw=2)) 

device = "cuda" if torch.cuda.is_available() else "cpu"  


#学習済みモデルの読み込み(モデルの場所は任意)
sam_checkpoint = "sam\\sam_vit_b.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

#セグメンテーションしたい画像を読み込み
image = cv2.imread('CFIL_for_NIP\\train_data\\20240925_170444_287\\initial_image.jpg')
image = cv2.resize(image, (256,256), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("CFIL_for_NIP\\train_data\\20240925_170444_287\\original.jpg", image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image) # 画像をembeddingにする

#画像上の座標の指示（プロンプト）ここは手動で座標を決める必要あり。
input_point = np.array([[150, 150]])
input_label = np.array([1])

#セグメンテーションの実行
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

#スコアの高い3つの結果の表示と保存
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    # plt.show()  
    plt.savefig(os.path.join("sam", str(i)+"_res.jpg"))
    save_mask(i, mask, output_path="sam")