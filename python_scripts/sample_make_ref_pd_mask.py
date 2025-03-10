import numpy as np
import os
import cv2

import time

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("../")

from calib.rs_utils import RealSense

from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.4])
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
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 


# nishikadoma
crop_settings = [{"crop_size_x": 240, "crop_size_y": 240, "crop_center_x": 320, "crop_center_y": 240}]
# D405  tsu
# crop_settings = [{"crop_size": 260, "crop_center_x": 350, "crop_center_y": 240}]

cam = RealSense(crop_settings=crop_settings)
camera_id = 0

ratio = np.load("../calib/camera_info/ratio.npy")
# center_pixels = [cam.crop_settings[camera_id]["crop_center_x"], cam.crop_settings[camera_id]["crop_center_y"]]
diff_center = crop_settings[camera_id]["crop_center_x"] - 320 
# center_pixels = [crop_settings[camera_id]["crop_size"]//2, crop_settings[camera_id]["crop_size"]//2]
center_pixels = [crop_settings[camera_id]["crop_size_x"]//2 - diff_center, (crop_settings[camera_id]["crop_size_y"])//2]
print(center_pixels)

images, depth_images, mask_images, max_contours = cam.get_image()


ref_folder_path = "ref_pd"

output_path = 'outputs2'
file_name = "original"
ext = "jpg"

ref_save_folder_path = ref_folder_path
os.makedirs(ref_save_folder_path, exist_ok=True)


image = images[0]
cv2.imwrite(os.path.join(ref_save_folder_path, f"{file_name}.{ext}"), image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

p = []

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')

def on_click(event):
    x = event.xdata
    y = event.ydata
    print(x,y)
    p.append([x,y]) 
    plt.plot(x,y,marker='.', markersize=10, color='red')

    plt.draw()
 
        
plt.connect('button_press_event', on_click)
plt.show()



sam_checkpoint = "../sam/sam_vit_h.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

input_point = np.array([p[-1]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  

best_idx = np.argmax(scores)
final_mask = masks[best_idx]
masked_image = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
masked_image[final_mask, :] = image[final_mask, :]
cv2.imwrite(os.path.join(ref_folder_path, f"masked_image.{ext}"), cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
mask_colors[final_mask, :] = np.array([[0, 0, 128]])
cv2.imwrite(os.path.join(ref_folder_path, f"mask.{ext}"), mask_colors)