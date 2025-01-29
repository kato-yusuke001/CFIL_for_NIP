import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import json
# sys.path.append('segment-anything')
from segment_anything import sam_model_registry, SamPredictor

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
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

settings_file_path = "cfil_config.json"
json_file = open(settings_file_path, "r")
json_dict = json.load(json_file)
file_path = os.path.join("CFIL_for_NIP\\train_data", json_dict["train_data_file"])

image_path = os.path.join(file_path, "initial_image.jpg")
save_path = os.path.join(file_path, "ref")
os.makedirs(save_path, exist_ok=True)


image = cv2.imread(image_path)
cv2.imwrite(os.path.join(save_path, "original.jpg"), image)


input_point = None
rate = 10
def onMouse(event, x, y, flags, params):
    global input_point
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        input_point = np.array([[x*rate, y*rate]])

while True:
    cv2.imshow('original', cv2.resize(image, (image.shape[1]//rate, image.shape[0]//rate)))
    cv2.setMouseCallback('original', onMouse)

    if cv2.waitKey(1):
        if input_point is not None:
            cv2.destroyWindow('original')
            break



image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "sam\\sam_vit_h.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

predictor.set_image(image)

# input_point = np.array([[2700, 2000]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

plt.figure(figsize=(15,5))
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.subplot(1, len(masks), i+1)
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
save_masked_image_path = os.path.join(save_path, "masked_image.jpg")
cv2.imwrite(save_masked_image_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
mask_colors[final_mask, :] = np.array([[0, 0, 128]])
save_mask_image_path = os.path.join(save_path, "mask.jpg")
cv2.imwrite(save_mask_image_path, mask_colors)