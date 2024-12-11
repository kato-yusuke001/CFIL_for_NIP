import torch
import cv2
import os
from memory import ApproachMemory
from PIL import Image

memory_size=2.5e6
device ="cuda" if torch.cuda.is_available() else "cpu"

approach_memory = ApproachMemory(memory_size, device)
last_inch_memory = ApproachMemory(memory_size, device)

path = "logs/20240327-1139"
approach_memory.load_joblib(os.path.join(path, 'approach.joblib'))
last_inch_memory.load_joblib(os.path.join(path, 'last_inch.joblib'))

print(approach_memory._n)

approach_save_path = os.path.join(path, "imgs/approach")
if not os.path.exists(approach_save_path):
    os.makedirs(approach_save_path
)

approach_imgs = approach_memory['image']
for n in range(approach_memory._n):
    approach_img = approach_imgs[n]
    cv2.imwrite(os.path.join(approach_save_path, 'img_{0:04}.jpg'.format(n)), approach_img)
    # Image.fromarray(approach_img).save(os.path.join(approach_save_path, 'img_{0:04}.png'.format(n)))

last_inch_save_path = os.path.join(path, "imgs/last_inch")
if not os.path.exists(last_inch_save_path):
    os.makedirs(last_inch_save_path
)

last_inch_imgs = last_inch_memory['image']
# positions = last_inch_memory['position']
# print(positions[:10])
for n in range(last_inch_memory._n):
    last_inch_img = last_inch_imgs[n]
    try:
        cv2.imwrite(os.path.join(last_inch_save_path, 'img_{0:04}.jpg'.format(n)), last_inch_img)
    except:
        print(last_inch_img)
    # Image.fromarray(last_inch_img).save(os.path.join(last_inch_save_path, 'img_{0:04}.png'.format(n)))
