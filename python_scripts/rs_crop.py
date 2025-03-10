import time
import numpy as np
import cv2

import sys
sys.path.append("../")

from calib.rs_utils import RealSense

Xc, Yc, X1, Y1, X2, Y2, Xt, Yt = 0, 0, 0, 0, 0, 0, 0, 0
CLICKED = 0
def mouseEvents(event, x, y, flags, param):
    global Xc, Yc, X1, Y1, X2, Y2, Xt, Yt, CLICKED
    Xt, Yt = x, y
    try:
        if event == cv2.EVENT_LBUTTONDOWN:
            if CLICKED == 0:
                X1, Y1 = x, y
                print("Click RIGHT BOTTOM of the target object. Press [q] to stop making masks.")
            elif CLICKED == 1:
                X2, Y2 = x, y
            CLICKED += 1
    except Exception as e:
        print(e)

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouseEvents)
    
rs = RealSense(crop_settings=False)
while True:
    before = time.time()
    color_images, depth_images, mask_images, max_contours = rs.get_image()
    image = color_images[0]
    H, W = image.shape[0], image.shape[1]
    cv2.line(image, (Xt, 0), (Xt, H), (0, 0, 255), 2)
    cv2.line(image, (0, Yt), (W, Yt), (0, 0, 255), 2)

    if CLICKED == 3:
        break
        
    if CLICKED == 1:
        cv2.rectangle(image, (X1, Y1), (Xt, Yt), color=(0, 255, 0), thickness=3, lineType=cv2.LINE_4, shift=0)
    
    if CLICKED >= 2:
        cv2.rectangle(image, (X1, Y1), (X2, Y2), color=(0, 255, 0), thickness=3, lineType=cv2.LINE_4, shift=0)
    
    cv2.imshow("image", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    dt = time.time() - before
    print("FPS:", 1/dt)

cv2.destroyAllWindows()

print("X1:{}, Y1:{}, X2:{}, Y2:{}".format(X1, Y1, X2, Y2))
crop_settings = [{"crop_size_x": X2-X1, "crop_size_y": Y2-Y1, "crop_center_x": (X1+X2)//2, "crop_center_y": (Y1+Y2)//2}]    
print(crop_settings)