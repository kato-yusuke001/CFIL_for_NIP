import cv2
from cv2 import aruco
import numpy as np
from rs_utils import RealSense

X1, Y1, = 0, 0
CLICKED = 0
def mouseEvents(event, x, y, flags, param):
    global X1, Y1, CLICKED
    try:
        if event == cv2.EVENT_LBUTTONDOWN:
            X1, Y1 = x, y
            print(X1, Y1)
            CLICKED += 1
            
    except Exception as e:
        print(e)

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouseEvents)

# マーカー種類を定義
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# ArucoDetectorオブジェクトを作成
detector = aruco.ArucoDetector(dictionary, parameters)

# Webカメラをキャプチャ
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

rs_settings= {"width": 640, "height": 480, "fps": 90, "clipping_distance": 1.0,
                  "crop_settings":[{'crop_size_x': 264, 'crop_size_y': 191, 'crop_center_x': 338, 'crop_center_y': 212}]}
cap = RealSense(**rs_settings)

qr4_x = 0.175
qr4_y = -0.450

b_t_t = {}
b_t_t["04"] = [qr4_x, qr4_y, -0.006]
marker_04 = b_t_t["04"]
b_t_t["00"] = [marker_04[0]+0.05, marker_04[1]-0.05, marker_04[2]]
b_t_t["01"] = [marker_04[0]-0.05, marker_04[1]-0.05, marker_04[2]]
b_t_t["02"] = [marker_04[0]+0.05, marker_04[1]+0.05, marker_04[2]]
b_t_t["03"] = [marker_04[0]-0.05, marker_04[1]+0.05, marker_04[2]]

while True:
    # フレームを取得
    frames, depth_images, _, _ = cap.get_image(crop=True, get_mask=False)
    frame = frames[0]
    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # マーカーを検出
    corners, ids, rejectedCandidates = detector.detectMarkers(gray)

    # 検出したマーカーを描画
    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        # print(f"検出されたマーカーID: {ids.flatten()}")
    
    if CLICKED > 0:
        cv2.circle(frame, (X1, Y1), 5, (0, 0, 255), -1)
        
        center_id = 4
        ratio = []
        for i,c in zip(ids.ravel(), corners):
            # print(i,c)
            if i == center_id:
                continue
            
            diff_X = abs(b_t_t["{:02}".format(i)][0] - b_t_t["{:02}".format(center_id)][0])
            diff_Y = abs(b_t_t["{:02}".format(i)][1] - b_t_t["{:02}".format(center_id)][1])
            diff_x = abs(c[0].mean(axis=0)[0] - corners[ids.ravel().tolist().index(center_id)][0].mean(axis=0)[0])
            diff_y = abs(c[0].mean(axis=0)[1] - corners[ids.ravel().tolist().index(center_id)][0].mean(axis=0)[1])

            ratio.append([diff_X/diff_x, diff_Y/diff_y])
        ratio = np.average(ratio, axis=0)
        # print("ratio:", ratio)
        np.save("ratio.npy", ratio)

        target_id = 3
        diff_x = corners[ids.ravel().tolist().index(target_id)][0].mean(axis=0)[0] - corners[ids.ravel().tolist().index(center_id)][0].mean(axis=0)[0]
        diff_y = corners[ids.ravel().tolist().index(target_id)][0].mean(axis=0)[1] - corners[ids.ravel().tolist().index(center_id)][0].mean(axis=0)[1]
        X = b_t_t["{:02}".format(center_id)][0] + diff_x*ratio[0] 
        Y = b_t_t["{:02}".format(center_id)][1] - diff_y*ratio[1]
        print("recall X:{}, Y:{}".format(X, Y))
        print("actual X:{}, Y:{}".format(b_t_t["{:02}".format(target_id)][0], b_t_t["{:02}".format(target_id)][1]))


        image_size_y = frame.shape[0]
        image_size_x = frame.shape[1]
        image_size = [image_size_x, image_size_y]

        qr4_x_in_image_x = corners[ids.ravel().tolist().index(center_id)][0].mean(axis=0)[0]
        qr4_x_in_image_y = corners[ids.ravel().tolist().index(center_id)][0].mean(axis=0)[1]
        qr4_x_in_image = [qr4_x_in_image_x, qr4_x_in_image_y]

        center_pos = [qr4_x + ((image_size[0]/2)-qr4_x_in_image[0])*ratio[0], qr4_y - ((image_size[1]/2)-qr4_x_in_image[1])*ratio[1]]
        print("center_pos:", center_pos)

    # フレームを表示
    cv2.imshow('frame', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.close()
cv2.destroyAllWindows()