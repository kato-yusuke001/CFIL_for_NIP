# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np

THICKNESS = 100
END_POINT = 150

# ベクトルを描画する
def drawAxis(img, start_pt, vec, colour, length, rate):
    # アンチエイリアス
    CV_AA = 16

    # 終了点
    end_pt = (int(start_pt[0] + length * vec[0]), int(start_pt[1] + length * vec[1]))

    # 中心を描画
    cv2.circle(img, (int(start_pt[0]), int(start_pt[1])), 5, colour, 1)

    # 軸線を描画
    cv2.line(img, (int(start_pt[0]), int(start_pt[1])), end_pt, colour, int(THICKNESS*rate), CV_AA)

    # 先端の矢印を描画
    angle = math.atan2(vec[1], vec[0])
    # print(angle)

    qx0 = int(end_pt[0] - END_POINT * rate * math.cos(angle + math.pi / 4))
    qy0 = int(end_pt[1] - END_POINT * rate * math.sin(angle + math.pi / 4))
    cv2.line(img, end_pt, (qx0, qy0), colour, int(THICKNESS*rate), CV_AA)

    qx1 = int(end_pt[0] - END_POINT * rate * math.cos(angle - math.pi / 4))
    qy1 = int(end_pt[1] - END_POINT * rate * math.sin(angle - math.pi / 4))
    cv2.line(img, end_pt, (qx1, qy1), colour, int(THICKNESS*rate), CV_AA)

    return angle

def pca(image, rate=1):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ２値化
    retval, bw = cv2.threshold(gray, 50*rate, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 輪郭を抽出
    #   contours : [領域][Point No][0][x=0, y=1]
    #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
    #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # 各輪郭に対する処理
    for i in range(0, len(contours)):

        # 輪郭の領域を計算
        area = cv2.contourArea(contours[i])
        
        # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
        if area < 1e5*rate or 1e10*rate < area:
            continue
        # if area < 1e4 or 1e9 < area:
        #     continue
        # print(area)
        
        # 輪郭を描画する
        cv2.drawContours(image, contours, i, (0, 0, 255), 2, 8, hierarchy, 0)

        x, y, w, h = cv2.boundingRect(contours[i])
        box = cv2.minAreaRect(contours[i])
        points = cv2.boxPoints(box)
        
        # 輪郭データを浮動小数点型の配列に格納
        X = np.array(contours[i], dtype=np.float32).reshape((contours[i].shape[0], contours[i].shape[2]))
        
        # PCA（１次元）
        mean, eigenvectors = cv2.PCACompute(X, mean=np.array([], dtype=np.float32), maxComponents=1)
        
        # 主成分方向のベクトルを描画
        pt = (mean[0][0], mean[0][1])
        vec = (eigenvectors[0][0], eigenvectors[0][1])
        angle = drawAxis(image, pt, vec, (255, 255, 0), 500*rate, rate)
        # print(angle, box)
    return image, pt, angle, (x,y), box, points