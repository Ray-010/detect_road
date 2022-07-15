import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import statistics
import math
#対象画像読み込み
img=cv2.imread("./images/miti1.jpg")

img_b=cv2.medianBlur(img,3)
#カーネルの設定
kernel=np.ones((9,9),dtype=np.uint8)
#色を検出する関数
def detect_miti_color(img_b):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 灰？色のHSVの値域1
    hsv_min = np.array([15, 0, 0])
    hsv_max = np.array([70,50,255])

    # 灰色領域のマスク（）
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img

miti_mask, miti_masked_img = detect_miti_color(img)

img_e=cv2.dilate(miti_mask,kernel)

contours,_ = cv2.findContours(img_e, 1, 2)

#塗りつぶし多角形を描写するためのゼロ埋め配列定義
#point:opencvの関数で扱えるように型をuint8で指定！
zero_img = np.zeros([miti_mask.shape[0], miti_mask.shape[1]], dtype="uint8")

#全ての輪郭座標配列を使って塗りつぶし多角形を描写
for p in contours:
    img_final=cv2.fillPoly(zero_img, [p], (255, 255, 255))

#ノイズ処理　収縮
img_e2=cv2.dilate(img_final,kernel)


#エッジ検出
contours, hierarchy = cv2.findContours(img_e2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

contours = list(filter(lambda x: cv2.contourArea(x) > 200, contours))

img_draw=cv2.drawContours(img, contours, -1, color=(0, 0, 255), thickness=10)
#エッジを補正

#画面の大きさを調整
cv2.namedWindow("resize1",cv2.WINDOW_NORMAL)
cv2.resizeWindow("resize1",1080,860)
#画像を表示
cv2.imshow("resize1", img)

cv2.namedWindow("resize2",cv2.WINDOW_NORMAL)
cv2.resizeWindow("resize2",1080,860)
cv2.imshow("resize2", img_e2)

cv2.waitKey(0)
cv2.destroyAllWindows()
