import cv2

img = cv2.imread('./test_image/t1.jpg', 0)
# img = cv2.imread('./test_image/t2.jpg', 0)

# 1. 创建 FAST 检测器
# threshold: 阈值，越小点越多
# nonmaxSuppression: 是否开启非极大值抑制（通常选 True）
fast = cv2.FastFeatureDetector_create(threshold=75, nonmaxSuppression=True)

# 2. 检测点
kp = fast.detect(img, None)

# 3. 画出来
img_fast = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

cv2.imshow('FAST Features', img_fast)
cv2.waitKey(0)
