import cv2

# 1. 读取图像（灰度图）
# img = cv2.imread('./test_image/object.jpg', 0)
img = cv2.imread('./test_image/left.jpg', 0)

# 2. 初始化 ORB 检测器
# nfeatures 是你想找多少个特征点
orb = cv2.ORB_create(nfeatures=500)

# 3. 检测特征点并计算描述子
# keypoints (kp): 特征点的位置、大小、方向
# descriptors (des): 刚才说的 0 和 1 的二进制矩阵
kp, des = orb.detectAndCompute(img, None)

# 4. 看看描述子长啥样
print(f"特征点数量: {len(kp)}")
print(f"第一个点的描述子形状: {des[0].shape}")  # 通常是 32字节 (256位)
print(f"第一个描述子的内容: \n{des[0]}")

# 5. 可视化：把点画出来
img_with_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))
cv2.imshow("ORB Features", img_with_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()
