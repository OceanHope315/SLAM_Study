import cv2
import numpy as np
from matplotlib import pyplot as plt


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1: 我们要在上面画线的图像
        lines: 对应的对极线方程 '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # 直线方程为 ax + by + c = 0
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        # 在图1画线，在图2画对应的特征点
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 2)
        img1 = cv2.circle(img1, tuple(pt1), 6, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 6, color, -1)
    return img1, img2


# 1. 读取图像（建议找两张同一物体但视角略有偏移的照片）
img1 = cv2.imread('./test_image/t1.jpg', 0)
img2 = cv2.imread('./test_image/t2.jpg', 0)

# 2. 使用 ORB 寻找特征点并匹配（大二学生常用的快速算法）
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 使用 BFMatcher 进行暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 提取匹配好的坐标点
# 只保留前50个最优匹配
good_matches = matches[:50]
pts1 = np.int32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.int32([kp2[m.trainIdx].pt for m in good_matches])

# 3. 【核心步骤】计算基础矩阵 F
# 使用 FM_LMEDS 方法（类似于 RANSAC），它会自动剔除不符合对极约束的误匹配点
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# 我们只保留那些被判定为“正确”的点
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# 4. 【核心步骤】计算对极线
# 在右图中找到与左图点对应的极线 (L2)
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img5, img6 = drawlines(img2, img1, lines2, pts2, pts1)

# 在左图中找到与右图点对应的极线 (L1)
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img3, img4 = drawlines(img1, img2, lines1, pts1, pts2)

# 5. 展示结果
plt.subplot(121), plt.imshow(img3)
plt.title('Left Image (with Epipolar Lines from Right)')
plt.subplot(122), plt.imshow(img5)
plt.title('Right Image (with Epipolar Lines from Left)')
plt.show()
