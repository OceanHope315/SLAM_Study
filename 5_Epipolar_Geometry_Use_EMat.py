# output:
# 已根据 P70 硬件优化 K 矩阵，估计像素焦距: 1083.33
# 相机旋转矩阵 R:
#  [[ 0.98802034  0.04123455 -0.14871289]
#  [-0.04233442  0.99909451 -0.00423672]
#  [ 0.14840354  0.01048164  0.98887134]]
# 观察旋转矩阵的对角线 [0.988, 0.999, 0.988]。因为这些值都非常接近 1，说明你拍摄两张照片时的姿态非常稳，没有剧烈的晃动。
#
# 相机平移向量 t (方向单位化):
# [[ 0.97363871] X 轴 (0.97)：主导方向。说明你拿着 P70 主要是向右侧进行了平移
#  [-0.14036244] Y 轴 (-0.14)：负值。说明在移动过程中，相机有轻微的向上提（因为相机坐标系 Y 轴向下）
#  [ 0.17979448]] Z 轴 (0.18)：正值。说明相机稍微向后退了一点点，或者是镜头中心稍微远离了物体

# 更新：在原有基础上新增了三角测量部分，能够从匹配的 2D 点对中恢复出稀疏的 3D 点云。通过定义两个相机的投影矩阵，并使用 OpenCV 的 triangulatePoints 函数，我们成功地重建了场景中的 3D 点坐标。最后还提供了一个简单的 3D 散点图来可视化这些重建的点云。

import cv2
import numpy as np
from matplotlib import pyplot as plt


def drawlines(img1, img2, lines, pts1, pts2):
    ''' 在 img1 上画出 img2 对应的对极线 '''
    r, c = img1.shape[:2]
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(
        img1.shape) == 2 else img1.copy()

    for r_line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # 对极线方程 ax + by + c = 0
        x0, y0 = map(int, [0, -r_line[2]/r_line[1]])
        x1, y1 = map(int, [c, -(r_line[2] + r_line[0]*c)/r_line[1]])

        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(
            pt1.astype(int)), 5, color, -1)
    return img1_color


# 1. 读取图像
img1 = cv2.imread('./test_image/t1.jpg', 0)
img2 = cv2.imread('./test_image/t2.jpg', 0)
h, w = img1.shape

# --- 针对华为 P70 优化的 K 矩阵 ---
# P70 主摄等效焦距约 24.5mm，全画幅参考 36mm
f_pixel = (24.5 * w) / 36
K = np.array([[f_pixel, 0,       w / 2],
              [0,       f_pixel, h / 2],
              [0,       0,       1]], dtype=np.float32)
print(f"已根据 P70 硬件优化 K 矩阵，估计像素焦距: {f_pixel:.2f}")

# 2. 特征提取 (ORB)
orb = cv2.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 3. 特征匹配 (BFMatcher)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 提取匹配点坐标
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# 4. 【关键】通过内参 K 计算本质矩阵 E，并用 RANSAC 过滤误匹配
# threshold=1.0 表示对极约束误差在 1 像素以内
E, mask = cv2.findEssentialMat(
    pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# 筛选 Inliers
pts1_in = pts1[mask.ravel() == 1]
pts2_in = pts2[mask.ravel() == 1]

# 5. 【姿态恢复】看看相机是怎么挪动的
_, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)
print("\n相机旋转矩阵 R:\n", R)
print("\n相机平移向量 t (方向单位化):\n", t)

# 6. 计算对极线用于可视化
# 计算 F 矩阵：F = inv(K).T * E * inv(K)
Ki = np.linalg.inv(K)
F = Ki.T @ E @ Ki

# 计算并绘制
lines1 = cv2.computeCorrespondEpilines(
    pts2_in.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
img_left = drawlines(img1, img2, lines1, pts1_in, pts2_in)

lines2 = cv2.computeCorrespondEpilines(
    pts1_in.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
img_right = drawlines(img2, img1, lines2, pts2_in, pts1_in)

# 7. 结果展示
plt.figure(figsize=(16, 8))
plt.subplot(121), plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
plt.title(f'Left Image (Inliers: {len(pts1_in)})')
plt.axis('off')
plt.subplot(122), plt.imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
plt.title('Right Image (Epipolar Lines)')
plt.axis('off')
plt.tight_layout()

# --- 8. 【新增】三角测量：从 2D 点恢复 3D 空间坐标 ---

# 定义投影矩阵 P = K [R | t]
# 第一个相机视角（参考系原点）：R = I, t = 0
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
# 第二个相机视角（相对于第一个）：使用 recoverPose 算出的 R, t
P2 = K @ np.hstack((R, t))

# 转换点格式为 2xN (float64) 以适配 triangulatePoints
pts1_tri = pts1_in.T
pts2_tri = pts2_in.T

# 调用 OpenCV 三角测量函数
# 返回的是齐次坐标 (4, N)
points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_tri, pts2_tri)

# 将齐次坐标转换为非齐次 3D 坐标 [X, Y, Z]
points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
points_3d = points_3d.T  # 转置为 (N, 3)

# 打印结果：由于是单目，这里的数值单位是相对的（尺度不确定性）
print("\n--- 三角测量结果 ---")
print(f"成功重建 3D 点数量: {points_3d.shape[0]}")
print("前 5 个点的 3D 坐标 (X, Y, Z):\n", points_3d[:5])

# 可视化 3D 点云（简单的 3D 散点图）
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1],
           points_3d[:, 2], c='r', marker='o', s=5)
ax.set_title("Reconstructed 3D Points (Sparse Cloud)")
ax.set_xlabel('X axis'), ax.set_ylabel('Y axis'), ax.set_zlabel('Z axis')
plt.show()
