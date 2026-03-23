import cv2
import matplotlib.pyplot as plt


# 1. 准备图片
img1 = cv2.imread('./test_image/t1.jpg', 0)
img2 = cv2.imread('./test_image/t2.jpg', 0)

# 如果图片没读到，报错提醒
if img1 is None or img2 is None:
    print("错误：没找到图片，请检查路径！")
    exit()

# 2. 【核心修改】创建一个 ORB 实例，但我们只用它的 BRIEF 部分
# 通过参数设置，让它表现得像 FAST + BRIEF
# scoreType=cv2.ORB_FAST_SCORE 让它主要参考 FAST 的逻辑
orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)

# 3. 检测并计算
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 4. 暴力匹配 (Hamming 距离)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 5. 排序
matches = sorted(matches, key=lambda x: x.distance)

# 6. 画图
img_result = cv2.drawMatches(
    img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 8))
# 转一下颜色，Matplotlib 看得更顺眼
plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
plt.title("FAST + BRIEF (via ORB Engine)")
plt.show()
