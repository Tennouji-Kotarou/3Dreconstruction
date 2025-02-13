import numpy as np
import cv2
from mplfonts import use_font

use_font('FZHei-B01')
import matplotlib.pyplot as plt
from pathlib import Path


def load_calibration():
    """加载相机标定结果"""
    data = np.load('camera_calibration.npz')
    return data['mtx'], data['dist']


def extract_features(image):
    """提取SIFT特征点和描述符"""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def match_features_knn(desc1, desc2, k=2, ratio_thresh=0.75):
    """使用KNN进行特征匹配"""
    # 创建BF匹配器
    bf = cv2.BFMatcher()

    # 对每个特征点找到k个最佳匹配
    matches = bf.knnMatch(desc1, desc2, k=k)

    # 应用Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches


def process_image_pair(img1_path, img2_path):
    """处理图像对并进行特征匹配和3D重建"""
    # 加载标定参数
    K, dist = load_calibration()

    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise ValueError("无法读取图像，请检查图像路径")

    # 校正图像畸变
    h, w = img1.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    img1_undist = cv2.undistort(img1, K, dist, None, newcameramtx)
    img2_undist = cv2.undistort(img2, K, dist, None, newcameramtx)

    # 转换为灰度图
    gray1 = cv2.cvtColor(img1_undist, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_undist, cv2.COLOR_BGR2GRAY)

    # 提取特征
    kp1, desc1 = extract_features(gray1)
    kp2, desc2 = extract_features(gray2)

    # 特征匹配
    matches = match_features_knn(desc1, desc2)

    # 可视化匹配结果
    match_img = cv2.drawMatches(img1_undist, kp1, img2_undist, kp2, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(15, 5))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f'特征匹配结果 - {len(matches)}个匹配点')
    plt.axis('off')
    plt.show()

    # 提取匹配点坐标
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算本质矩阵
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # 从本质矩阵恢复R和t
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    # 三角测量
    # 将点转换为归一化坐标
    pts1_norm = cv2.undistortPoints(pts1, K, dist)
    pts2_norm = cv2.undistortPoints(pts2, K, dist)

    # 构建投影矩阵
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))

    # 三角测量点
    points_4d = cv2.triangulatePoints(P1, P2, pts1_norm.reshape(2, -1), pts2_norm.reshape(2, -1))
    points_3d = (points_4d[:3] / points_4d[3]).T

    return points_3d, R, t, len(matches)


def visualize_3d_points(points_3d):
    """可视化3D点云"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                         c=points_3d[:, 2], cmap='viridis', marker='.')

    # 添加颜色条
    plt.colorbar(scatter)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置标题
    plt.title('重建的3D点云')
    plt.show()


# 主函数
def main(img1_path, img2_path):
    """主函数"""
    try:
        points_3d, R, t, num_matches = process_image_pair(img1_path, img2_path)

        print(f"\n重建信息:")
        print(f"- 匹配点数量: {num_matches}")
        print(f"- 重建点数量: {len(points_3d)}")
        print("\n相机运动:")
        print("旋转矩阵 R:")
        print(R)
        print("\n平移向量 t:")
        print(t)

        visualize_3d_points(points_3d)

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")






main('image3.jpg', 'image4.jpg')
