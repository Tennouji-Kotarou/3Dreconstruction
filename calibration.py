import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'FZHei-B01'


def calibrate_from_images(image_dir='calibration_images', pattern_size=(8, 5)):
    """
    从图片文件夹中读取图片进行相机标定

    参数:
    image_dir: 存放标定图片的文件夹路径
    pattern_size: 标定板内角点数量 (列数, 行数)
    """
    # 设置寻找亚像素角点的参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 准备对象点
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    # 存储所有图像的对象点和图像点
    objpoints = []  # 3d点在真实世界中的坐标
    imgpoints = []  # 2d点在图像平面的坐标

    # 获取所有图片文件路径
    images = glob.glob(f'{image_dir}/*.jpg') + glob.glob(f'{image_dir}/*.png')

    if not images:
        print(f"在 {image_dir} 目录下没有找到图片文件")
        return None, None

    print(f"找到 {len(images)} 张图片")

    successful_images = 0
    # 处理每张图片
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # 如果找到角点，添加对象点和图像点
        if ret:
            objpoints.append(objp)
            # 角点精细化
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            successful_images += 1

            # 绘制角点
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)

            # 显示结果
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f'已处理图片 {successful_images}/{len(images)}')
            plt.show()

    if successful_images == 0:
        print("没有找到有效的标定图片！")
        return None, None

    print(f"\n成功处理 {successful_images}/{len(images)} 张图片")

    # 执行相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("\n相机矩阵：")
    print(mtx)
    print("\n畸变系数：")
    print(dist)
    print(f"\n平均重投影误差: {mean_error / len(objpoints)}")

    return mtx, dist


# 保存标定结果
def save_calibration(mtx, dist, filename='camera_calibration.npz'):
    """保存标定结果"""
    np.savez(filename, mtx=mtx, dist=dist)
    print(f"标定结果已保存到 {filename}")


# 加载标定结果
def load_calibration(filename='camera_calibration.npz'):
    """加载标定结果"""
    data = np.load(filename)
    return data['mtx'], data['dist']

camera_matrix, dist_coeffs = calibrate_from_images()
if camera_matrix is not None and dist_coeffs is not None:
    save_calibration(camera_matrix, dist_coeffs)