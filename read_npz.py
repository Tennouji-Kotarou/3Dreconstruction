import numpy as np
from mplfonts import use_font
use_font('Noto Serif CJK SC')
import matplotlib.pyplot as plt

# 首先查看 npz 文件中的内容
def inspect_npz(file_path='camera_calibration.npz'):
    """检查 npz 文件中的内容"""
    data = np.load(file_path)
    print("NPZ 文件中包含的键：")
    for key in data.files:
        print(f"- {key}")
        array = data[key]
        print(f"  形状: {array.shape}")
        print(f"  类型: {array.dtype}")
        print()
    return data

# 运行检查
try:
    data = inspect_npz()
except Exception as e:
    print(f"加载文件时出错: {str(e)}")
