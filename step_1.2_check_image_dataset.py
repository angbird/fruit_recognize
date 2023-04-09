import numpy as np
from tqdm import tqdm
import cv2
import os


dataset_path = 'dataset'                # 数据集路径
for fruit in tqdm(os.listdir(dataset_path)):
    for file in os.listdir(os.path.join(dataset_path, fruit)):
        file_path = os.path.join(dataset_path, fruit, file)
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)       # 读取中文路径图片，-1表示读入完整图像，0读入灰度图，1表示读入三通道
        if img is None:
            print(file_path, '读取错误，删除')
            os.remove(file_path)
        else:
            channel = img.shape[2]
            if channel != 3:
                print(file_path, '非三通道，删除')
                os.remove(file_path)
