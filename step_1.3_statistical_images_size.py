import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 指定数据集路径
dataset_path = 'dataset'
os.chdir(dataset_path)              # os.chdir(path)的作用是改变当前工作目录到指定路径
fruit_class = os.listdir()
print('水果种类有{}'.format(fruit_class))

df = pd.DataFrame()
for fruit in tqdm(os.listdir(), desc='开始获取图像信息'):  # 遍历每个类别
    os.chdir(fruit)                   # 改变工作目录
    for file in os.listdir():  # 遍历每张图像
        try:
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
            data = [
                {'类别': fruit, '文件名': file, '图像宽': img.shape[1], '图像高': img.shape[0]}
                   ]
            temp_df = pd.DataFrame(data)
            df = pd.concat([df, temp_df], ignore_index=True)
        except:
            print(os.path.join(fruit, file), '读取错误')
    os.chdir('../')                # 返回上一级目录
os.chdir('../')                    # 返回上一级目录
print(df)


# 可视化图像尺寸分布

x = df['图像宽']
y = df['图像高']
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

plt.figure(figsize=(6, 6))
plt.scatter(x, y, c=z,  s=20, cmap='Spectral_r')
plt.tick_params(labelsize=10)

xy_max = max(max(df['图像宽']), max(df['图像高']))
plt.xlim(xmin=0, xmax=xy_max/2)
plt.ylim(ymin=0, ymax=xy_max/2)
plt.ylabel('height', fontsize=15)
plt.xlabel('width', fontsize=15)
plt.savefig('chart_and_table/图像尺寸分布.pdf', dpi=120, bbox_inches='tight')
plt.show()
