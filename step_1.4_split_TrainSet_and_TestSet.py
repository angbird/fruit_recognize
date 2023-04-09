import os
import shutil
import random
import pandas as pd

if os.path.exists('dataset_split'):                    # 判断划分后的数据集是否存在，如果存在就删除，不存在就跳过
    shutil.rmtree('dataset_split')
else:
    pass
# 指定数据集路径
original_dataset_path = 'dataset'                       # 原始数据集路径
os.mkdir('dataset_split')                               # 创建划分后的数据集文件夹
split_dataset_path = 'dataset_split'                    # 划分后的数据集路径
classes = os.listdir(original_dataset_path)             # 获得水果类别
print('水果类别有{}'.format(classes))
os.mkdir(os.path.join(split_dataset_path, 'train'))     # 创建 train 文件夹
os.mkdir(os.path.join(split_dataset_path, 'val'))       # 创建 test 文件夹
# 在 train 和 test 文件夹中创建各类别子文件夹
for fruit in classes:
    os.mkdir(os.path.join(split_dataset_path, 'train', fruit))
    os.mkdir(os.path.join(split_dataset_path, 'val', fruit))


test_frac = 0.2   # 测试集比例
random.seed(123)  # 随机数种子，便于复现

df = pd.DataFrame()

print('{:^18} {:^18} {:^6}'.format('类别', '训练集数据个数', '测试集数据个数'))

for fruit in classes:  # 遍历每个类别
    # 读取该类别的所有图像文件名
    old_dir = os.path.join(original_dataset_path, fruit)
    images_filename = os.listdir(old_dir)
    random.shuffle(images_filename)  # 随机打乱
    # 划分训练集和测试集
    test_set_numer = int(len(images_filename) * test_frac)  # 测试集图像个数
    test_set_images = images_filename[:test_set_numer]  # 获取准备移动至 test 目录的测试集图像文件名
    train_set_images = images_filename[test_set_numer:]  # 获取准备移动至 train 目录的训练集图像文件名
    # 移动图像至 test 目录
    for image in test_set_images:
        old_img_path = os.path.join(original_dataset_path, fruit, image)  # 获取原始文件路径
        new_test_path = os.path.join(split_dataset_path, 'val', fruit, image)  # 获取 test 目录的新文件路径
        shutil.copy(old_img_path, new_test_path)  # 复制文件
    # 移动图像至 train 目录
    for image in train_set_images:
        old_img_path = os.path.join(original_dataset_path, fruit, image)  # 获取原始文件路径
        new_train_path = os.path.join(split_dataset_path, 'train', fruit, image)  # 获取 train 目录的新文件路径
        shutil.copy(old_img_path, new_train_path)  # 复制文件

    # 工整地输出每一类别的数据个数
    print('{:^18} {:^18} {:^18}'.format(fruit, len(train_set_images), len(test_set_images)))

    # 保存到表格中
    data = [
        {'class': fruit, 'train_set': len(train_set_images), 'test_set': len(test_set_images)}
           ]
    temp_df = pd.DataFrame(data)
    df = pd.concat([df, temp_df], ignore_index=True)

# 数据集各类别数量统计表格，导出为 csv 文件
df['total'] = df['train_set'] + df['test_set']
df.to_csv('chart_and_table/数据量统计.csv', index=False)
