import os
from tqdm import tqdm
from torchvision import datasets
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
import numpy as np
import warnings


warnings.filterwarnings("ignore")  # 忽略烦人的红色提示
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
# 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])
# 数据集文件夹路径
dataset_dir = 'fruit30_split'
train_path = os.path.join(dataset_dir, 'train')
print('训练集路径', train_path)
# 载入训练集
train_dataset = datasets.ImageFolder(train_path, train_transform)
print('训练集图像数量', len(train_dataset))
print('类别个数', len(train_dataset.classes))
print('各类别名称', train_dataset.classes)
# 保存映射关系
idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
np.save('chart_and_table/idx_to_labels.npy', idx_to_labels)
np.save('/chart_and_table/labels_to_idx.npy', train_dataset.class_to_idx)
# 训练集的数据加载器
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0
                          )
model = models.resnet18(pretrained=True)  # 载入预训练模型
# 修改全连接层，使得全连接层的输出与当前数据集类别数对应
# 新建的层默认 requires_grad=True
model.fc = nn.Linear(model.fc.in_features, 30)
# 只微调训练最后一层全连接层的参数，其它层冻结
optimizer = optim.Adam(model.fc.parameters())
model = model.to(device)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 训练轮次 Epoch
EPOCHS = 20
# 遍历每个 EPOCH
for epoch in tqdm(range(EPOCHS)):
    model.train()
    for images, labels in train_loader:  # 获得一个 batch 的数据和标注
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
torch.save(model, 'weights/fruit30_pytorch.pth')
