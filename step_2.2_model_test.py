import os
from tqdm import tqdm
from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import warnings


warnings.filterwarnings("ignore")  # 忽略烦人的红色提示
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
dataset_dir = 'fruit30_split'
test_path = os.path.join(dataset_dir, 'val')
print('测试集路径', test_path)
# 测试集图像预处理：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                     ])
# 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)
print('测试集图像数量', len(test_dataset))
print('类别个数', len(test_dataset.classes))
print('各类别名称', test_dataset.classes)
BATCH_SIZE = 32
# 测试集的数据加载器
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=0
                         )
model = torch.load('weights/fruit30_pytorch.pth')
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicts = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicts == labels).sum()

    print('测试集上的准确率为 {:.3f} %'.format(100 * correct / total))
