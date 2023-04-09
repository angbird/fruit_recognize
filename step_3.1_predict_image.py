import torch
import cv2
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw


def predict_single_frame(image, n=5):
    """
    输入摄像头画面bgr-array，输出前n个图像分类预测结果的图像bgr-array
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
    img_pil = Image.fromarray(img_rgb)  # array 转 pil
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    predict_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    predict_softmax_temp = f.softmax(predict_logits, dim=1)  # 对 logit 分数做 softmax 运算
    top_n = torch.topk(predict_softmax_temp, n)  # 取置信度最大的 n 个结果
    predict_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析出置信度
    draw = ImageDraw.Draw(img_pil)   # 在图像上写字
    for i in range(len(confs)):    # 在图像上写字
        predict_class = idx_to_labels[predict_ids[i]]
        text = '{:<15} {:>.3f}'.format(predict_class, confs[i])
        draw.text((50, 100 + 100 * i), text, font=font, fill=(255, 0, 0, 1))  # 文字坐标，中文字符串，字体，rgba颜色
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # RGB转BGR
    return img_bgr, predict_softmax_temp


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 有 GPU 就用 GPU，没有就用 CPU
print('device:', device)
idx_to_labels = np.load('chart_and_table/idx_to_labels.npy', allow_pickle=True).item()  # 载入类别
model = torch.load('weights/fruit30_pytorch.pth')  # 载入训练好的模型
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
font = ImageFont.truetype('SimHei.ttf', 100)  # 导入中文字体，指定字号
model = model.eval().to(device)              # 模型调整为测试模式
test_transform = transforms.Compose([transforms.Resize(256),   # 测试集图像预处理：缩放、裁剪、转 Tensor、归一化
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                     ])
img_path = 'files_for_model_test/test_pomegranate.jpg'  # 载入测试图片
img = cv2.imread(img_path)
img, predict_softmax = predict_single_frame(img, n=3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(18, 6))  # 绘图
ax1 = plt.subplot(1, 2, 1)  # 绘制左图-预测图
ax1.imshow(img)
ax1.axis('off')
ax2 = plt.subplot(1, 2, 2)  # 绘制右图-柱状图
x = idx_to_labels.values()
y = predict_softmax.cpu().detach().numpy()[0] * 100
ax2.bar(x, y, alpha=0.5, width=0.3, edgecolor='none', lw=3)
ax = plt.bar(x, y, width=0.45)
plt.bar_label(ax, fmt='%.2f', fontsize=7)  # 置信度数值
plt.title('{} 图像分类预测结果'.format(img_path.split('/')[1]), fontsize=12)
plt.xlabel('类别', fontsize=20)
plt.ylabel('置信度', fontsize=20)
plt.ylim([0, 110])  # y轴取值范围
ax2.tick_params(labelsize=12)  # 坐标文字大小
plt.xticks(rotation=90)  # 横轴文字旋转
plt.tight_layout()
plt.show()
fig.savefig('chart_and_table/预测图+柱状图.jpg')
