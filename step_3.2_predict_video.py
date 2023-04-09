import os
import time
import shutil
import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import gc
import matplotlib
import torch
import torch.nn.functional as f
import mmcv
import matplotlib.pyplot as plt
from torchvision import transforms


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


def predict_single_frame_bar(image):
    """
    输入predict_single_frame函数输出的bgr-array，加柱状图，保存
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
    fig = plt.figure(figsize=(18, 6))
    # 绘制左图-视频图
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.axis('off')
    # 绘制右图-柱状图
    ax2 = plt.subplot(1, 2, 2)
    x = idx_to_labels.values()
    y = predict_softmax.cpu().detach().numpy()[0] * 100
    ax2.bar(x, y, alpha=0.5, width=0.3, edgecolor='none', lw=3)
    plt.xlabel('类别', fontsize=20)
    plt.ylabel('置信度', fontsize=20)
    ax2.tick_params(labelsize=16)  # 坐标文字大小
    plt.ylim([0, 100])  # y轴取值范围
    plt.xlabel('类别', fontsize=25)
    plt.ylabel('置信度', fontsize=25)
    plt.title('图像分类预测结果', fontsize=30)
    plt.xticks(rotation=90)  # 横轴文字旋转
    plt.tight_layout()
    fig.savefig(f'{temp_out_dir}/{frame_id:06d}.jpg')
    # 释放内存
    fig.clf()
    plt.close()
    gc.collect()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 有 GPU 就用 GPU，没有就用 CPU
print('device:', device)
font = ImageFont.truetype('SimHei.ttf', 100)  # 导入中文字体，指定字号
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.use('Agg')  # 后端绘图，不显示，只保存
idx_to_labels = np.load('chart_and_table/idx_to_labels.npy', allow_pickle=True).item()  # 载入类别
model = torch.load('weights/fruit30_pytorch.pth')  # 载入模型
model = model.eval().to(device)
test_transform = transforms.Compose([transforms.Resize(256),  # 测试集图像预处理：缩放裁剪、转 Tensor、归一化
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                     ])
input_video = 'files_for_model_test/fruits_video.mp4'
temp_out_dir = time.strftime('%Y%m%d%H%M%S')  # 创建临时文件夹，存放每帧结果
os.mkdir(temp_out_dir)
print('创建临时文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))
# 读入待预测视频
images = mmcv.VideoReader(input_video)
prog_bar = mmcv.ProgressBar(len(images))
# 对视频逐帧处理
for frame_id, img in enumerate(images):
    # 可视化方案一:原始图像+预测结果文字
    # img, predict_softmax = predict_single_frame(img, n=3)
    # cv2.imwrite(f'{temp_out_dir}/{frame_id:06d}.jpg', img)
    # 可视化方案二:原始图像+预测结果文字+各类别置信度柱状图
    img, predict_softmax = predict_single_frame(img, n=3)
    predict_single_frame_bar(img)
    prog_bar.update()  # 更新进度条
# 把每一帧串成视频文件
mmcv.frames2video(temp_out_dir, 'files_for_model_test/output_bar.mp4', fps=images.fps, fourcc='mp4v')
shutil.rmtree(temp_out_dir)  # 删除存放每帧画面的临时文件夹
print('删除临时文件夹', temp_out_dir)
