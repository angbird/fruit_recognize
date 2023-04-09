import numpy as np
import time
import cv2
from PIL import Image, ImageFont, ImageDraw
import torch
import torch.nn.functional as f
from torchvision import transforms


def process_frame(img):  # 处理帧函数
    # 记录该帧开始处理的时间
    start_time = time.time()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
    img_pil = Image.fromarray(img_rgb)  # array 转 PIL
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    predict_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    predict_softmax = f.softmax(predict_logits, dim=1)  # 对 logit 分数做 softmax 运算
    top_n = torch.topk(predict_softmax, 3)  # 取置信度最大的 n 个结果
    predict_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析预测类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析置信度
    draw = ImageDraw.Draw(img_pil)  # 使用PIL绘制中文
    # 在图像上写字
    for i in range(len(confs)):
        predict_class = idx_to_labels[predict_ids[i]]
        text = '{:<15} {:>.3f}'.format(predict_class, confs[i])
        # 文字坐标，中文字符串，字体，b_g_r_alpha颜色
        draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 1))
    img = np.array(img_pil)  # PIL 转 array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB转BGR
    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    fps = 1 / (end_time - start_time)
    # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，线宽，线型
    img = cv2.putText(img, 'FPS ' + str(int(fps)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4, cv2.LINE_AA)
    return img


# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)
font = ImageFont.truetype('SimHei.ttf', 26)  # 导入中文字体，指定字号
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

cap = cv2.VideoCapture(1)  # 获取摄像头，传入0表示获取系统默认摄像头
cap.open(0)  # 打开cap
# 无限循环，直到break被触发
while cap.isOpened():
    # 获取画面
    success, frame = cap.read()
    if not success:
        print('Error')
        break
    frame = process_frame(frame)  # 处理帧函数

    cv2.imshow('my_window', frame)  # 展示处理后的三通道图像
    if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break
cap.release()  # 关闭摄像头
cv2.destroyAllWindows()  # 关闭图像窗口
