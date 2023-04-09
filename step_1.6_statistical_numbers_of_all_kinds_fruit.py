import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

df = pd.read_csv('chart_and_table/数据量统计.csv')


# 指定可视化的特征
feature = 'total'
# feature = 'train_set'
# feature = 'test_set'

df = df.sort_values(by=feature, ascending=False)

print(df.head())


plt.figure(figsize=(12, 7))

x = df['class']
y = df[feature]

plt.bar(x, y, facecolor='#1f77b4', edgecolor='k')

plt.xticks(rotation=90)
plt.tick_params(labelsize=15)
plt.xlabel('类别', fontsize=20)
plt.ylabel('图像数量', fontsize=20)
plt.savefig('chart_and_table/各类别图片数量.pdf', dpi=120, bbox_inches='tight')

plt.show()


plt.figure(figsize=(12, 7))
x = df['class']
y1 = df['test_set']
y2 = df['train_set']

width = 0.55  # 柱状图宽度

plt.xticks(rotation=90)  # 横轴文字旋转

plt.bar(x, y1, width, label='测试集')
plt.bar(x, y2, width, label='训练集', bottom=y1)


plt.xlabel('类别', fontsize=20)
plt.ylabel('图像数量', fontsize=20)
plt.tick_params(labelsize=13)   # 设置坐标文字大小

plt.legend(fontsize=16)  # 图例

# 保存为高清的 pdf 文件
plt.savefig('chart_and_table/各类别图像数量.pdf', dpi=120, bbox_inches='tight')

plt.show()
