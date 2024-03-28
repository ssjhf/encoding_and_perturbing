import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义要绘制的函数
def f(fi, p, q):
    return fi * (1 - p - q) * (p - q) / (fi * (1 - p - q) * (p - q) + q * (1 - q))

# 常量
fi = 0.1
p = np.linspace(0.01, 0.99, 100)
q = np.linspace(0.01, 0.99, 100)

# 创建p和q的网格
P, Q = np.meshgrid(p, q)

# 计算函数值
Z = f(fi, P, Q)

# 绘图
fig = plt.figure(figsize=(10, 8))  # 尝试调整图形尺寸
ax = fig.add_subplot(111, projection='3d')

# 表面图
surf = ax.plot_surface(P, Q, Z, cmap='viridis')

# 标签和标题
ax.set_xlabel('p*', fontsize=14)
ax.set_ylabel('q*', fontsize=14)
ax.set_zlabel('Ratio', labelpad=35, fontsize=14)  # 调整z轴标签位置

# 调整刻度标签的字体大小
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='z', labelsize=14)

# 调整视角
ax.view_init(elev=20, azim=-35)  # 调整视角

# 显示颜色条
cbar = fig.colorbar(surf, shrink=0.5, aspect=10)  # 添加颜色条
cbar.ax.tick_params(labelsize=14)
plt.savefig("the proportion of the second term.pdf")
plt.show()
