import torch
import seaborn as sns
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 假设 attention_matrix 是你的注意力矩阵
attention_matrix = torch.randn(16, 8, 25, 25)

# 取第一个 batch 的注意力矩阵
first_batch_attention = attention_matrix[0]

# 遍历八个头，绘制热力图
for head_index in range(8):
    # 取出一个头的注意力图
    attention_head = first_batch_attention[head_index].numpy()

    # 使用 Seaborn 绘制热力图
    sns.heatmap(attention_head, cmap='YlGnBu', annot=True, fmt=".2f", cbar=False)
    plt.title(f"Attention Head {head_index + 1}")
    plt.show()
