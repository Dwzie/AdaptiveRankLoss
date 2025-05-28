---
date: 2025-05-24
aliases:
  - 自适应损失
tags:
  - 二分类问题
  - 损失函数
  - AUC优化
---
# 自适应排序损失函数（AdaptiveRankLoss）

## 1. 基本定义

设当前批次的样本为 $(x_i, y_i)_{i=1}^{n}$，其中 $x_i \in \mathbb{R}^d$ 为特征向量，$y_i \in {0,1}$ 为标签（1表示target，0表示decoy）。

神经网络的预测分数为 $s_i = \sigma(f_\theta(x_i))$，其中 $\sigma$ 为sigmoid函数，$f_\theta$ 为神经网络，$\theta$ 为模型参数。因此，$s_i \in [0,1]$。

历史信息为上一轮epoch的预测分数和标签：${(s_i^{\text{prev}}, y_i^{\text{prev}})}_{i=1}^{m}$。

---

## 2. 损失函数公式

自适应排序损失函数（AdaptiveRankLoss）定义为：

$$  
\mathcal{L}_{\text{ARL}} = \mathcal{L}_1 + \mathcal{L}_2  
$$

其中：

- **损失项1（Current Target vs Previous Decoy）：**

$$  
\mathcal{L}_1 = \frac{1}{N_{pd}} \sum_{i \in \mathcal{T}_{\text{curr}}} \sum_{j \in \mathcal{D}_{\text{prev}}} \alpha_i \cdot \left[ \max(0, s_j^{\text{prev}} - s_i + \xi) \right]^2  
$$

- **损失项2（Current Decoy vs Previous Target）：**

$$  
\mathcal{L}_2 = \frac{1}{N_{pt}} \sum_{i \in \mathcal{D}_{\text{curr}}} \sum_{j \in \mathcal{T}_{\text{prev}}} \left[ \max(0, s_i - s_j^{\text{prev}} + \xi) \right]^2  
$$

其中：

- $\mathcal{T}_{\text{curr}}$ 和 $\mathcal{D}_{\text{curr}}$ 分别为当前批次的target和decoy样本集合。
- $\mathcal{T}_{\text{prev}}$ 和 $\mathcal{D}_{\text{prev}}$ 分别为上一轮epoch的target和decoy样本集合。
- $\xi$ 为可学习的margin参数，$\xi \geq 0$。
- $N_{pd}$ 和 $N_{pt}$ 为归一化常数，通常取为对应的样本对数量。
- $\alpha_i$ 为自适应缩放因子。

---

## 3. 自适应缩放因子

自适应缩放因子 $\alpha_i$ 定义为：

$$  
\alpha_i = \exp\left( -\max\left(0, s_i - \max_{j \in \mathcal{D}_{\text{prev}}} s_j^{\text{prev}}\right) \right)  
$$

---

## 4. 可学习margin参数

margin参数 $\xi$ 通过梯度下降法更新：

$$  
\xi \leftarrow \max\left(0, \xi - \eta_{\xi} \frac{\partial \mathcal{L}_{\text{ARL}}}{\partial \xi}\right)  
$$

其中 $\eta_{\xi}$ 为margin的学习率。

---

## 5. 损失函数的梯度

### 5.1 关于预测分数 $s_i$ 的梯度

对于 $i \in \mathcal{T}_{\text{curr}}$：

$$  
\frac{\partial \mathcal{L}_1}{\partial s_i} = -\frac{2}{N_{pd}} \sum_{j \in \mathcal{D}_{\text{prev}}} \alpha_i \cdot \mathbb{I}[s_j^{\text{prev}} - s_i + \xi > 0] \cdot (s_j^{\text{prev}} - s_i + \xi)  
$$

对于 $i \in \mathcal{D}_{\text{curr}}$：

$$  
\frac{\partial \mathcal{L}_2}{\partial s_i} = \frac{2}{N_{pt}} \sum_{j \in \mathcal{T}_{\text{prev}}} \mathbb{I}[s_i - s_j^{\text{prev}} + \xi > 0] \cdot (s_i - s_j^{\text{prev}} + \xi)  
$$

### 5.2 关于margin $\xi$ 的梯度

$$  
\frac{\partial \mathcal{L}_{\text{ARL}}}{\partial \xi} = \frac{\partial \mathcal{L}_1}{\partial \xi} + \frac{\partial \mathcal{L}_2}{\partial \xi}  
$$

其中：

$$  
\frac{\partial \mathcal{L}_1}{\partial \xi} = \frac{2}{N_{pd}} \sum_{i \in \mathcal{T}_{\text{curr}}} \sum_{j \in \mathcal{D}_{\text{prev}}} \alpha_i \cdot \mathbb{I}[s_j^{\text{prev}} - s_i + \xi > 0] \cdot (s_j^{\text{prev}} - s_i + \xi)  
$$

$$  
\frac{\partial \mathcal{L}_2}{\partial \xi} = \frac{2}{N_{pt}} \sum_{i \in \mathcal{D}_{\text{curr}}} \sum_{j \in \mathcal{T}_{\text{prev}}} \mathbb{I}[s_i - s_j^{\text{prev}} + \xi > 0] \cdot (s_i - s_j^{\text{prev}} + \xi)  
$$