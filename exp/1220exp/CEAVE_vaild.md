# CEVAE 验证 LLM 生成混杂变量的实验方案

## 1. 研究背景与目标

### 1.1 问题描述

在因果推断中，未观测混杂因子（Unobserved Confounders）是影响因果效应估计准确性的核心问题。本项目使用 **框架** 通过 LLM 生成隐藏的混杂变量，但需要一种**量化方法**来验证生成的混杂变量质量。

### 1.2 现有工作

- **框架**（已完成）：通过 LLM 的三阶段流程生成混杂变量
  - P_var: 生成混杂变量名称和含义
  - P_dist: 确定分布类型
  - P_param: 为每个个体生成分布参数
- **PC 算法验证**（已尝试）：只能定性验证结构，无法量化

### 1.3 研究目标

使用 **CEVAE（Causal Effect Variational Autoencoder）** 来量化评估 LLM 生成的混杂变量 U 的质量。

---

## 2. 数据说明

### 2.1 原始数据

- **数据集**: Jobs Dataset (NSW),twins
- **位置**: `oringnal_data/bnlearn/jobs/nsw.dta` `oringnal_data/bnlearn/twins/twins.csv`
- **样本量**: 722 samples;140000 samples


---

## 3. CEVAE 理论框架

### 3.1 核心思想

CEVAE 从观测数据 (X, T, Y) 中推断隐藏的混杂因子 z。如果 LLM 生成的 U 是真正的混杂因子，那么 CEVAE 学到的 z 应该与 U 有一致性。

### 3.2 因果图结构

```
        U (未观测混杂因子)
       ↙         ↘
      ↓           ↓
X → T (处理) ──→ Y (结果)
```

### 3.3 模型结构

```
编码器: q(z | X, T, Y)
  - 输入: 协变量 X, 处理 T, 结果 Y
  - 输出: 隐变量 z 的分布参数 (μ, σ)

解码器1 - Treatment: p(T | z, X)
  - 输入: 隐变量 z, 协变量 X
  - 输出: 处理分配概率

解码器2 - Outcome: p(Y | z, X, T)
  - 输入: 隐变量 z, 协变量 X, 处理 T
  - 输出: 结果的分布参数
```

### 3.4 损失函数

```
L = L_T + L_Y + β × D_KL

- L_T: Treatment 重构损失 (BCE)
- L_Y: Outcome 重构损失 (MSE 或 Gaussian NLL)
- D_KL: KL 散度 (后验与先验的差异)
```

---

## 4. 验证方案

### 4.1 验证流程概览

```
Step 1: 独立一致性验证
  └─ 训练 CEVAE(X,T,Y) → z
  └─ 计算 z 与 U 的相关性

Step 2: 信息增益验证
  └─ 对比 CEVAE(X,T,Y) vs CEVAE(X,T,Y,U) 的 ELBO

Step 3: 因果效应一致性验证
  └─ 比较使用 z 和使用 U 调整后的 ATE 估计
```

### 4.2 量化指标

| 指标 | 计算方法 | 好的 U 应该有的表现 |
|------|---------|-------------------|
| 相关系数 | corr(z, U) | \|r\| > 0.3 |
| 互信息 | MI(z, U) | MI > 0，越高越好 |
| 预测 R² | LinearRegression(z→U).R² | R² > 0.1 |
| ELBO 提升 | ELBO(+U) - ELBO(baseline) | 显著正值 |
| ATE 差异 | \|ATE_z - ATE_U\| | 差异越小越好 |

---

## 5. 实施步骤

### Step 1: 数据准备

```python
# 从 JSON 加载数据
# 提取: X (协变量), T (treat), Y (re78), U (LLM生成的混杂变量)
```

### Step 2: 训练 CEVAE

```python
# 使用 (X, T, Y) 训练 CEVAE
# 不使用 U，让模型自己学习隐变量 z
```

### Step 3: 提取隐变量 z

```python
# 对每个样本，使用编码器提取 z
# z = Encoder(X, T, Y).mean
```

### Step 4: 计算验证指标

```python
# 计算 corr(z, U), MI(z, U), R²
# 计算 ATE_z 和 ATE_U
```

### Step 5: 信息增益验证（可选）

```python
# 训练 CEVAE(X, T, Y, U)
# 比较 ELBO 差异
```

---

## 6. 预期结果解读

| 结果 | 解读 |
|------|------|
| corr(z, U) 显著 > 0 | CEVAE 发现的隐变量与 LLM 的 U 一致，U 可信 |
| corr(z, U) ≈ 0 | U 可能不是主要混杂因子，或 CEVAE 发现了不同因素 |
| ELBO 提升显著 | U 包含超出 X 的额外信息 |
| ATE_z ≈ ATE_U | z 和 U 有相似的"去混杂"效果 |

---

## 7. 注意事项与局限性

### 7.1 理论局限

- **可识别性问题**: VAE 的隐变量不可识别，z 和 U 可能信息相同但数值不同
- **因果假设**: CEVAE 假设特定因果结构，真实结构可能不同

### 7.2 实践注意

- **样本量**: 当前 500 样本可能偏少，建议使用较小网络
- **KL 坍塌**: 训练时注意监控 KL 散度，必要时使用 KL annealing
- **超参数**: 隐变量维度、网络结构需要调参

---

## 8. 文件结构

```

exp/1220exp/                  # 本实验代码
├── CEVAE_Validation_README.md # 本文档

outcome/1220_outcome/         #本次实验结果
```

---

## 9. CEVAE 详细架构

### 9.1 整体数据流

```
输入: (X, T, Y)
      |
      v
+---------------------------------------------+
|           编码器 Encoder                     |
|  [X, T, Y] -> FC -> ReLU -> FC -> (mu, log_var) |
+---------------------------------------------+
      |
      v
重参数化采样: z = mu + sigma * eps,  eps ~ N(0,1)
      |
      +----------------------+------------------------+
      |                      |                        |
      v                      v                        v
+---------------+  +------------------+  +------------------+
| Treatment Dec |  |  Outcome Dec     |  | (可选) X Dec     |
| [z,X] -> pi   |  | [z,X,T] -> mu_y  |  | [z] -> X_hat     |
+---------------+  +------------------+  +------------------+
      |                      |                        |
      v                      v                        v
    L_T = BCE            L_Y = NLL                  L_X
                    |
                    v
            总损失 L = L_T + L_Y + beta * KL
```

### 9.2 网络结构建议（小样本场景）

```
编码器:
  - 输入层: dim(X) + 1 + 1 = 9  (7个协变量 + T + Y)
  - 隐藏层: 64 -> 32
  - 输出层: z_dim * 2  (mu 和 log_var)

Treatment 解码器:
  - 输入层: z_dim + dim(X) = z_dim + 7
  - 隐藏层: 32 -> 16
  - 输出层: 1 (sigmoid -> 概率)

Outcome 解码器:
  - 输入层: z_dim + dim(X) + 1 = z_dim + 8
  - 隐藏层: 32 -> 16
  - 输出层: 2 (mu_y 和 log_var_y)

推荐超参数:
  - z_dim: 2~5 (样本量小，不宜过大)
  - beta: 1.0 (可尝试 0.5~2.0)
  - 学习率: 1e-3
  - Batch size: 32~64
```

---

## 10. 代码实现框架 (PyTorch 伪代码)

### 10.1 CEVAE 模型类

```python
class CEVAE(nn.Module):
    def __init__(self, x_dim, z_dim=3):
        # 编码器: (X, T, Y) -> (mu, sigma)
        self.encoder = nn.Sequential(...)
        self.fc_mu = nn.Linear(32, z_dim)
        self.fc_logvar = nn.Linear(32, z_dim)

        # Treatment 解码器: (z, X) -> P(T=1)
        self.treatment_decoder = nn.Sequential(...)

        # Outcome 解码器: (z, X, T) -> (mu_y, sigma_y)
        self.outcome_decoder = nn.Sequential(...)

    def encode(self, x, t, y):
        # 返回 mu, logvar
        pass

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps
        pass

    def forward(self, x, t, y):
        # 完整前向传播
        pass
```

### 10.2 损失函数

```python
def cevae_loss(t, y, t_pred, y_mu, y_logvar, mu, logvar, beta=1.0):
    # Treatment 重构损失 (BCE)
    loss_t = F.binary_cross_entropy(t_pred, t)

    # Outcome 重构损失 (Gaussian NLL)
    loss_y = 0.5 * ((y - y_mu)**2 / y_var + y_logvar).mean()

    # KL 散度
    kl_div = -0.5 * (1 + logvar - mu**2 - logvar.exp()).mean()

    return loss_t + loss_y + beta * kl_div
```

### 10.3 验证指标计算

```python
def compute_validation_metrics(z, u):
    # 1. Pearson 相关系数
    corr = pearsonr(z, u)

    # 2. 互信息
    mi = mutual_info_regression(z, u)

    # 3. 预测 R^2
    r2 = LinearRegression().fit(z, u).score(z, u)

    return {'corr': corr, 'mi': mi, 'r2': r2}
```

---

## 11. 参考文献

1. ProCI 论文: "Mitigating hidden confounding by progressive confounder imputation via large language models"
2. CEVAE 论文: Louizos et al. "Causal Effect Inference with Deep Latent-Variable Models" (NeurIPS 2017)
3. beta-VAE 论文: Higgins et al. "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (ICLR 2017)

---


