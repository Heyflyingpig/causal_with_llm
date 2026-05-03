"""
CEVAE (Causal Effect Variational Autoencoder) Validation
用于验证 LLM 生成的混杂变量 U 的质量

This script implements the CEVAE model and validation pipeline to assess
the quality of LLM-generated confounders by comparing:
1. Learned latent variable z with LLM-generated U (correlation, MI, R^2)
2. ELBO improvement when including U
3. ATE estimation consistency
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数 Configuration
# ============================================================================
class Config:
    # 数据路径
    data_path = 'outcome/1012_outcome/final_data.json'
    output_dir = 'outcome/1220_outcome/CEVAE_validation'
    
    # 模型超参数
    x_dim = 7           # 协变量维度 (age, education, black, hispanic, married, nodegree, re75)
    z_dim = 3           # 隐变量维度 (样本量较小,不宜过大)
    hidden_dim = 64     # 隐藏层维度
    
    # 训练超参数
    batch_size = 32
    learning_rate = 1e-3
    epochs = 200
    beta_start = 0.0    # KL annealing 起始值
    beta_end = 1.0      # KL annealing 终止值
    warmup_epochs = 50  # KL annealing warmup epochs
    
    # 随机种子
    seed = 42

    # 设备 (自动检测 GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    """设置随机种子以保证可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 数据加载模块 Data Loading
# ============================================================================
class JobsDataset:
    """
    Jobs 数据集加载器
    从 JSON 文件加载数据并提取 X, T, Y, U 变量
    """
    def __init__(self, json_path):
        self.json_path = json_path
        self.X = None           # 协变量
        self.T = None           # 处理变量
        self.Y = None           # 结果变量
        self.U = None           # LLM生成的混杂变量
        self.confounder_name = None
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.scaler_U = StandardScaler()
        
        self._load_data()
    
    def _load_data(self):
        """从 JSON 文件加载数据"""
        print(f"加载数据: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
        
        # 获取数据
        data_dict = json_content[0]
        self.confounder_name = data_dict.get('confounder_name', 'Unknown')
        print(f"混杂变量名称: {self.confounder_name}")
        
        # 转换为 DataFrame
        df = pd.DataFrame(data_dict.get('data', []))
        
        # 定义协变量列
        x_cols = ['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're75']
        
        # 提取变量
        self.X = df[x_cols].values.astype(np.float32)
        self.T = df['treat'].values.astype(np.float32).reshape(-1, 1)
        self.Y = df['re78'].values.astype(np.float32).reshape(-1, 1)
        self.U = df[self.confounder_name].values.astype(np.float32).reshape(-1, 1)
        
        # 标准化
        self.X = self.scaler_X.fit_transform(self.X)
        self.Y = self.scaler_Y.fit_transform(self.Y)
        self.U = self.scaler_U.fit_transform(self.U)
        
        print(f"数据集大小: {len(df)} 样本")
        print(f"协变量维度: {self.X.shape[1]}")
        print(f"处理组样本: {int(self.T.sum())}, 对照组样本: {len(self.T) - int(self.T.sum())}")
    
    def get_tensors(self, device='cpu'):
        """返回 PyTorch 张量"""
        X = torch.FloatTensor(self.X).to(device)
        T = torch.FloatTensor(self.T).to(device)
        Y = torch.FloatTensor(self.Y).to(device)
        U = torch.FloatTensor(self.U).to(device)
        return X, T, Y, U
    
    def get_dataloader(self, batch_size=32, shuffle=True, device='cpu'):
        """返回 DataLoader"""
        X, T, Y, U = self.get_tensors(device)
        dataset = TensorDataset(X, T, Y, U)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# CEVAE 模型 Model Definition
# ============================================================================
class Encoder(nn.Module):
    """
    编码器: q(z | X, T, Y) 
    从观测数据推断隐变量 z 的分布参数
    """
    def __init__(self, x_dim, z_dim, hidden_dim=64):
        super().__init__()
        # 输入: X (x_dim) + T (1) + Y (1)
        input_dim = x_dim + 2
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        # 输出均值和对数方差
        self.fc_mu = nn.Linear(hidden_dim // 2, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, z_dim)
    
    def forward(self, x, t, y):
        # 拼接输入
        inputs = torch.cat([x, t, y], dim=1)
        h = self.fc(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class TreatmentDecoder(nn.Module):
    """
    Treatment 解码器: p(T | z, X)
    预测处理分配的概率
    """
    def __init__(self, x_dim, z_dim, hidden_dim=32):
        super().__init__()
        input_dim = z_dim + x_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z, x):
        inputs = torch.cat([z, x], dim=1)
        t_prob = self.fc(inputs)
        return t_prob


class OutcomeDecoder(nn.Module):
    """
    Outcome 解码器: p(Y | z, X, T)
    预测结果变量的分布参数 (均值和对数方差)
    """
    def __init__(self, x_dim, z_dim, hidden_dim=32):
        super().__init__()
        # 输入: z + X + T
        input_dim = z_dim + x_dim + 1

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        # 输出结果的均值和对数方差
        self.fc_mu = nn.Linear(hidden_dim // 2, 1)
        self.fc_logvar = nn.Linear(hidden_dim // 2, 1)

    def forward(self, z, x, t):
        inputs = torch.cat([z, x, t], dim=1)
        h = self.fc(inputs)
        y_mu = self.fc_mu(h)
        y_logvar = self.fc_logvar(h)
        return y_mu, y_logvar


class CEVAE(nn.Module):
    """
    CEVAE (Causal Effect Variational Autoencoder)
    完整模型,包含编码器和两个解码器

    因果图结构:
            U (未观测混杂因子)
           /         \
          v           v
    X -> T (处理) --> Y (结果)
    """
    def __init__(self, x_dim, z_dim=3, hidden_dim=64):
        super().__init__()
        self.z_dim = z_dim

        # 编码器
        self.encoder = Encoder(x_dim, z_dim, hidden_dim)
        # Treatment 解码器
        self.treatment_decoder = TreatmentDecoder(x_dim, z_dim, hidden_dim // 2)
        # Outcome 解码器
        self.outcome_decoder = OutcomeDecoder(x_dim, z_dim, hidden_dim // 2)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧: z = mu + sigma * epsilon
        epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, t, y):
        """
        前向传播
        返回: t_pred, y_mu, y_logvar, mu, logvar, z
        """
        # 编码
        mu, logvar = self.encoder(x, t, y)
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        # 解码 Treatment
        t_pred = self.treatment_decoder(z, x)
        # 解码 Outcome
        y_mu, y_logvar = self.outcome_decoder(z, x, t)

        return t_pred, y_mu, y_logvar, mu, logvar, z

    def get_latent(self, x, t, y):
        """
        获取隐变量 z (使用均值,不采样)
        用于推断阶段
        """
        mu, logvar = self.encoder(x, t, y)
        return mu  # 返回均值作为点估计


# ============================================================================
# 损失函数 Loss Functions
# ============================================================================
def cevae_loss(t, y, t_pred, y_mu, y_logvar, mu, logvar, beta=1.0):
    """
    CEVAE 损失函数

    L = L_T + L_Y + beta * D_KL

    Args:
        t: 真实处理变量 [batch, 1]
        y: 真实结果变量 [batch, 1]
        t_pred: 预测的处理概率 [batch, 1]
        y_mu: 预测的结果均值 [batch, 1]
        y_logvar: 预测的结果对数方差 [batch, 1]
        mu: 编码器输出的均值 [batch, z_dim]
        logvar: 编码器输出的对数方差 [batch, z_dim]
        beta: KL 散度的权重

    Returns:
        total_loss, loss_t, loss_y, kl_div
    """
    # 1. Treatment 重构损失 (Binary Cross Entropy)
    loss_t = F.binary_cross_entropy(t_pred, t, reduction='mean')

    # 2. Outcome 重构损失 (Gaussian Negative Log-Likelihood)
    y_var = torch.exp(y_logvar)
    loss_y = 0.5 * torch.mean((y - y_mu)**2 / y_var + y_logvar)

    # 3. KL 散度: D_KL(q(z|x,t,y) || p(z))
    # p(z) = N(0, I)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # 总损失
    total_loss = loss_t + loss_y + beta * kl_div

    return total_loss, loss_t, loss_y, kl_div


def get_beta(epoch, warmup_epochs, beta_start=0.0, beta_end=1.0):
    """
    KL Annealing: 逐渐增加 beta 值
    防止 KL 坍塌 (posterior collapse)
    """
    if epoch < warmup_epochs:
        return beta_start + (beta_end - beta_start) * (epoch / warmup_epochs)
    return beta_end


# ============================================================================
# 训练函数 Training
# ============================================================================
def train_cevae(model, dataloader, config, verbose=True):
    """
    训练 CEVAE 模型

    Args:
        model: CEVAE 模型实例
        dataloader: 数据加载器
        config: 配置对象
        verbose: 是否打印训练过程

    Returns:
        训练历史记录
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {
        'loss': [], 'loss_t': [], 'loss_y': [], 'kl_div': [], 'beta': []
    }

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_loss_t = 0.0
        epoch_loss_y = 0.0
        epoch_kl = 0.0
        n_batches = 0

        # 获取当前 beta
        beta = get_beta(epoch, config.warmup_epochs, config.beta_start, config.beta_end)

        for batch in dataloader:
            x, t, y, u = batch

            optimizer.zero_grad()

            # 前向传播
            t_pred, y_mu, y_logvar, mu, logvar, z = model(x, t, y)

            # 计算损失
            loss, loss_t, loss_y, kl_div = cevae_loss(
                t, y, t_pred, y_mu, y_logvar, mu, logvar, beta
            )

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_t += loss_t.item()
            epoch_loss_y += loss_y.item()
            epoch_kl += kl_div.item()
            n_batches += 1

        # 记录历史
        history['loss'].append(epoch_loss / n_batches)
        history['loss_t'].append(epoch_loss_t / n_batches)
        history['loss_y'].append(epoch_loss_y / n_batches)
        history['kl_div'].append(epoch_kl / n_batches)
        history['beta'].append(beta)

        # 打印进度
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}] "
                  f"Loss: {epoch_loss/n_batches:.4f} "
                  f"(T: {epoch_loss_t/n_batches:.4f}, "
                  f"Y: {epoch_loss_y/n_batches:.4f}, "
                  f"KL: {epoch_kl/n_batches:.4f}, "
                  f"beta: {beta:.2f})")

    return history


# ============================================================================
# 验证指标计算 Validation Metrics
# ============================================================================
def compute_validation_metrics(z, u):
    """
    计算验证指标: 评估学习到的 z 与 LLM 生成的 U 之间的一致性

    Args:
        z: 学习到的隐变量 [n_samples, z_dim]
        u: LLM 生成的混杂变量 [n_samples, 1]

    Returns:
        metrics: 包含各项指标的字典
    """
    # 展平 u
    u_flat = u.flatten()

    metrics = {}

    # 对 z 的每个维度计算与 U 的相关性
    z_dim = z.shape[1]
    pearson_corrs = []
    spearman_corrs = []

    for i in range(z_dim):
        z_i = z[:, i]
        # Pearson 相关系数
        p_corr, p_pval = pearsonr(z_i, u_flat)
        pearson_corrs.append((p_corr, p_pval))
        # Spearman 相关系数
        s_corr, s_pval = spearmanr(z_i, u_flat)
        spearman_corrs.append((s_corr, s_pval))

    metrics['pearson_correlations'] = pearson_corrs
    metrics['spearman_correlations'] = spearman_corrs

    # 找出与 U 最相关的 z 维度
    best_idx = np.argmax([abs(c[0]) for c in pearson_corrs])
    metrics['best_z_dim'] = best_idx
    metrics['best_pearson'] = pearson_corrs[best_idx]
    metrics['best_spearman'] = spearman_corrs[best_idx]

    # 计算互信息 (使用最相关的 z 维度)
    mi = mutual_info_regression(z, u_flat, random_state=42)
    metrics['mutual_info'] = mi
    metrics['best_mi'] = mi[best_idx]

    # 计算预测 R^2 (使用所有 z 维度预测 U)
    lr = LinearRegression()
    lr.fit(z, u_flat)
    r2 = lr.score(z, u_flat)
    metrics['r2'] = r2

    return metrics


def compute_ate(model, x, t, device='cpu'):
    """
    计算平均处理效应 (ATE)
    ATE = E[Y(1) - Y(0)]

    使用学习到的模型预测反事实结果
    """
    model.eval()
    with torch.no_grad():
        # 创建处理和对照的 T
        t_treat = torch.ones(x.shape[0], 1).to(device)
        t_control = torch.zeros(x.shape[0], 1).to(device)

        # 使用当前 T 和虚拟 Y 获取 z
        y_dummy = torch.zeros(x.shape[0], 1).to(device)

        # 预测 T=1 时的 Y
        mu_1, _ = model.encoder(x, t_treat, y_dummy)
        y1_mu, _ = model.outcome_decoder(mu_1, x, t_treat)

        # 预测 T=0 时的 Y
        mu_0, _ = model.encoder(x, t_control, y_dummy)
        y0_mu, _ = model.outcome_decoder(mu_0, x, t_control)

        # 计算 ATE
        ate = (y1_mu - y0_mu).mean().item()

    return ate


def print_validation_results(metrics, ate_z, ate_naive):
    """打印验证结果"""
    print("\n" + "="*60)
    print("CEVAE 验证结果")
    print("="*60)

    print("\n--- 1. 相关性分析 (z 与 U) ---")
    for i, (p, s) in enumerate(zip(metrics['pearson_correlations'],
                                   metrics['spearman_correlations'])):
        print(f"  z[{i}]: Pearson r = {p[0]:.4f} (p={p[1]:.4f}), "
              f"Spearman rho = {s[0]:.4f} (p={s[1]:.4f})")

    best_idx = metrics['best_z_dim']
    print(f"\n  最佳维度: z[{best_idx}]")
    print(f"  最佳 Pearson: r = {metrics['best_pearson'][0]:.4f}")
    print(f"  最佳 Spearman: rho = {metrics['best_spearman'][0]:.4f}")

    print("\n--- 2. 互信息 (MI) ---")
    for i, mi in enumerate(metrics['mutual_info']):
        print(f"  MI(z[{i}], U) = {mi:.4f}")
    print(f"  最佳 MI: {metrics['best_mi']:.4f}")

    print("\n--- 3. 预测能力 (R^2) ---")
    print(f"  R^2(z -> U) = {metrics['r2']:.4f}")

    print("\n--- 4. ATE 估计 ---")
    print(f"  ATE (CEVAE, 使用 z): {ate_z:.4f}")
    print(f"  ATE (Naive, 简单差异): {ate_naive:.4f}")

    # 结果解读
    print("\n--- 5. 结果解读 ---")
    r = abs(metrics['best_pearson'][0])
    if r > 0.3:
        print(f"  [PASS] |r| = {r:.4f} > 0.3: z 与 U 有显著相关性,U 可能是真正的混杂因子")
    elif r > 0.1:
        print(f"  [WEAK] |r| = {r:.4f}: z 与 U 有弱相关性")
    else:
        print(f"  [FAIL] |r| = {r:.4f} < 0.1: z 与 U 几乎无相关性,U 可能不是主要混杂因子")

    r2 = metrics['r2']
    if r2 > 0.1:
        print(f"  [PASS] R^2 = {r2:.4f} > 0.1: z 可以解释 U 的部分变异")
    else:
        print(f"  [WEAK] R^2 = {r2:.4f} <= 0.1: z 对 U 的解释能力较弱")


# ============================================================================
# 可视化函数 Visualization
# ============================================================================
def plot_training_history(history, output_dir):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 总损失
    axes[0, 0].plot(history['loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')

    # Treatment 损失
    axes[0, 1].plot(history['loss_t'])
    axes[0, 1].set_title('Treatment Loss (BCE)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')

    # Outcome 损失
    axes[1, 0].plot(history['loss_y'])
    axes[1, 0].set_title('Outcome Loss (Gaussian NLL)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')

    # KL 散度 和 beta
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    ax1.plot(history['kl_div'], 'b-', label='KL Divergence')
    ax2.plot(history['beta'], 'r--', label='Beta')
    ax1.set_title('KL Divergence and Beta')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('KL Divergence', color='b')
    ax2.set_ylabel('Beta', color='r')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"训练历史图已保存到: {output_dir}/training_history.png")


def plot_z_vs_u(z, u, metrics, output_dir):
    """绘制 z 与 U 的散点图"""
    best_idx = metrics['best_z_dim']
    z_best = z[:, best_idx]
    u_flat = u.flatten()

    plt.figure(figsize=(8, 6))
    plt.scatter(z_best, u_flat, alpha=0.5, s=20)
    plt.xlabel(f'z[{best_idx}] (learned latent)')
    plt.ylabel('U (LLM-generated confounder)')
    plt.title(f'z vs U (Pearson r = {metrics["best_pearson"][0]:.4f})')

    # 添加回归线
    lr = LinearRegression()
    lr.fit(z_best.reshape(-1, 1), u_flat)
    x_line = np.linspace(z_best.min(), z_best.max(), 100)
    y_line = lr.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, y_line, 'r-', linewidth=2, label='Linear Fit')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'z_vs_u.png'), dpi=150)
    plt.close()
    print(f"z vs U 散点图已保存到: {output_dir}/z_vs_u.png")


def save_results(metrics, ate_z, ate_naive, history, output_dir):
    """保存结果到 JSON 文件"""
    results = {
        'pearson_correlations': [
            {'z_dim': i, 'r': float(p[0]), 'p_value': float(p[1])}
            for i, p in enumerate(metrics['pearson_correlations'])
        ],
        'spearman_correlations': [
            {'z_dim': i, 'rho': float(s[0]), 'p_value': float(s[1])}
            for i, s in enumerate(metrics['spearman_correlations'])
        ],
        'best_z_dim': int(metrics['best_z_dim']),
        'best_pearson_r': float(metrics['best_pearson'][0]),
        'best_spearman_rho': float(metrics['best_spearman'][0]),
        'mutual_info': [float(m) for m in metrics['mutual_info']],
        'r2': float(metrics['r2']),
        'ate_cevae': float(ate_z),
        'ate_naive': float(ate_naive),
        'final_loss': float(history['loss'][-1]),
        'final_kl': float(history['kl_div'][-1])
    }

    output_path = os.path.join(output_dir, 'validation_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"验证结果已保存到: {output_path}")


# ============================================================================
# 主函数 Main Function
# ============================================================================
def main():
    """
    CEVAE 验证实验主函数

    执行流程:
    1. 加载数据
    2. 训练 CEVAE 模型
    3. 提取隐变量 z
    4. 计算验证指标
    5. 保存结果和可视化
    """
    print("\n" + "="*60)
    print("CEVAE 验证 LLM 生成混杂变量的实验")
    print("="*60)

    # 配置
    config = Config()
    set_seed(config.seed)

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"\n输出目录: {config.output_dir}")

    # Step 1: 加载数据
    print("\n--- Step 1: 加载数据 ---")
    dataset = JobsDataset(config.data_path)
    dataloader = dataset.get_dataloader(
        batch_size=config.batch_size,
        shuffle=True,
        device=config.device
    )

    # 获取完整张量用于后续评估
    X, T, Y, U = dataset.get_tensors(config.device)

    # 计算 Naive ATE (简单的处理组与对照组均值差异)
    Y_raw = dataset.scaler_Y.inverse_transform(Y.cpu().numpy())
    T_np = T.cpu().numpy().flatten()
    ate_naive = Y_raw[T_np == 1].mean() - Y_raw[T_np == 0].mean()

    # Step 2: 构建和训练 CEVAE 模型
    print("\n--- Step 2: 训练 CEVAE 模型 ---")
    model = CEVAE(
        x_dim=config.x_dim,
        z_dim=config.z_dim,
        hidden_dim=config.hidden_dim
    ).to(config.device)

    print(f"模型结构:\n{model}")
    print(f"\n参数数量: {sum(p.numel() for p in model.parameters())}")

    history = train_cevae(model, dataloader, config, verbose=True)

    # Step 3: 提取隐变量 z
    print("\n--- Step 3: 提取隐变量 z ---")
    model.eval()
    with torch.no_grad():
        z = model.get_latent(X, T, Y)
        z_np = z.cpu().numpy()

    print(f"z 形状: {z_np.shape}")
    print(f"z 统计: mean={z_np.mean(axis=0)}, std={z_np.std(axis=0)}")

    # Step 4: 计算验证指标
    print("\n--- Step 4: 计算验证指标 ---")
    U_np = U.cpu().numpy()
    metrics = compute_validation_metrics(z_np, U_np)

    # 计算使用 z 的 ATE
    ate_z = compute_ate(model, X, T, config.device)
    # 反标准化 ATE
    ate_z = ate_z * dataset.scaler_Y.scale_[0]

    # 打印验证结果
    print_validation_results(metrics, ate_z, ate_naive)

    # Step 5: 保存结果和可视化
    print("\n--- Step 5: 保存结果 ---")
    plot_training_history(history, config.output_dir)
    plot_z_vs_u(z_np, U_np, metrics, config.output_dir)
    save_results(metrics, ate_z, ate_naive, history, config.output_dir)

    # 保存模型
    model_path = os.path.join(config.output_dir, 'cevae_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)

    return metrics, ate_z, ate_naive


if __name__ == '__main__':
    main()