"""
CEVAE 信息增益验证实验
Information Gain Validation for LLM-generated Confounder

核心思想:
- 训练两个 CEVAE 模型:
  1. Baseline: CEVAE(X, T, Y) - 不使用 U
  2. With U:   CEVAE(X, T, Y, U) - 将 U 作为额外输入
- 比较两个模型的 ELBO (Evidence Lower Bound)
- 如果加入 U 后 ELBO 显著提升,说明 U 包含了额外的有用信息

ELBO = E_q[log p(T|z,X)] + E_q[log p(Y|z,X,T)] - KL(q(z|...) || p(z))
     = -L_T - L_Y - KL
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
    output_dir = 'outcome/1220_outcome/CEVAE_gain'
    
    # 模型超参数
    x_dim = 7           # 协变量维度
    z_dim = 3           # 隐变量维度
    hidden_dim = 64     # 隐藏层维度
    
    # 训练超参数
    batch_size = 32
    learning_rate = 1e-3
    epochs = 500
    beta = 1.0          # KL 权重 (固定)
    
    # 多次实验
    n_runs = 10         # 运行次数,取平均以减少随机性
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 数据加载 (复用)
# ============================================================================
class JobsDataset:
    """Jobs 数据集加载器"""
    def __init__(self, json_path):
        self.json_path = json_path
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.scaler_U = StandardScaler()
        self._load_data()
    
    def _load_data(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
        
        data_dict = json_content[0]
        self.confounder_name = data_dict.get('confounder_name', 'Unknown')
        df = pd.DataFrame(data_dict.get('data', []))
        
        x_cols = ['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're75']
        
        self.X = self.scaler_X.fit_transform(df[x_cols].values.astype(np.float32))
        self.T = df['treat'].values.astype(np.float32).reshape(-1, 1)
        self.Y = self.scaler_Y.fit_transform(df['re78'].values.astype(np.float32).reshape(-1, 1))
        self.U = self.scaler_U.fit_transform(df[self.confounder_name].values.astype(np.float32).reshape(-1, 1))
        
        print(f"数据集: {len(df)} 样本, 混杂变量: {self.confounder_name}")
    
    def get_tensors(self, device='cpu'):
        return (torch.FloatTensor(self.X).to(device),
                torch.FloatTensor(self.T).to(device),
                torch.FloatTensor(self.Y).to(device),
                torch.FloatTensor(self.U).to(device))
    
    def get_dataloader(self, batch_size=32, shuffle=True, device='cpu'):
        X, T, Y, U = self.get_tensors(device)
        dataset = TensorDataset(X, T, Y, U)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# 通用 CEVAE 模型 (支持可选的 U 输入)
# ============================================================================
class CEVAEWithOptionalU(nn.Module):
    """
    CEVAE 模型,支持可选地将 U 作为编码器的额外输入
    
    - use_u=False: 编码器输入为 (X, T, Y)
    - use_u=True:  编码器输入为 (X, T, Y, U)
    """
    def __init__(self, x_dim, z_dim=3, hidden_dim=64, use_u=False):
        super().__init__()
        self.z_dim = z_dim
        self.use_u = use_u
        
        # 编码器输入维度: X + T + Y (+ U if use_u)
        encoder_input_dim = x_dim + 1 + 1 + (1 if use_u else 0)
        
        # 编码器
        self.encoder_fc = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, z_dim)
        
        # Treatment 解码器: p(T | z, X)
        self.treatment_decoder = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Outcome 解码器: p(Y | z, X, T)
        self.outcome_fc = nn.Sequential(
            nn.Linear(z_dim + x_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU()
        )
        self.outcome_mu = nn.Linear(16, 1)
        self.outcome_logvar = nn.Linear(16, 1)

    def encode(self, x, t, y, u=None):
        """编码器前向传播"""
        if self.use_u and u is not None:
            inputs = torch.cat([x, t, y, u], dim=1)
        else:
            inputs = torch.cat([x, t, y], dim=1)
        h = self.encoder_fc(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode_treatment(self, z, x):
        """Treatment 解码器"""
        inputs = torch.cat([z, x], dim=1)
        return self.treatment_decoder(inputs)

    def decode_outcome(self, z, x, t):
        """Outcome 解码器"""
        inputs = torch.cat([z, x, t], dim=1)
        h = self.outcome_fc(inputs)
        y_mu = self.outcome_mu(h)
        y_logvar = self.outcome_logvar(h)
        return y_mu, y_logvar

    def forward(self, x, t, y, u=None):
        """完整前向传播"""
        mu, logvar = self.encode(x, t, y, u)
        z = self.reparameterize(mu, logvar)
        t_pred = self.decode_treatment(z, x)
        y_mu, y_logvar = self.decode_outcome(z, x, t)
        return t_pred, y_mu, y_logvar, mu, logvar, z


# ============================================================================
# 损失函数和 ELBO 计算
# ============================================================================
def compute_loss_and_elbo(t, y, t_pred, y_mu, y_logvar, mu, logvar, beta=1.0):
    """
    计算损失和 ELBO

    ELBO = E_q[log p(T|z,X)] + E_q[log p(Y|z,X,T)] - KL(q(z) || p(z))

    由于我们使用负对数似然作为损失:
    Loss = -ELBO = L_T + L_Y + beta * KL
    ELBO = -Loss (当 beta=1)

    Returns:
        loss, elbo, loss_t, loss_y, kl_div
    """
    # Treatment 重构损失 (BCE = 负对数似然)
    loss_t = F.binary_cross_entropy(t_pred, t, reduction='mean')

    # Outcome 重构损失 (Gaussian NLL)
    y_var = torch.exp(y_logvar)
    loss_y = 0.5 * torch.mean((y - y_mu)**2 / y_var + y_logvar)

    # KL 散度
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # 总损失
    loss = loss_t + loss_y + beta * kl_div

    # ELBO (当 beta=1 时, ELBO = -loss)
    # 更精确地: ELBO = -loss_t - loss_y - kl_div
    elbo = -loss_t - loss_y - kl_div

    return loss, elbo, loss_t, loss_y, kl_div


def train_model(model, dataloader, config, use_u=False, verbose=False):
    """
    训练模型并返回最终的 ELBO

    Args:
        model: CEVAE 模型
        dataloader: 数据加载器
        config: 配置
        use_u: 是否使用 U 作为输入
        verbose: 是否打印训练过程

    Returns:
        final_elbo: 最后一个 epoch 的平均 ELBO
        history: 训练历史
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {'loss': [], 'elbo': [], 'loss_t': [], 'loss_y': [], 'kl': []}

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_elbo = 0.0
        epoch_loss_t = 0.0
        epoch_loss_y = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for batch in dataloader:
            x, t, y, u = batch

            optimizer.zero_grad()

            # 前向传播
            if use_u:
                t_pred, y_mu, y_logvar, mu, logvar, z = model(x, t, y, u)
            else:
                t_pred, y_mu, y_logvar, mu, logvar, z = model(x, t, y, None)

            # 计算损失和 ELBO
            loss, elbo, loss_t, loss_y, kl_div = compute_loss_and_elbo(
                t, y, t_pred, y_mu, y_logvar, mu, logvar, config.beta
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_elbo += elbo.item()
            epoch_loss_t += loss_t.item()
            epoch_loss_y += loss_y.item()
            epoch_kl += kl_div.item()
            n_batches += 1

        # 记录
        history['loss'].append(epoch_loss / n_batches)
        history['elbo'].append(epoch_elbo / n_batches)
        history['loss_t'].append(epoch_loss_t / n_batches)
        history['loss_y'].append(epoch_loss_y / n_batches)
        history['kl'].append(epoch_kl / n_batches)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1}/{config.epochs}] "
                  f"Loss: {epoch_loss/n_batches:.4f}, "
                  f"ELBO: {epoch_elbo/n_batches:.4f}")

    final_elbo = history['elbo'][-1]
    return final_elbo, history


# ============================================================================
# 评估函数: 在完整数据集上计算 ELBO
# ============================================================================
def evaluate_elbo(model, X, T, Y, U, use_u=False):
    """在完整数据集上评估 ELBO"""
    model.eval()
    with torch.no_grad():
        if use_u:
            t_pred, y_mu, y_logvar, mu, logvar, z = model(X, T, Y, U)
        else:
            t_pred, y_mu, y_logvar, mu, logvar, z = model(X, T, Y, None)

        _, elbo, loss_t, loss_y, kl_div = compute_loss_and_elbo(
            T, Y, t_pred, y_mu, y_logvar, mu, logvar, beta=1.0
        )

    return elbo.item(), loss_t.item(), loss_y.item(), kl_div.item()


# ============================================================================
# 可视化
# ============================================================================
def plot_comparison(results, output_dir):
    """绘制对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. ELBO 对比 (柱状图)
    ax1 = axes[0]
    labels = ['Baseline\n(without U)', 'With U']
    elbos = [results['baseline_elbo_mean'], results['with_u_elbo_mean']]
    stds = [results['baseline_elbo_std'], results['with_u_elbo_std']]
    colors = ['steelblue', 'coral']

    bars = ax1.bar(labels, elbos, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax1.set_ylabel('ELBO')
    ax1.set_title('ELBO Comparison')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 添加数值标签
    for bar, elbo in zip(bars, elbos):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{elbo:.3f}', ha='center', va='bottom', fontsize=10)

    # 2. ELBO 提升量
    ax2 = axes[1]
    gain = results['elbo_gain_mean']
    gain_std = results['elbo_gain_std']
    ax2.bar(['ELBO Gain'], [gain], yerr=[gain_std], capsize=5,
            color='green' if gain > 0 else 'red', alpha=0.8)
    ax2.set_ylabel('ELBO Improvement')
    ax2.set_title(f'Information Gain: {gain:.4f} +/- {gain_std:.4f}')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 3. 各运行的 ELBO
    ax3 = axes[2]
    runs = range(1, len(results['baseline_elbos']) + 1)
    ax3.plot(runs, results['baseline_elbos'], 'o-', label='Baseline', color='steelblue')
    ax3.plot(runs, results['with_u_elbos'], 's-', label='With U', color='coral')
    ax3.set_xlabel('Run')
    ax3.set_ylabel('ELBO')
    ax3.set_title('ELBO across Multiple Runs')
    ax3.legend()
    ax3.set_xticks(runs)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'information_gain_comparison.png'), dpi=150)
    plt.close()
    print(f"对比图已保存到: {output_dir}/information_gain_comparison.png")


def plot_training_curves(baseline_history, with_u_history, output_dir):
    """绘制训练曲线对比"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(baseline_history['elbo']) + 1)

    # ELBO
    axes[0, 0].plot(epochs, baseline_history['elbo'], label='Baseline', color='steelblue')
    axes[0, 0].plot(epochs, with_u_history['elbo'], label='With U', color='coral')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('ELBO')
    axes[0, 0].set_title('ELBO during Training')
    axes[0, 0].legend()

    # Total Loss
    axes[0, 1].plot(epochs, baseline_history['loss'], label='Baseline', color='steelblue')
    axes[0, 1].plot(epochs, with_u_history['loss'], label='With U', color='coral')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Total Loss during Training')
    axes[0, 1].legend()

    # Treatment Loss
    axes[1, 0].plot(epochs, baseline_history['loss_t'], label='Baseline', color='steelblue')
    axes[1, 0].plot(epochs, with_u_history['loss_t'], label='With U', color='coral')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Treatment Loss')
    axes[1, 0].set_title('Treatment Reconstruction Loss')
    axes[1, 0].legend()

    # Outcome Loss
    axes[1, 1].plot(epochs, baseline_history['loss_y'], label='Baseline', color='steelblue')
    axes[1, 1].plot(epochs, with_u_history['loss_y'], label='With U', color='coral')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Outcome Loss')
    axes[1, 1].set_title('Outcome Reconstruction Loss')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_comparison.png'), dpi=150)
    plt.close()
    print(f"训练曲线对比图已保存到: {output_dir}/training_curves_comparison.png")


def save_results(results, output_dir):
    """保存结果到 JSON"""
    output = {
        'baseline_elbo_mean': results['baseline_elbo_mean'],
        'baseline_elbo_std': results['baseline_elbo_std'],
        'with_u_elbo_mean': results['with_u_elbo_mean'],
        'with_u_elbo_std': results['with_u_elbo_std'],
        'elbo_gain_mean': results['elbo_gain_mean'],
        'elbo_gain_std': results['elbo_gain_std'],
        'baseline_elbos': results['baseline_elbos'],
        'with_u_elbos': results['with_u_elbos'],
        'n_runs': len(results['baseline_elbos']),
        'interpretation': results['interpretation']
    }

    path = os.path.join(output_dir, 'information_gain_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"结果已保存到: {path}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """
    信息增益验证实验主函数

    比较:
    - Baseline: CEVAE(X, T, Y) - 仅使用观测数据
    - With U:   CEVAE(X, T, Y, U) - 加入 LLM 生成的混杂变量
    """
    print("\n" + "="*60)
    print("CEVAE 信息增益验证实验")
    print("比较 CEVAE(X,T,Y) vs CEVAE(X,T,Y,U)")
    print("="*60)

    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    # 加载数据
    print("\n--- 加载数据 ---")
    dataset = JobsDataset(config.data_path)
    X, T, Y, U = dataset.get_tensors(config.device)

    # 存储多次运行的结果
    baseline_elbos = []
    with_u_elbos = []
    baseline_histories = []
    with_u_histories = []

    print(f"\n--- 开始实验 ({config.n_runs} 次运行) ---")

    for run in range(config.n_runs):
        seed = 42 + run
        set_seed(seed)

        print(f"\n[Run {run+1}/{config.n_runs}] seed={seed}")

        # 创建数据加载器
        dataloader = dataset.get_dataloader(
            batch_size=config.batch_size, shuffle=True, device=config.device
        )

        # --- 训练 Baseline 模型 (无 U) ---
        print("  训练 Baseline (without U)...")
        model_baseline = CEVAEWithOptionalU(
            x_dim=config.x_dim, z_dim=config.z_dim,
            hidden_dim=config.hidden_dim, use_u=False
        ).to(config.device)

        elbo_baseline, history_baseline = train_model(
            model_baseline, dataloader, config, use_u=False, verbose=False
        )

        # 在完整数据上评估
        elbo_baseline_eval, _, _, _ = evaluate_elbo(model_baseline, X, T, Y, U, use_u=False)
        baseline_elbos.append(elbo_baseline_eval)
        baseline_histories.append(history_baseline)
        print(f"    Baseline ELBO: {elbo_baseline_eval:.4f}")

        # --- 训练 With U 模型 ---
        print("  训练 With U...")
        model_with_u = CEVAEWithOptionalU(
            x_dim=config.x_dim, z_dim=config.z_dim,
            hidden_dim=config.hidden_dim, use_u=True
        ).to(config.device)

        elbo_with_u, history_with_u = train_model(
            model_with_u, dataloader, config, use_u=True, verbose=False
        )

        # 在完整数据上评估
        elbo_with_u_eval, _, _, _ = evaluate_elbo(model_with_u, X, T, Y, U, use_u=True)
        with_u_elbos.append(elbo_with_u_eval)
        with_u_histories.append(history_with_u)
        print(f"    With U ELBO: {elbo_with_u_eval:.4f}")
        print(f"    ELBO Gain: {elbo_with_u_eval - elbo_baseline_eval:.4f}")

    # 汇总结果
    print("\n" + "="*60)
    print("实验结果汇总")
    print("="*60)

    baseline_mean = np.mean(baseline_elbos)
    baseline_std = np.std(baseline_elbos)
    with_u_mean = np.mean(with_u_elbos)
    with_u_std = np.std(with_u_elbos)

    elbo_gains = [w - b for w, b in zip(with_u_elbos, baseline_elbos)]
    gain_mean = np.mean(elbo_gains)
    gain_std = np.std(elbo_gains)

    print(f"\nBaseline ELBO: {baseline_mean:.4f} +/- {baseline_std:.4f}")
    print(f"With U ELBO:   {with_u_mean:.4f} +/- {with_u_std:.4f}")
    print(f"ELBO Gain:     {gain_mean:.4f} +/- {gain_std:.4f}")

    # 结果解读
    print("\n--- 结果解读 ---")
    if gain_mean > 0.1:
        interpretation = "SIGNIFICANT_GAIN"
        print(f"[PASS] ELBO 提升显著 ({gain_mean:.4f} > 0.1)")
        print("       U 确实包含了超出 X 的额外信息,是有价值的混杂变量!")
    elif gain_mean > 0:
        interpretation = "MARGINAL_GAIN"
        print(f"[WEAK] ELBO 有轻微提升 ({gain_mean:.4f})")
        print("       U 可能包含少量额外信息,但效果不明显")
    else:
        interpretation = "NO_GAIN"
        print(f"[FAIL] ELBO 无提升或下降 ({gain_mean:.4f})")
        print("       U 未提供超出 X 的额外信息")

    # 统计显著性检验 (简单 t-test)
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(with_u_elbos, baseline_elbos)
    print(f"\n配对 t 检验: t = {t_stat:.4f}, p = {p_value:.4f}")
    if p_value < 0.05:
        print("  差异在统计上显著 (p < 0.05)")
    else:
        print("  差异在统计上不显著 (p >= 0.05)")

    # 整理结果
    results = {
        'baseline_elbo_mean': baseline_mean,
        'baseline_elbo_std': baseline_std,
        'with_u_elbo_mean': with_u_mean,
        'with_u_elbo_std': with_u_std,
        'elbo_gain_mean': gain_mean,
        'elbo_gain_std': gain_std,
        'baseline_elbos': baseline_elbos,
        'with_u_elbos': with_u_elbos,
        'interpretation': interpretation,
        't_statistic': t_stat,
        'p_value': p_value
    }

    # 保存和可视化
    print("\n--- 保存结果 ---")
    plot_comparison(results, config.output_dir)
    plot_training_curves(baseline_histories[-1], with_u_histories[-1], config.output_dir)
    save_results(results, config.output_dir)

    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)

    return results


if __name__ == '__main__':
    main()