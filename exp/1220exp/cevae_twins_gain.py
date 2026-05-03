"""
CEVAE 信息增益验证实验 - Twins 数据集版本
Information Gain Validation for LLM-generated Confounder (Twins Dataset)

数据集特点:
- 样本量: 20000 条
- Treatment T: 是否为较重的双胞胎 (0/1)
- Outcome Y: 一年内死亡率 (0/1, 二元变量!)
- 协变量 X: 15 个母亲/妊娠特征
- 混杂变量 U: Placental Function Efficiency (LLM 生成)

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
# 配置参数 Configuration - Twins 数据集专用
# ============================================================================
class Config:
    # 数据路径
    data_path = 'outcome/1220_outcome/twins_final_data.json'
    output_dir = 'outcome/1220_outcome/CEVAE_twins_gain'
    
    # Twins 数据集的协变量列 (15 个)
    x_cols = [
        'mager8',      # 母亲年龄组
        'mrace',       # 母亲种族
        'meduc6',      # 母亲教育水平
        'dmar',        # 婚姻状况
        'gestat10',    # 妊娠周数
        'csex',        # 婴儿性别
        'nprevistq',   # 产前检查次数
        'anemia',      # 贫血
        'cardiac',     # 心脏病
        'diabetes',    # 糖尿病
        'chyper',      # 慢性高血压
        'phyper',      # 妊娠高血压
        'eclamp',      # 子痫
        'tobacco',     # 吸烟
        'alcohol',     # 饮酒
    ]
    
    # 模型超参数 - 针对大数据集调整
    x_dim = 15          # 协变量维度 (Twins 有 15 个)
    z_dim = 5           # 隐变量维度 (增加以捕捉更多信息)
    hidden_dim = 128    # 隐藏层维度 (增加模型容量)
    
    # 训练超参数 - 针对大数据集调整
    batch_size = 256    # 增大批量大小
    learning_rate = 1e-3
    epochs = 100        # 数据量大，可以减少 epochs
    beta = 1.0          # KL 权重
    
    # 多次实验
    n_runs = 10         # 运行次数
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Outcome 类型
    outcome_type = 'binary'  # 'binary' for mort_1yr, 'continuous' for re78


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 数据加载 - Twins 数据集
# ============================================================================
class TwinsDataset:
    """Twins 数据集加载器"""
    def __init__(self, json_path, x_cols):
        self.json_path = json_path
        self.x_cols = x_cols
        self.scaler_X = StandardScaler()
        self.scaler_U = StandardScaler()
        # Y 是二元变量，不需要标准化
        self._load_data()
    
    def _load_data(self):
        print(f"加载数据: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
        
        data_dict = json_content[0]
        self.confounder_name = data_dict.get('confounder_name', 'Unknown')
        self.sample_size = data_dict.get('sample_size', 0)
        
        df = pd.DataFrame(data_dict.get('data', []))
        
        # 提取变量
        self.X = self.scaler_X.fit_transform(df[self.x_cols].values.astype(np.float32))
        self.T = df['treat'].values.astype(np.float32).reshape(-1, 1)
        self.Y = df['mort_1yr'].values.astype(np.float32).reshape(-1, 1)  # 二元变量
        self.U = self.scaler_U.fit_transform(
            df[self.confounder_name].values.astype(np.float32).reshape(-1, 1)
        )
        
        # 统计信息
        n_treated = int(self.T.sum())
        n_control = len(self.T) - n_treated
        mort_treated = self.Y[self.T.flatten() == 1].mean()
        mort_control = self.Y[self.T.flatten() == 0].mean()
        
        print(f"数据集: {len(df)} 样本")
        print(f"混杂变量: {self.confounder_name}")
        print(f"协变量维度: {self.X.shape[1]}")
        print(f"处理组 (Heavier Twin): {n_treated}, 死亡率: {mort_treated:.4f}")
        print(f"对照组 (Lighter Twin): {n_control}, 死亡率: {mort_control:.4f}")
    
    def get_tensors(self, device='cpu'):
        return (torch.FloatTensor(self.X).to(device),
                torch.FloatTensor(self.T).to(device),
                torch.FloatTensor(self.Y).to(device),
                torch.FloatTensor(self.U).to(device))
    
    def get_dataloader(self, batch_size=256, shuffle=True, device='cpu'):
        X, T, Y, U = self.get_tensors(device)
        dataset = TensorDataset(X, T, Y, U)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# CEVAE 模型 - 支持二元 Outcome
# ============================================================================
class CEVAEBinary(nn.Module):
    """
    CEVAE 模型，支持二元 Outcome (适用于 Twins 数据集的死亡率)

    与连续 Outcome 版本的区别:
    - Outcome 解码器输出概率而非 (mu, logvar)
    - 使用 BCE 损失而非 Gaussian NLL
    """
    def __init__(self, x_dim, z_dim=5, hidden_dim=128, use_u=False):
        super().__init__()
        self.z_dim = z_dim
        self.use_u = use_u

        # 编码器输入: X + T + Y (+ U if use_u)
        encoder_input_dim = x_dim + 1 + 1 + (1 if use_u else 0)

        # 编码器 q(z | X, T, Y, [U])
        self.encoder_fc = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, z_dim)

        # Treatment 解码器 p(T | z, X)
        self.treatment_decoder = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Outcome 解码器 p(Y | z, X, T) - 二元输出
        self.outcome_decoder = nn.Sequential(
            nn.Linear(z_dim + x_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def encode(self, x, t, y, u=None):
        if self.use_u and u is not None:
            inputs = torch.cat([x, t, y, u], dim=1)
        else:
            inputs = torch.cat([x, t, y], dim=1)
        h = self.encoder_fc(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode_treatment(self, z, x):
        return self.treatment_decoder(torch.cat([z, x], dim=1))

    def decode_outcome(self, z, x, t):
        return self.outcome_decoder(torch.cat([z, x, t], dim=1))

    def forward(self, x, t, y, u=None):
        mu, logvar = self.encode(x, t, y, u)
        z = self.reparameterize(mu, logvar)
        t_pred = self.decode_treatment(z, x)
        y_pred = self.decode_outcome(z, x, t)
        return t_pred, y_pred, mu, logvar, z


# ============================================================================
# 损失函数 - 二元 Outcome 版本
# ============================================================================
def compute_loss_and_elbo_binary(t, y, t_pred, y_pred, mu, logvar, beta=1.0):
    """
    计算损失和 ELBO (二元 Outcome 版本)

    与连续版本的区别: Y 使用 BCE 而非 Gaussian NLL
    """
    # Treatment 损失 (BCE)
    loss_t = F.binary_cross_entropy(t_pred, t, reduction='mean')

    # Outcome 损失 (BCE，因为 Y 是二元的)
    loss_y = F.binary_cross_entropy(y_pred, y, reduction='mean')

    # KL 散度
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # 总损失
    loss = loss_t + loss_y + beta * kl_div

    # ELBO
    elbo = -loss_t - loss_y - kl_div

    return loss, elbo, loss_t, loss_y, kl_div


def train_model(model, dataloader, config, use_u=False, verbose=False):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {'loss': [], 'elbo': [], 'loss_t': [], 'loss_y': [], 'kl': []}

    model.train()
    for epoch in range(config.epochs):
        epoch_loss, epoch_elbo = 0.0, 0.0
        epoch_loss_t, epoch_loss_y, epoch_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in dataloader:
            x, t, y, u = batch
            optimizer.zero_grad()

            if use_u:
                t_pred, y_pred, mu, logvar, z = model(x, t, y, u)
            else:
                t_pred, y_pred, mu, logvar, z = model(x, t, y, None)

            loss, elbo, loss_t, loss_y, kl_div = compute_loss_and_elbo_binary(
                t, y, t_pred, y_pred, mu, logvar, config.beta
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_elbo += elbo.item()
            epoch_loss_t += loss_t.item()
            epoch_loss_y += loss_y.item()
            epoch_kl += kl_div.item()
            n_batches += 1

        history['loss'].append(epoch_loss / n_batches)
        history['elbo'].append(epoch_elbo / n_batches)
        history['loss_t'].append(epoch_loss_t / n_batches)
        history['loss_y'].append(epoch_loss_y / n_batches)
        history['kl'].append(epoch_kl / n_batches)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{config.epochs}] "
                  f"Loss: {epoch_loss/n_batches:.4f}, "
                  f"ELBO: {epoch_elbo/n_batches:.4f}")

    return history['elbo'][-1], history


def evaluate_elbo(model, X, T, Y, U, use_u=False):
    """在完整数据上评估 ELBO"""
    model.eval()
    with torch.no_grad():
        if use_u:
            t_pred, y_pred, mu, logvar, z = model(X, T, Y, U)
        else:
            t_pred, y_pred, mu, logvar, z = model(X, T, Y, None)

        _, elbo, loss_t, loss_y, kl_div = compute_loss_and_elbo_binary(
            T, Y, t_pred, y_pred, mu, logvar, beta=1.0
        )

    return elbo.item(), loss_t.item(), loss_y.item(), kl_div.item()


# ============================================================================
# 可视化
# ============================================================================
def plot_comparison(results, output_dir):
    """绘制 ELBO 对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. ELBO 对比柱状图
    ax1 = axes[0]
    labels = ['Baseline\n(without U)', 'With U']
    elbos = [results['baseline_elbo_mean'], results['with_u_elbo_mean']]
    stds = [results['baseline_elbo_std'], results['with_u_elbo_std']]
    colors = ['steelblue', 'coral']

    bars = ax1.bar(labels, elbos, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax1.set_ylabel('ELBO')
    ax1.set_title('ELBO Comparison (Twins Dataset)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    for bar, elbo in zip(bars, elbos):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{elbo:.4f}', ha='center', va='bottom', fontsize=10)

    # 2. ELBO 提升量
    ax2 = axes[1]
    gain = results['elbo_gain_mean']
    gain_std = results['elbo_gain_std']
    color = 'green' if gain > 0 else 'red'
    ax2.bar(['ELBO Gain'], [gain], yerr=[gain_std], capsize=5, color=color, alpha=0.8)
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
    plt.savefig(os.path.join(output_dir, 'twins_information_gain.png'), dpi=150)
    plt.close()
    print(f"对比图已保存到: {output_dir}/twins_information_gain.png")


def save_results(results, output_dir):
    """保存结果到 JSON"""
    output = {
        'dataset': 'twins',
        'sample_size': results.get('sample_size', 20000),
        'baseline_elbo_mean': results['baseline_elbo_mean'],
        'baseline_elbo_std': results['baseline_elbo_std'],
        'with_u_elbo_mean': results['with_u_elbo_mean'],
        'with_u_elbo_std': results['with_u_elbo_std'],
        'elbo_gain_mean': results['elbo_gain_mean'],
        'elbo_gain_std': results['elbo_gain_std'],
        'baseline_elbos': results['baseline_elbos'],
        'with_u_elbos': results['with_u_elbos'],
        'n_runs': len(results['baseline_elbos']),
        'interpretation': results['interpretation'],
        't_statistic': results.get('t_statistic'),
        'p_value': results.get('p_value')
    }

    path = os.path.join(output_dir, 'twins_information_gain_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"结果已保存到: {path}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """
    Twins 数据集信息增益验证实验

    比较:
    - Baseline: CEVAE(X, T, Y) - 仅使用观测数据
    - With U:   CEVAE(X, T, Y, U) - 加入 LLM 生成的 Placental Function Efficiency
    """
    print("\n" + "="*70)
    print("CEVAE 信息增益验证实验 - Twins 数据集")
    print("比较 CEVAE(X,T,Y) vs CEVAE(X,T,Y,U)")
    print("="*70)

    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"\n配置:")
    print(f"  - 设备: {config.device}")
    print(f"  - 协变量维度: {config.x_dim}")
    print(f"  - 隐变量维度: {config.z_dim}")
    print(f"  - 隐藏层维度: {config.hidden_dim}")
    print(f"  - 批量大小: {config.batch_size}")
    print(f"  - 训练轮次: {config.epochs}")
    print(f"  - 实验次数: {config.n_runs}")

    # 加载数据
    print("\n--- 加载数据 ---")
    dataset = TwinsDataset(config.data_path, config.x_cols)
    X, T, Y, U = dataset.get_tensors(config.device)

    # 存储结果
    baseline_elbos = []
    with_u_elbos = []

    print(f"\n--- 开始实验 ({config.n_runs} 次运行) ---")

    for run in range(config.n_runs):
        seed = 42 + run
        set_seed(seed)

        print(f"\n[Run {run+1}/{config.n_runs}] seed={seed}")

        dataloader = dataset.get_dataloader(
            batch_size=config.batch_size, shuffle=True, device=config.device
        )

        # --- Baseline (无 U) ---
        print("  训练 Baseline (without U)...")
        model_baseline = CEVAEBinary(
            x_dim=config.x_dim, z_dim=config.z_dim,
            hidden_dim=config.hidden_dim, use_u=False
        ).to(config.device)

        _, _ = train_model(model_baseline, dataloader, config, use_u=False, verbose=False)
        elbo_baseline, _, _, _ = evaluate_elbo(model_baseline, X, T, Y, U, use_u=False)
        baseline_elbos.append(elbo_baseline)
        print(f"    Baseline ELBO: {elbo_baseline:.4f}")

        # --- With U ---
        print("  训练 With U...")
        model_with_u = CEVAEBinary(
            x_dim=config.x_dim, z_dim=config.z_dim,
            hidden_dim=config.hidden_dim, use_u=True
        ).to(config.device)

        _, _ = train_model(model_with_u, dataloader, config, use_u=True, verbose=False)
        elbo_with_u, _, _, _ = evaluate_elbo(model_with_u, X, T, Y, U, use_u=True)
        with_u_elbos.append(elbo_with_u)
        print(f"    With U ELBO: {elbo_with_u:.4f}")
        print(f"    ELBO Gain: {elbo_with_u - elbo_baseline:.4f}")

    # 汇总结果
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)

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

    # 对于二元 Outcome，ELBO 的量级不同，调整阈值
    if gain_mean > 0.01:
        interpretation = "SIGNIFICANT_GAIN"
        print(f"[PASS] ELBO 提升显著 ({gain_mean:.4f} > 0.01)")
        print("       U (Placental Function Efficiency) 包含了额外信息!")
    elif gain_mean > 0:
        interpretation = "MARGINAL_GAIN"
        print(f"[WEAK] ELBO 有轻微提升 ({gain_mean:.4f})")
        print("       U 可能包含少量额外信息")
    else:
        interpretation = "NO_GAIN"
        print(f"[FAIL] ELBO 无提升或下降 ({gain_mean:.4f})")
        print("       U 未提供额外信息")

    # 统计检验
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(with_u_elbos, baseline_elbos)
    print(f"\n配对 t 检验: t = {t_stat:.4f}, p = {p_value:.4f}")
    if p_value < 0.05:
        print("  差异在统计上显著 (p < 0.05)")
    else:
        print("  差异在统计上不显著 (p >= 0.05)")

    # 整理结果
    results = {
        'sample_size': dataset.sample_size,
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

    # 保存结果
    print("\n--- 保存结果 ---")
    plot_comparison(results, config.output_dir)
    save_results(results, config.output_dir)

    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)

    return results


if __name__ == '__main__':
    main()