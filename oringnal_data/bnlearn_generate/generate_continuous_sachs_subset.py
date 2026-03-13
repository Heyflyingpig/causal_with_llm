import pandas as pd
import numpy as np
from pgmpy.models import LinearGaussianBayesianNetwork
import networkx as nx

# 1. 读取原始的连续数据集
try:
    continuous_data = pd.read_csv("oringnal_data/bnlearn/Sachs/sachs_dataset.csv")
    print("成功读取连续数据集 sachs_dataset.csv")
    continuous_data.columns = [col.lower() for col in continuous_data.columns]
except FileNotFoundError:
    print("错误：未找到 'oringnal_data/bnlearn/Sachs/sachs_dataset.csv' 文件。")
    exit()

# 2. 定义因果图结构
sachs_model_structure = [
    ('pkc', 'pka'), ('pkc', 'raf'), ('pka', 'raf'), ('pkc', 'mek'),
    ('pka', 'mek'), ('raf', 'mek'), ('mek', 'erk'), ('pka', 'erk'),
    ('erk', 'akt'), ('pka', 'akt'), ('pkc', 'p38'), ('pka', 'p38'),
    ('pkc', 'jnk'), ('pka', 'jnk'), ('plc', 'pip3'), ('plc', 'pip2'),
    ('pip3', 'pip2')
]
model = LinearGaussianBayesianNetwork(sachs_model_structure)

# 3. 学习模型参数
print("正在从连续数据中学习模型参数...")
model.fit(continuous_data)
print("模型参数学习完毕。")
assert model.check_model(), "模型构建失败。"

# 4. 为 LinearGaussianBayesianNetwork 手动实现前向采样
def forward_sample_lg(model, size=1):
    """
    从线性高斯贝叶斯网络手动进行前向采样。
    """
    samples = pd.DataFrame(columns=model.nodes())
    topological_order = list(nx.topological_sort(model))
    
    for _ in range(size):
        node_values = {}
        for node in topological_order:
            cpd = model.get_cpds(node)
            parents = model.get_parents(node)
            
            if not parents:
                # 根节点：直接从其高斯分布采样
                mean = cpd.mean[0]
                variance = cpd.variance[0]
                node_values[node] = np.random.normal(mean, np.sqrt(variance))
            else:
                # 非根节点：根据父节点的值计算均值，然后采样
                parent_values = [node_values[p] for p in parents]
                # cpd.beta[0] 是截距, cpd.beta[1:] 是父节点的系数
                mean = cpd.mean[0] + np.dot(cpd.mean[1:], parent_values)
                variance = cpd.variance[0]
                node_values[node] = np.random.normal(mean, np.sqrt(variance))
        
        samples.loc[len(samples)] = node_values
        
    return samples

print("正在从完整模型中生成新的连续数据样本...")
generated_continuous_data = forward_sample_lg(model, size=1000)
print("新数据生成完毕！")

# 5. 从生成的数据中选择您需要的三个变量
columns_to_keep = ['pka', 'p38', 'jnk']
subset_data = generated_continuous_data[columns_to_keep]

print(f"已从生成的数据中选择 {', '.join(columns_to_keep)} 的子集:")
print(subset_data.head())

# 6. 将结果保存到新的CSV文件
output_filename = "oringnal_data/bnlearn_generate/generated_continuous_sachs_subset.csv"
subset_data.to_csv(output_filename, index=False)

print(f"已将只包含三个变量的连续数据集保存到: {output_filename}")
