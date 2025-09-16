from pgmpy.readwrite import BIFReader

from pgmpy.sampling import BayesianModelSampling

import pandas as pd

#读取 .bif 文件（bif文件放在当前目录）
reader = BIFReader("cancer.bif")

#从 BIFReader 中获取 Bayesian Network 模型
model = reader.get_model()

# 检测模型有效性
assert model.check_model(), "❌ 模型有错误！"

# 提示模型已经读入成功
print("BIF文件已成功加载！")

# 显示当前网络中的所有节点名称
print("节点：", list(model.nodes()))

#创建一个采样器对象，用于从这个贝叶斯网络里生成数据样本
sampler = BayesianModelSampling(model)

# 打印提示，开始采样
print("正在生成样本...")

# 可设置随机种子 seed，用于复现结果
dataset = sampler.forward_sample(size=1000)
# dataset = sampler.forward_sample(size=1000, seed=42)

print("生成完毕！")
print(dataset.head())

# 将采样数据保存到 CSV 文件
dataset.to_csv("generated_cancer_dataset.csv", index=False)

print("已保存")