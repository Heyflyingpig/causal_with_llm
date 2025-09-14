 # 导入BIF文件读取器
from pgmpy.readwrite import BIFReader

# 读取.bif文件并获取模型
# 这行代码会解析文件的结构和概率表，在内存中创建一个模型对象
model = BIFReader('data/bnlearn/asia/asia.bif').get_model()

# 打印模型的因果边，验证是否加载成功
print("模型真实的因果边:", model.edges())
# 预期输出: [('Pollution', 'Cancer'), ('Smoker', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', '    ')]