## 分析llm生成数据的因果效应

import os
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def discover_causal_structure(data):
    """
    对给定的数据集运行PC因果发现算法。

    参数:
        data (pd.DataFrame): 经过标签编码的数值型数据集。

    返回:
        causallearn.Graph.GeneralGraph: 发现的因果图对象。
    """
    # 将DataFrame转换为NumPy数组，这是causal-learn需要的格式
    data_np = data.to_numpy()
    
    # 运行PC算法
    # alpha参数是显著性水平，用于独立性检验
    cg = pc(data_np, alpha=0.05)
    
    return cg

def main():
    """
    主函数，遍历所有生成的数据文件，执行因果发现并打印结果。
    """
    data_dir = 'llm_generated_data2/'
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 '{data_dir}' 不存在。")
        return

    # 获取目录下所有的CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        print(f"\n{'='*20}")
        print(f"正在分析文件: {csv_file}")
        print(f"{'='*20}")

        # --- 1. 加载数据 ---
        df = pd.read_csv(file_path)
        
        # --- 2. 数据预处理：标签编码 ---
        # 我们需要将所有列从类别型（字符串）转换为数值型（整数）
        df_encoded = df.apply(LabelEncoder().fit_transform)
        
        # 获取编码后的列名，用于后续展示
        column_names = df.columns.tolist()

        # --- 3. 运行因果发现算法 ---
        print("正在运行PC算法进行因果发现...")
        # 将DataFrame转换为NumPy数组
        data_np = df_encoded.to_numpy()
    
        # 运行PC算法，直接传入节点名称
        cg = pc(data_np, alpha=0.05, node_names=column_names)
        
        # --- 4. 打印发现的因果图 ---
        print("\nPC算法发现的因果图边:")
        # 正确的API，获取图的边
        edges = cg.G.get_graph_edges()
        
        if not edges:
            print("  -> 算法未发现任何因果边。")
        else:
            # 将数字索引映射回我们原始的变量名
            for edge in edges:
                node1 = edge.get_node1()
                node2 = edge.get_node2()
                
                print(f"  -> {node1.get_name()} {edge.get_endpoint1()}--{edge.get_endpoint2()} {node2.get_name()}")
        


if __name__ == '__main__':
    main()
