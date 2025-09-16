## 计算基数据因果效应

import os
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.preprocessing import LabelEncoder

def discover_causal_structure(data, node_names):
    """
    对给定的数据集运行PC因果发现算法。

    参数:
        data (np.ndarray): 经过标签编码的数值型数据集 (NumPy数组)。
        node_names (list): 变量名称列表。

    返回:
        causallearn.Graph.GeneralGraph: 发现的因果图对象。
    """
    # 运行PC算法
    # alpha参数是显著性水平，用于独立性检验
    cg = pc(data, alpha=0.05, node_names=node_names)
    return cg

def main():
    """
    主函数，加载基准数据文件，执行因果发现并打印结果。
    """
    # 我们要分析的基准数据文件
    benchmark_file_path = 'data_generate/generated_cancer_dataset.csv'
    
    if not os.path.exists(benchmark_file_path):
        print(f"错误: 基准数据文件 '{benchmark_file_path}' 不存在。")
        return

    print(f"\n{'='*30}")
    print(f"正在分析基准数据文件: {os.path.basename(benchmark_file_path)}")
    print(f"{'='*30}")

    # --- 1. 加载数据 ---
    df = pd.read_csv(benchmark_file_path)
    
    # --- 2. 数据预处理：标签编码 ---
    df_encoded = df.apply(LabelEncoder().fit_transform)
    column_names = df.columns.tolist()

    # --- 3. 运行因果发现算法 ---
    print("正在运行PC算法进行因果发现...")
    data_np = df_encoded.to_numpy()
    causal_graph = discover_causal_structure(data_np, column_names)
    
    # --- 4. 打印发现的因果图 ---
    print("\nPC算法发现的因果图边:")
    edges = causal_graph.G.get_graph_edges()
    
    if not edges:
        print("  -> 算法未发现任何因果边。")
    else:
        for edge in edges:
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            print(f"  -> {node1.get_name()} {edge.get_endpoint1()}--{edge.get_endpoint2()} {node2.get_name()}")
            


if __name__ == '__main__':
    main()
