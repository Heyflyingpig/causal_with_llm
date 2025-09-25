## 分析由LLM直接生成的JSON数据中的因果效应

import os
import json
import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.preprocessing import LabelEncoder

def discover_causal_structure(data, node_names):
    """
    对给定的数据集运行PC因果发现算法。
    """
    data_np = data.to_numpy()
    cg = pc(data_np, alpha=0.05, node_names=node_names)
    return cg

def main():
    """
    主函数，加载LLM直接生成的JSON数据文件，执行因果发现并打印结果。
    """
    # 定义新的输入文件路径
    json_file_path = 'llm_exp/925_outcome/data_glm_data_test.json'
    
    if not os.path.exists(json_file_path):
        print(f"错误: 数据文件 '{json_file_path}' 不存在。")
        return

    print(f"\n{'='*30}")
    print(f"正在分析文件: {os.path.basename(json_file_path)}")
    print(f"{'='*30}")

    # --- 1. 加载并解析JSON数据 ---
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
        
        if isinstance(json_content, list) and len(json_content) > 0:
            for data in json_content:
                data_list = data.get('data', [])
                df = pd.DataFrame(data_list)
    
                # --- 3. 数据清洗：移除不相关的'id'列 ---
                if 'id' in df.columns:
                    df = df.drop('id', axis=1)
                    print("已移除'id'列，因为它不参与因果分析。")

                # --- 4. 数据预处理：标签编码 ---
                # 我们需要将所有列从类别型（字符串、布尔值）转换为数值型（整数）
                df_encoded = df.apply(LabelEncoder().fit_transform)
                

                corr_matrix = df_encoded.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] == 1)]

                if to_drop:
                    print(f"警告: 检测到以下列存在完全相关性: {to_drop}")
                    print("将从分析中移除这些列以避免奇异矩阵错误。")
                    df_encoded = df_encoded.drop(columns=to_drop)

                column_names = df_encoded.columns.tolist()

                # 运行因果发现算法 ---
                print("正在运行PC算法进行因果发现...")
                causal_graph = discover_causal_structure(df_encoded, column_names)
                
                print("\nPC算法发现的因果图边:")
                edges = causal_graph.G.get_graph_edges()
                
                if not edges:
                    print("  -> 算法未发现任何因果边。")
                else:
                    for edge in edges:
                        node1 = edge.get_node1()
                        node2 = edge.get_node2()
                        print(f"  -> {node1.get_name()} {edge.get_endpoint1()}--{edge.get_endpoint2()} {node2.get_name()}")
        else:
            print("错误: JSON格式不正确或为空。")
            return
                        
    except Exception as e:
        print(f"读取或解析JSON文件时出错: {e}")
        return

   
            
if __name__ == '__main__':
    main()
