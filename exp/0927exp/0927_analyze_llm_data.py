## 分析由LLM直接生成的连续型数据中的因果效应

import os
import json
import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from sklearn.preprocessing import StandardScaler

def discover_causal_structure(data, node_names):
    """
    对给定的数据集运行PC因果发现算法。
    """
    data_np = data.to_numpy()
    cg = pc(data_np, alpha=0.05, node_names=node_names)
    return cg

def main():
    """
    主函数，加载LLM直接生成的连续型数据文件，执行因果发现并打印结果。
    """
    # 定义新的输入文件路径
    json_file_path = 'outcome/927_outcome/final_data.json'
    
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
    
                # 数据清洗：移除不相关的'id'列
                if 'id' in df.columns:
                    df = df.drop('id', axis=1)
                    print("已移除'id'列，因为它不参与因果分析。")

                # 数据预处理：对连续数据进行标准化
                # 连续数据通常需要标准化以确保PC算法的最佳性能
                print("正在对连续数据进行标准化...")
                scaler = StandardScaler()
                
                df_scaled = pd.DataFrame(
                    scaler.fit_transform(df), 
                    columns=df.columns,
                    index=df.index
                )
                
                # 转换为标准列
                column_names = df_scaled.columns.tolist()
                print(f"数据预处理完成。最终数据维度: {df_scaled.shape}")
                print(f"变量列表: {column_names}")

                # 运行因果发现算法
                print("正在运行PC算法进行因果发现...")
                causal_graph = discover_causal_structure(df_scaled, column_names)
                
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
