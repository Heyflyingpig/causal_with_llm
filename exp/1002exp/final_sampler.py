import json
import os
import pandas as pd
import numpy as np

def sample_from_distribution(record, confounder_name, num_samples=7466):
    """
    根据单条记录中的分布类型和参数，生成一个随机样本。
    """
    distributions_type = record[confounder_name]
    dist_type = distributions_type.lower()  # 转换为小写以进行不区分大小写的匹配

    params = record["参数"]
    
    sampled_value = [] # 默认返回空列表

    # 使用更鲁棒的关键词匹配
    if any(keyword in dist_type for keyword in ["正态", "normal", "gaussian", "高斯"]):
        print("正态分布")
        mean = params.get("mean", params.get("mu", 0))
        std = params.get("std", params.get("sigma", 1))
        if std < 0:
            print(f"警告: 标准差为负数 ({std})。将使用其绝对值。")
            std = abs(std)
        sampled_value = np.random.normal(loc=mean, scale=std, size=num_samples)
    
    # 伯努利
    elif any(keyword in dist_type for keyword in ["伯努利", "bernoulli"]):
        print(f"伯努利分布")
        p = params.get("p", 0.5)  # 成功（即为1）的概率
        sampled_value = np.random.binomial(1, p, size=num_samples)
    
    # 均匀
    elif any(keyword in dist_type for keyword in ["均匀", "uniform"]):
        low = params.get("low", 0)
        high = params.get("high", 1)
        sampled_value = np.random.uniform(low=low, high=high, size=num_samples)
    # 分类
    elif any(keyword in dist_type for keyword in ["分类", "categorical"]):
        categories = params.get("categories", [])
        probabilities = params.get("probabilities", [])
        if categories and probabilities and len(categories) == len(probabilities):
            sampled_value = np.random.choice(categories, p=probabilities, size=num_samples)
        else:
            print(f"警告: 分类分布 '{confounder_name}' 的参数不完整或不匹配。")
    else:
        print(f"警告: 未知的分布类型 '{record[confounder_name]}分布类型'，无法进行采样。")

    # 将numpy数组转换为列表，以便JSON序列化
    return sampled_value.tolist() if isinstance(sampled_value, np.ndarray) else sampled_value

def main():
    """
    主函数，加载包含分布参数的JSON，进行采样，并保存最终的数据集。
    """
    input_path = 'outcome/1002_outcome/data_glm_data_test.json'
    output_path = 'outcome/1002_outcome/final_data.json'

    if not os.path.exists(input_path):
        print(f"错误: 输入文件 '{input_path}' 不存在。")
        return

    print(f"--- 开始处理文件: {input_path} ---")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    processed_data = []  # 用于存储所有处理后的运行数据
    # 遍历JSON中的每个部分
    for run_data in all_data:
        confounder_name = run_data.get("confounder_variables", [None])[0]
        if not confounder_name:
            continue

        data_records = run_data.get("data", [])
        
        # 为当前运行创建一个结果字典
        run_result = {}
        
        # 遍历每一条记录进行采样和替换
        for record in data_records:
            # 1. 生成采样值
            final_values = sample_from_distribution(record, confounder_name)
            
            # 2. 将采样值存入结果字典
            if final_values:
                run_result[confounder_name] = final_values
        
        processed_data.append(run_result)

    # 保存处理后的完整数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
        
    print(f"最终的采样数据集已成功保存到: {output_path}")



if __name__ == '__main__':
    main()
