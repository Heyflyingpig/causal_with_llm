import json
import os
import pandas as pd
import numpy as np

def sample_from_distribution(record, confounder_name):
    """
    根据单条记录中的分布类型和参数，生成一个随机样本。
    """
    
    dist_type_key = f"{confounder_name}分布类型"
    
    # 检查是否存在对应的分布类型
    if dist_type_key not in record or confounder_name not in record:
        # 正常情况：如果已经是最终数据，则跳过
        return record.get(confounder_name)

    dist_type = record[dist_type_key].lower() # 转换为小写以进行不区分大小写的匹配
    params = record[confounder_name]


    # 使用更鲁棒的关键词匹配
    if any(keyword in dist_type for keyword in ["正态", "normal", "gaussian", "高斯"]):
        # 【修正】同时兼容 'mean'/'std' 和 'mu'/'sigma' 两种参数名
        mean = params.get("mean", params.get("mu", 0))
        std = params.get("std", params.get("sigma", 1))
        if std < 0:
            print(f"警告: 标准差为负数 ({std})。将使用其绝对值。")
            std = abs(std)
        sampled_value = np.random.normal(loc=mean, scale=std)
    
    # 伯努利
    elif any(keyword in dist_type for keyword in ["伯努利", "bernoulli"]):
        p = params.get("p", 0.5) # 成功（即为1）的概率
        sampled_value = np.random.binomial(1, p)
    
    # 均匀
    elif any(keyword in dist_type for keyword in ["均匀", "uniform"]):
        low = params.get("low", 0)
        high = params.get("high", 1)
        sampled_value = np.random.uniform(low=low, high=high)
    # 分类
    elif any(keyword in dist_type for keyword in ["分类", "categorical"]):
        categories = params.get("categories", [])
        probabilities = params.get("probabilities", [])
        if categories and probabilities and len(categories) == len(probabilities):
            sampled_value = np.random.choice(categories, p=probabilities)
        else:
            print(f"警告: 分类分布 '{confounder_name}' 的参数不完整或不匹配。")
    else:
        # 【新增】将未知类型明确打印出来，便于调试
        print(f"警告: 未知的分布类型 '{record[dist_type_key]}'，无法进行采样。")

    return sampled_value

def main():
    """
    主函数，加载包含分布参数的JSON，进行采样，并保存最终的数据集。
    """
    input_path = 'outcome/927_outcome/data_glm_data_test.json'
    output_path = 'outcome/927_outcome/final_data.json'

    if not os.path.exists(input_path):
        print(f"错误: 输入文件 '{input_path}' 不存在。")
        return

    print(f"--- 开始处理文件: {input_path} ---")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # 遍历JSON中的每个部分
    for run_data in all_data:
        confounder_name = run_data.get("confounder_variables", [None])[0]
        if not confounder_name:
            continue

        data_records = run_data.get("data", [])
        
        # 遍历每一条记录进行采样和替换
        for i, record in enumerate(data_records):
            original_params = record.get(confounder_name) # 保存原始参数以供检查
            
            # 1. 生成采样值
            new_value = sample_from_distribution(record, confounder_name)

            if new_value is not None:
                # 2. 用采样值替换参数字典
                record[confounder_name] = new_value
                
                # 3. 移除分布类型字段
                dist_type_key = f"{confounder_name}分布类型"
                if dist_type_key in record:
                    del record[dist_type_key]
            
            # 检查替换是否成功，如果不成功则发出明确警告
            if isinstance(record.get(confounder_name), dict):
                 print(f"错误: 第 {i+1} 条记录的参数替换失败！")
                 print(f"  - 混淆变量: {confounder_name}")
                 print(f"  - 分布类型: {record.get(f'{confounder_name}分布类型', '未找到')}")
                 print(f"  - 原始参数: {original_params}")


    # 保存处理后的完整数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
        
    print(f"最终的采样数据集已成功保存到: {output_path}")

    # 打印前5条记录作为预览
    if all_data and all_data[0].get("data"):
        print("\n最终数据集预览 (前5条记录):")
        preview_df = pd.DataFrame(all_data[0]["data"][:5])
        print(preview_df)


if __name__ == '__main__':
    main()
