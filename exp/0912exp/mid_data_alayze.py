import json
from collections import Counter
import statistics

with open("exp/outcome/mid_glm_output.json", "r", encoding="utf-8") as f:
    data_list = json.load(f)

truth_ans = {"亚洲","肺结核","肺癌","支气管炎","长期吸烟"}

num_runs = len(data_list)
all_precisions = [] # 存储每一个假说的精确率
all_recalls = []    # 存储每一个假说的召回率
all_hypotheses_flat = [] # 用于存储所有被提出的假设

# 专门用于rank-1假设的统计
rank_1_precisions = []  # 存储每次运行中rank-1假设的精确率
rank_1_recalls = []     # 存储每次运行中rank-1假设的召回率

# 用于按运行分组的统计
run_avg_precisions = []  # 每次运行的平均精确率
run_avg_recalls = []     # 每次运行的平均召回率

# --- 4. 遍历并评估每一次运行的结果 ---
for run_idx, run in enumerate(data_list):
    run_precisions = []  # 当前运行的所有精确率
    run_recalls = []     # 当前运行的所有召回率
    
    for hypothesis in run["confounder_hypotheses"]:
        proposed_vars = hypothesis["confounder_variables"]
        
        all_hypotheses_flat.extend(proposed_vars) # 用于最终的频率统计,用于假设一致性

        # 实现模糊匹配来计算“正确识别”的数量
        correctly_identified_count = 0
        # 遍历LLM提出的每一个变量
        for prop_var in proposed_vars:
            if any(keyword in prop_var for keyword in truth_ans):
                correctly_identified_count += 1
        
        # b. 计算精确率
        precision = correctly_identified_count / len(proposed_vars) if proposed_vars else 0.0
        all_precisions.append(precision)
        run_precisions.append(precision)
        
        # c. 模糊匹配计算"覆盖"的关键词数
        covered_keywords = {kw for var in proposed_vars for kw in truth_ans if kw in var}
        
        # d. 计算召回率
        recall = len(covered_keywords) / len(truth_ans) if truth_ans else 0.0
        all_recalls.append(recall)
        run_recalls.append(recall)

        # e. 如果是rank-1假设，单独记录
        if hypothesis.get("rank") == 1:
            rank_1_precisions.append(precision)
            rank_1_recalls.append(recall)
    
    # f. 记录每次运行的平均指标
    run_avg_precisions.append(statistics.mean(run_precisions) if run_precisions else 0)
    run_avg_recalls.append(statistics.mean(run_recalls) if run_recalls else 0)

# --- 5. 计算最终的聚合指标 ---
# 全局平均指标（所有假设的平均）
overall_avg_precision = statistics.mean(all_precisions) if all_precisions else 0
overall_avg_recall = statistics.mean(all_recalls) if all_recalls else 0

# 最佳表现指标
max_precision = max(all_precisions) if all_precisions else 0
max_recall = max(all_recalls) if all_recalls else 0

# Rank-1平均指标（所有运行中rank-1假设的平均）
rank_1_avg_precision = statistics.mean(rank_1_precisions) if rank_1_precisions else 0
rank_1_avg_recall = statistics.mean(rank_1_recalls) if rank_1_recalls else 0



# 稳定性指标（标准差）
precision_std = statistics.stdev(run_avg_precisions) if len(run_avg_precisions) > 1 else 0
recall_std = statistics.stdev(run_avg_recalls) if len(run_avg_recalls) > 1 else 0

# 假设一致性
hypothesis_frequency = Counter(all_hypotheses_flat)

# --- 6. 打印全面的多次运行分析报告 ---
print("=" * 50)
print(f"多次运行复杂因果叙事分析报告")
print(f"运行次数: {num_runs} | 总叙事数: {len(all_precisions)}")
print("=" * 50)

print(f"\n 全局平均指标 :")
print(f"  - 全局平均精确率: {overall_avg_precision:.2%}")
print(f"  - 全局平均召回率: {overall_avg_recall:.2%}")


print(f"\n最佳表现指标 :")
print(f"  - 最高精确率: {max_precision:.2%}")
print(f"  - 最高召回率: {max_recall:.2%}")


print(f"\nRank-1平均指标 (Rank-1 Average Metrics):")
print(f"  - Rank-1平均精确率: {rank_1_avg_precision:.2%}")
print(f"  - Rank-1平均召回率: {rank_1_avg_recall:.2%}")

print(f"\n稳定性分析 、:")
if len(run_avg_precisions) > 1:
    print(f"  - 精确率标准差: {precision_std:.2%} (越小表示越稳定)")
    print(f"  - 召回率标准差: {recall_std:.2%} (越小表示越稳定)")
    stability_score = "稳定" if precision_std < 0.1 and recall_std < 0.1 else "中等" if precision_std < 0.2 and recall_std < 0.2 else "波动较大"
    print(f"  - 稳定性评价: {stability_score}")
else:
    print("  - 需要更多运行数据来计算稳定性")

print(f"\n假设一致性分析 (按提出频率排序):")
sorted_hypotheses = hypothesis_frequency.most_common(10)  # 只显示前10个
for term, count in sorted_hypotheses:
    percentage = (count / len(all_hypotheses_flat)) * 100
    print(f"  - {term:<30}: {count:>2}次 ({percentage:.1f}%)")
