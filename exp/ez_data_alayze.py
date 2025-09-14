import json
from collections import Counter

with open("exp/outcome/ez_gpt_output.json", "r", encoding="utf-8") as f:
    data_list = json.load(f)

truth_ans = {"癌症","肺癌"}
topk_value = [1,3,5]
# 正确率
accuary = 0
# topk列表
topk_list = {k:0 for k in topk_value}
# all hypotheses
all_hypotheses = []
# 倒数排名总和
reciprocal_ranks_sum = 0.0


for data in data_list:
    found = False
    for hypothesis in data["confounder_hypotheses"]:
        all_hypotheses.append(hypothesis["confounder"])
        

        if any(ans in hypothesis["confounder"] for ans in truth_ans) and not found:
            accuary += 1
            found = True
            rank = hypothesis["rank"]

            reciprocal_ranks_sum += 1.0 / rank

            for k in topk_value:
                if rank <= k:
                    topk_list[k] += 1

hit_rate = accuary / len(data_list) if len(data_list) > 0 else 0
mrr = reciprocal_ranks_sum / len(data_list) if len(data_list) > 0 else 0
top_k_accuracy = {k: (count / len(data_list) if len(data_list) > 0 else 0) for k, count in topk_list.items()}

hypothesis_frequency = Counter(all_hypotheses)

sorted_hypotheses = hypothesis_frequency.most_common() 
for hypothesis, count in sorted_hypotheses:
    percentage = (count / len(data_list)) * 100
    print(f"  - {hypothesis:<25}: 提出了 {count:>2} 次 (频率: {percentage:.1f}%)")


print(f"正确率: {hit_rate}")
print(f"倒数排名总和: {mrr}")
print(f"topk列表: {top_k_accuracy}")
