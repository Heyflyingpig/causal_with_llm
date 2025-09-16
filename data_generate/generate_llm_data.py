## 生成llm生成数据

import json
import os
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

def create_and_sample_network(llm_run_data, output_dir):
    """
    根据单次LLM运行的数据，为每个混淆变量假设创建、验证并采样贝叶斯网络。

    参数:
        llm_run_data (dict): 从ez_glm_output.json中读取的单次运行数据。
        output_dir (str): 保存生成的数据集的目录。
    """
    run_id = llm_run_data['id']
    observed_variables = llm_run_data['variables']

    # 确保混淆变量、概率和假设都存在
    if not all(k in llm_run_data for k in ['confounder_variables', 'Probability', 'conditional_probabilities']):
        print(f"[Run ID: {run_id}] 数据不完整，缺少关键键，跳过。")
        return

    # 遍历该次运行中LLM提出的每个混淆变量
    for confounder_hypothesis in llm_run_data['confounder_hypotheses']:
        confounder_name = confounder_hypothesis['confounder']
        
        print(f"\n[Run ID: {run_id}] 正在处理混淆变量: {confounder_name}")

        # --- 1. 定义贝叶斯网络结构 ---
        # 结构是一个简单的分叉：Confounder -> Observed_Var1, Confounder -> Observed_Var2
        model = DiscreteBayesianNetwork([
            (confounder_name, observed_variables[0]),
            (confounder_name, observed_variables[1])
        ])

        # --- 2. 查找并定义先验概率 (CPD for Confounder) ---
        prior_prob_data = next((p for p in llm_run_data['Probability'] if p['confounder'] == confounder_name), None)
        if not prior_prob_data:
            print(f"  错误: 找不到混淆变量 '{confounder_name}' 的先验概率。")
            continue
        
        prob_true = prior_prob_data['probability']
        # 假设混淆变量是二元的 (True/False)
        cpd_confounder = TabularCPD(
            variable=confounder_name,
            variable_card=2,
            values=[[prob_true], [1 - prob_true]],
            state_names={confounder_name: [True, False]}
        )

        # --- 3. 查找并定义条件概率 (CPDs for Observed Variables) ---
        conditional_prob_data = next((c for c in llm_run_data['conditional_probabilities'] if c['confounder'] == confounder_name), None)
        if not conditional_prob_data:
            print(f"  错误: 找不到混淆变量 '{confounder_name}' 的条件概率表。")
            continue

        cpds_observed = []
        all_cpds_found = True
        for obs_var in observed_variables:
            obs_var_prob = next((p for p in conditional_prob_data['probabilities'] if p['observed_variable'] == obs_var), None)
            if not obs_var_prob:
                print(f"错误: 找不到观察变量 '{obs_var}' 的条件概率。")
                all_cpds_found = False
                break
            
            cpt = obs_var_prob['cpt']
            states = list(cpt['when_confounder_true'].keys())
            
            # 从JSON中提取概率值并构建CPD的值表
            # pgmpy的values格式是: [[P(V=s1|E=e1), P(V=s1|E=e2)], [P(V=s2|E=e1), P(V=s2|E=e2)]]
            prob_values = []
            for state in states:
                prob_when_true = cpt['when_confounder_true'][state]
                prob_when_false = cpt['when_confounder_false'][state]
                prob_values.append([prob_when_true, prob_when_false])

            cpd_obs = TabularCPD(
                variable=obs_var,
                variable_card=len(states),
                values=prob_values,
                evidence=[confounder_name],
                evidence_card=[2],
                state_names={
                    obs_var: states,
                    confounder_name: [True, False]
                }
            )
            cpds_observed.append(cpd_obs)

        if not all_cpds_found:
            continue

        # --- 4. 将所有CPD添加到模型并进行验证 ---
        model.add_cpds(cpd_confounder, *cpds_observed)
        
        try:
            model.check_model()
            print(f" 模型构建成功并通过验证。")
        except Exception as e:
            print(f"  模型验证失败: {e}")
            continue

        # --- 5. 从模型中采样生成数据 ---
        sampler = BayesianModelSampling(model)
        dataset = sampler.forward_sample(size=1000, seed=42)
        
        # --- 6. 保存数据到CSV文件 ---
        output_filename = f"run_{run_id}_{confounder_name}.csv"
        output_path = os.path.join(output_dir, output_filename)
        dataset.to_csv(output_path, index=False)
        print(f"  数据已成功保存到: {output_path}")


def main():
    """
    主函数，加载LLM输出，创建输出目录，并为每个假设生成数据。
    """
    # 定义输入文件和输出目录
    input_json_path = 'exp/914_outcome/ez_glm_output.json'
    output_data_dir = 'generate_data/llm_generated_data'

    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
        print(f"已创建输出目录: {output_data_dir}")

    # 读取LLM生成的JSON文件
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            all_runs_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 解析JSON文件失败 {input_json_path}")
        return

    # 遍历JSON中的每一次运行结果
    for run_data in all_runs_data:
        if run_data.get("is_confounder", False):
            create_and_sample_network(run_data, output_data_dir)
        else:
            print(f"\n[Run ID: {run_data.get('id', 'N/A')}] LLM判断无混淆变量，跳过。")

if __name__ == '__main__':
    main()
