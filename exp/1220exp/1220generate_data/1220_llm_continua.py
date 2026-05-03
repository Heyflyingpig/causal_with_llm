from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import time

load_dotenv()

# ==================== 分批处理配置 ====================
BATCH_SIZE = 100  # 每批处理的样本数量
MAX_SAMPLES = 20000  # 最大处理样本数量（设为None处理全部数据）
CHECKPOINT_DIR = 'outcome/1220_outcome/checkpoints'  # 断点续传保存目录
RETRY_DELAY = 5  # 失败后重试的等待时间(秒)
MAX_RETRIES = 3  # 最大重试次数


def load_twins_dataset():
    """
    Load Twins dataset from CSV files and extract relevant variables.
    Returns DataFrame with treatment, outcome, and covariates.

    Twins数据集说明 (基于ProCI论文):
    - 来源: 1989-1991年美国双胞胎出生记录
    - 样本: 出生体重低于2000g的双胞胎对
    - Treatment: 是否为较重的双胞胎 (heavier twin indicator)
    - Outcome: 一年死亡率 (one-year mortality)
    - Confounders: 父母特征、妊娠和出生相关的协变量

    数据处理说明:
    - twins.csv: 包含协变量信息 (每行是一对双胞胎)
    - death.csv: 包含死亡率信息 (mort_0=lighter twin, mort_1=heavier twin)
    - 我们将每对双胞胎拆分为两行，分别表示 lighter twin (T=0) 和 heavier twin (T=1)
    """
    # 加载协变量数据
    df_covariates = pd.read_csv('oringnal_data/bnlearn/twins/twins.csv')
    # 加载死亡率数据
    df_death = pd.read_csv('oringnal_data/bnlearn/twins/death.csv')

    # 删除索引列
    for col in ['Unnamed: 0.1', 'Unnamed: 0']:
        if col in df_covariates.columns:
            df_covariates = df_covariates.drop(columns=[col])
        if col in df_death.columns:
            df_death = df_death.drop(columns=[col])

    # 合并数据 (按行索引匹配)
    df = pd.concat([df_covariates.reset_index(drop=True),
                    df_death.reset_index(drop=True)], axis=1)

    # 选择关键协变量 (母亲特征 + 妊娠特征 + 健康状况)
    covariate_cols = [
        # 母亲人口统计特征
        'mager8',      # 母亲年龄组 (1-8 scale)
        'mrace',       # 母亲种族 (1=white, 2=black, etc.)
        'meduc6',      # 母亲教育水平 (1-6 scale)
        'dmar',        # 婚姻状况 (0=unmarried, 1=married)
        # 妊娠和出生特征
        'gestat10',    # 妊娠周数 (gestational age in 10 categories)
        'csex',        # 婴儿性别 (0=female, 1=male)
        'nprevistq',   # 产前检查次数
        # 母亲健康状况
        'anemia',      # 贫血
        'cardiac',     # 心脏病
        'diabetes',    # 糖尿病
        'chyper',      # 慢性高血压
        'phyper',      # 妊娠高血压
        'eclamp',      # 子痫
        # 行为因素
        'tobacco',     # 吸烟
        'alcohol',     # 饮酒
    ]

    # 确保所有列都存在
    available_cols = [col for col in covariate_cols if col in df.columns]

    # 将每对双胞胎拆分为两行数据
    # 行1: lighter twin (T=0), Outcome=mort_0
    # 行2: heavier twin (T=1), Outcome=mort_1
    rows_list = []

    for idx, row in df.iterrows():
        # 获取协变量值
        covariates = {col: row[col] for col in available_cols}

        # Lighter twin (Treatment = 0)
        row_lighter = covariates.copy()
        row_lighter['treat'] = 0.0
        row_lighter['mort_1yr'] = row['mort_0'] if pd.notna(row.get('mort_0')) else 0.0
        row_lighter['pair_id'] = idx  # 记录配对ID
        rows_list.append(row_lighter)

        # Heavier twin (Treatment = 1)
        row_heavier = covariates.copy()
        row_heavier['treat'] = 1.0
        row_heavier['mort_1yr'] = row['mort_1'] if pd.notna(row.get('mort_1')) else 0.0
        row_heavier['pair_id'] = idx  # 记录配对ID
        rows_list.append(row_heavier)

    df_twins = pd.DataFrame(rows_list)

    # 处理缺失值
    df_twins = df_twins.fillna(df_twins.median())

    # 重新排列列顺序
    final_cols = ['treat', 'mort_1yr', 'pair_id'] + available_cols
    df_twins = df_twins[final_cols]

    print(f"Loaded Twins dataset: {df_twins.shape[0]} samples, {df_twins.shape[1]} variables")
    print(f"Variables: {list(df_twins.columns)}")
    print(f"Treatment distribution: {df_twins['treat'].value_counts().to_dict()}")
    print(f"Mortality rate - Lighter twin (T=0): {df_twins[df_twins['treat']==0]['mort_1yr'].mean():.4f}")
    print(f"Mortality rate - Heavier twin (T=1): {df_twins[df_twins['treat']==1]['mort_1yr'].mean():.4f}")

    return df_twins


def get_prefix_prompt():
    """
    Generate prefix prompt following Appendix E.1 from the paper.
    This provides dataset introduction and variable descriptions for Twins dataset.
    """
    prefix = """Brief introduction of the Twins dataset:
The Twins dataset is derived from all recorded twin births in the United States between 1989 and 1991. It is widely used in causal inference research for evaluating treatment effect estimation methods. We focus on twin pairs where both individuals weighed less than 2000 grams at birth.

This observational dataset contains:
(1) Treatment - Heavier Twin Indicator: T ∈ {0,1} indicating whether the individual is the heavier twin of the pair.
(2) Outcome - One-Year Mortality (mort_1yr): Binary indicator of whether the infant died within one year. Y=1 means died, Y=0 means survived.
(3) Confounders - Parental and Birth Characteristics:
    - mager8: Mother's age group (1-8 scale)
    - mrace: Mother's race (1=white, 2=black, etc.)
    - meduc6: Mother's education level (1-6 scale)
    - dmar: Marital status (0=unmarried, 1=married)
    - gestat10: Gestational age categories (1-10, higher means more weeks)
    - csex: Child sex (0=female, 1=male)
    - nprevistq: Number of prenatal visits
    - anemia, cardiac, diabetes: Mother's health conditions (binary)
    - chyper, phyper, eclamp: Hypertension-related conditions (binary)
    - tobacco, alcohol: Mother's substance use during pregnancy (binary)"""

    return prefix


def get_confounder_variable(client: OpenAI):
    """
    P_var(X, T, Y): Generate a new confounder variable.
    Following Appendix E.2 from the paper.
    """
    prefix_prompt = get_prefix_prompt()
    
    prompt = f"""{prefix_prompt}

Based on your WORLD KNOWLEDGE, please propose one additional confounder which BOTH affects the treatment and outcome.

Make sure that the proposed confounder has a DIFFERENT MEANING compared to existing confounders.

For this proposed confounder, please provide:
(1) A clear name for the confounder.
(2) A brief explanation of why it affects both treatment and outcome.

You must output in strict JSON format without any explanatory text outside the JSON. The JSON object should contain:
- "confounder_name": The name of the confounder (string)
- "explanation": Brief explanation of why it affects both treatment and outcome (string)

Output format example:
```json
{{
  "confounder_name": "Transportation Access",
  "explanation": "Access to reliable transportation can influence both participation in the job training program (treatment) and subsequent employment (outcome). Individuals without transportation may be less likely to enroll in or attend the program due to logistical barriers. Similarly, lack of transportation can hinder job search efforts and commuting to workplaces, reducing the likelihood of employment."
}}
```"""

    response = client.chat.completions.create(
        model="glm-4.5-air",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    result = response.choices[0].message.content
    return result


def get_distribution_type(confounder_name: str, confounder_explanation: str, client: OpenAI):
    """
    P_dist(X, T, Y, U): Infer the distribution type of the confounder.
    Following Appendix E.3 from the paper.
    """
    prefix_prompt = get_prefix_prompt()
    
    prompt = f"""{prefix_prompt}

We have identified a potential confounder: {confounder_name}
Explanation: {confounder_explanation}

Based on your WORLD KNOWLEDGE, please provide the distribution type of confounder '{confounder_name}'. For example:
(1) Continuous - e.g., Normal distribution
(2) Discrete - e.g., Multi-categorical distribution
(3) Binary - e.g., Bernoulli distribution

You must output in strict JSON format. The JSON object should contain:
- "confounder": The confounder name (string)
- "distribution_type": The distribution type (string, e.g., "Normal", "Bernoulli", "Uniform", "Categorical")
- "value_description": Brief description of what the values represent (string)

Output format example:
```json
{{
  "confounder": "Transportation Access",
  "distribution_type": "Bernoulli",
  "value_description": "0 indicates the individual lacks reliable transportation, 1 indicates they have reliable transportation"
}}
```"""

    response = client.chat.completions.create(
        model="glm-4.5-air",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    result = response.choices[0].message.content
    return result


def get_param_description(distribution_type: str):
    """
    根据分布类型返回参数描述和示例
    """
    if "normal" in distribution_type.lower() or "gaussian" in distribution_type.lower():
        param_desc = "mean and standard deviation (std)"
        example_params = '{"mean": 0.5, "std": 0.2}'
    elif "bernoulli" in distribution_type.lower():
        param_desc = "probability p (between 0 and 1)"
        example_params = '{"p": 0.7}'
    elif "uniform" in distribution_type.lower():
        param_desc = "lower bound (low) and upper bound (high)"
        example_params = '{"low": 0, "high": 1}'
    elif "categorical" in distribution_type.lower():
        param_desc = "categories list and probabilities list"
        example_params = '{"categories": [0, 1, 2], "probabilities": [0.3, 0.5, 0.2]}'
    elif "exponential" in distribution_type.lower():
        param_desc = "lambda (rate parameter)"
        example_params = '{"lambda": 1.0}'
    elif "gamma" in distribution_type.lower():
        param_desc = "shape and scale parameters"
        example_params = '{"shape": 2.0, "scale": 1.0}'
    else:
        param_desc = "appropriate distribution parameters"
        example_params = '{"param1": "value1", "param2": "value2"}'
    return param_desc, example_params


def generate_parameters_for_batch(confounder_name: str, distribution_type: str,
                                   df_batch: pd.DataFrame, batch_start_idx: int,
                                   client: OpenAI):
    """
    为单个批次生成分布参数

    参数:
    - confounder_name: 混淆变量名称
    - distribution_type: 分布类型
    - df_batch: 当前批次的数据
    - batch_start_idx: 当前批次在原始数据中的起始索引
    - client: OpenAI客户端

    返回:
    - 解析后的参数列表
    """
    prefix_prompt = get_prefix_prompt()

    # 只保留关键列，减少token数量
    key_columns = [
        'treat', 'mort_1yr',  # treatment和outcome
        'mager8', 'mrace', 'meduc6', 'dmar',  # 母亲人口统计
        'gestat10', 'csex', 'nprevistq',  # 妊娠特征
        'anemia', 'cardiac', 'diabetes',  # 健康状况
        'chyper', 'phyper', 'eclamp',  # 高血压相关
        'tobacco', 'alcohol'  # 行为因素
    ]
    # 只选择存在的列
    available_cols = [col for col in key_columns if col in df_batch.columns]
    df_slim = df_batch[available_cols].copy()

    # 将批次数据转换为列表
    data_list = df_slim.to_dict(orient='records')
    # 为每条记录添加全局索引
    for i, record in enumerate(data_list):
        record['global_id'] = batch_start_idx + i
    data_str = json.dumps(data_list, ensure_ascii=False, indent=2)

    param_desc, example_params = get_param_description(distribution_type)

    prompt = f"""{prefix_prompt}

The values of existing confounders, treatments, and outcomes are given by:
{data_str}

For the confounder '{confounder_name}', which follows a {distribution_type} distribution, please specify {param_desc} for each individual from which we can sample the confounder value.

Base your parameter estimates on:
- The infant's observed features (mother's age, race, education, marital status, gestational age, sex)
- The mother's health conditions (anemia, cardiac, diabetes, hypertension, etc.)
- The mother's behaviors during pregnancy (tobacco, alcohol use)
- The treatment assignment (treat - heavier twin indicator)
- The outcome value (mort_1yr - one-year mortality)
- Your world knowledge about how '{confounder_name}' relates to infant mortality in twins

You must output in strict JSON format as a list. Each object in the list should contain:
- "id": The global_id of the individual (integer, use the global_id from input data)
- "parameters": The distribution parameters (object with parameter names and values)

Output format example:
```json
[
  {{"id": {batch_start_idx}, "parameters": {example_params}}},
  {{"id": {batch_start_idx + 1}, "parameters": {example_params}}},
  ...
]
```"""

    response = client.chat.completions.create(
        model="glm-4.5-air",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    result = response.choices[0].message.content

    # 调试：如果返回内容为空或异常，打印详细信息
    if not result or len(result.strip()) == 0:
        print(f"\n  [调试] LLM返回内容为空!")
        print(f"  [调试] response对象: {response}")
        raise ValueError("LLM返回内容为空")

    # 尝试解析JSON，失败时打印原始内容便于调试
    try:
        parsed = parse_llm_json(result)
        return parsed
    except json.JSONDecodeError as e:
        print(f"\n  [调试] JSON解析失败: {e}")
        print(f"  [调试] 原始返回内容前500字符: {result[:500]}")
        raise


def save_checkpoint(checkpoint_data: dict, checkpoint_file: str):
    """
    保存断点数据
    """
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    print(f"  [Checkpoint] 已保存到 {checkpoint_file}")


def load_checkpoint(checkpoint_file: str):
    """
    加载断点数据
    """
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def generate_parameters_batched(confounder_name: str, distribution_type: str,
                                 df_twins: pd.DataFrame, client: OpenAI,
                                 batch_size: int = BATCH_SIZE):
    """
    分批生成分布参数，支持断点续传

    参数:
    - confounder_name: 混淆变量名称
    - distribution_type: 分布类型
    - df_twins: 完整数据集
    - client: OpenAI客户端
    - batch_size: 每批处理的样本数量

    返回:
    - all_params: 所有样本的参数列表
    - total_samples: 处理的总样本数
    """
    total_samples = len(df_twins)
    total_batches = (total_samples + batch_size - 1) // batch_size

    # 断点文件路径
    checkpoint_file = os.path.join(
        CHECKPOINT_DIR,
        f'checkpoint_{confounder_name.replace(" ", "_")}.json'
    )

    # 尝试加载断点
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        all_params = checkpoint.get('params', [])
        completed_batches = checkpoint.get('completed_batches', 0)
        print(f"  [断点续传] 发现已完成 {completed_batches}/{total_batches} 批次，继续处理...")
    else:
        all_params = []
        completed_batches = 0
        print(f"  [新任务] 开始处理，共 {total_samples} 条数据，分 {total_batches} 批")

    # 使用tqdm显示进度，设置动态信息
    pbar = tqdm(total=total_batches, initial=completed_batches,
                desc="生成参数", unit="batch",
                dynamic_ncols=True,  # 动态调整宽度
                leave=True)  # 完成后保留进度条

    for batch_idx in range(completed_batches, total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        df_batch = df_twins.iloc[start_idx:end_idx].copy()

        # 更新进度条描述，显示当前处理的批次
        pbar.set_description(f"批次 {batch_idx + 1}/{total_batches}")
        pbar.set_postfix({"样本": f"{start_idx}-{end_idx}", "已生成": len(all_params)})

        # 重试机制
        success = False
        for retry in range(MAX_RETRIES):
            try:
                batch_params = generate_parameters_for_batch(
                    confounder_name, distribution_type,
                    df_batch, start_idx, client
                )
                all_params.extend(batch_params)

                # 保存断点（每批次成功后立即保存）
                checkpoint_data = {
                    'confounder_name': confounder_name,
                    'distribution_type': distribution_type,
                    'completed_batches': batch_idx + 1,
                    'total_batches': total_batches,
                    'total_samples': total_samples,
                    'params': all_params
                }
                save_checkpoint(checkpoint_data, checkpoint_file)

                success = True
                break  # 成功则跳出重试循环

            except Exception as e:
                # 使用tqdm.write避免打乱进度条
                tqdm.write(f"\n  [错误] 批次 {batch_idx + 1}/{total_batches} 失败: {e}")
                if retry < MAX_RETRIES - 1:
                    tqdm.write(f"  [重试] 等待 {RETRY_DELAY} 秒后重试 ({retry + 2}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY)
                else:
                    tqdm.write(f"  [失败] 批次 {batch_idx + 1} 达到最大重试次数，保存断点后退出")
                    # 保存当前进度（即使失败也保存已完成的部分）
                    checkpoint_data = {
                        'confounder_name': confounder_name,
                        'distribution_type': distribution_type,
                        'completed_batches': batch_idx,  # 注意：这里是batch_idx，不是batch_idx+1
                        'total_batches': total_batches,
                        'total_samples': total_samples,
                        'params': all_params,
                        'last_error': str(e)
                    }
                    save_checkpoint(checkpoint_data, checkpoint_file)
                    pbar.close()
                    raise Exception(f"批次 {batch_idx + 1} 处理失败，已保存断点（{len(all_params)}条数据），可重新运行继续")

        if success:
            # 成功后立即更新进度条
            pbar.update(1)
            pbar.set_postfix({"样本": f"{start_idx}-{end_idx}", "已生成": len(all_params), "状态": "OK"})

    pbar.close()
    print(f"  [完成] 共生成 {len(all_params)} 条参数数据")

    # 处理完成后删除断点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"  [清理] 已删除断点文件")

    return all_params, total_samples


def generate_parameters(confounder_name: str, distribution_type: str, df_twins: pd.DataFrame, client: OpenAI):
    """
    P_param(x_i, t_i, y_i): Generate distribution parameters for each individual.
    Following Appendix E.4 from the paper.

    这是原有函数的包装器，现在调用分批处理版本

    针对Twins数据集进行适配:
    - Treatment: treat (heavier twin indicator)
    - Outcome: mort_1yr (one-year mortality)
    - Confounders: 母亲特征、妊娠特征、健康状况等
    """
    # 调用分批处理版本
    all_params, sample_size = generate_parameters_batched(
        confounder_name, distribution_type, df_twins, client
    )

    # 返回格式与原函数兼容（但这里返回的是已解析的列表而非原始字符串）
    return all_params, sample_size


# 格式化输出
def parse_llm_json(llm_output: str) -> dict:
    """
    Parse LLM output, handling common JSON formatting issues.
    """
    # Remove markdown code blocks if present
    if "```json" in llm_output:
        start_idx = llm_output.find("```json") + 7
        end_idx = llm_output.find("```", start_idx)
        llm_output = llm_output[start_idx:end_idx].strip()
    elif llm_output.strip().startswith("```"):
        llm_output = llm_output.strip()[3:-3].strip()
    
    # Parse JSON
    result = json.loads(llm_output)
    return result


def main():
    """
    Main function implementing ProCI confounder generation for Twins dataset.
    Steps: Variable Generation -> Distribution Identification -> Parameter Inference

    针对Twins数据集:
    - Treatment: 是否为较重的双胞胎
    - Outcome: 一年死亡率
    - Confounders: 母亲特征、妊娠特征、健康状况等
    """


    # Initialize OpenAI client
    client = OpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Step 1: Load Twins dataset
    print("\n[Step 1/4] Loading Twins dataset...")
    df_twins = load_twins_dataset()

    # 限制处理的数据量
    if MAX_SAMPLES is not None and len(df_twins) > MAX_SAMPLES:
        print(f"  [限制] 只处理前 {MAX_SAMPLES} 条数据 (原始数据: {len(df_twins)} 条)")
        df_twins = df_twins.head(MAX_SAMPLES).reset_index(drop=True)

    # Step 2: Generate confounder variable
    print("\n[Step 2/4] Generating confounder variable...")
    try:
        var_output = get_confounder_variable(client)
        var_data = parse_llm_json(var_output)

        confounder_name = var_data['confounder_name']
        confounder_explanation = var_data['explanation']

        print(f"Generated confounder: {confounder_name}")
        print(f"  Explanation: {confounder_explanation}")

        # Save variable generation result
        with open('outcome/1220_outcome/twins_var_glm_output.json', 'w', encoding='utf-8') as f:
            json.dump(var_data, f, indent=4, ensure_ascii=False)
        print("Saved to outcome/1220_outcome/twins_var_glm_output.json")

    except Exception as e:
        print(f"Error in variable generation: {e}")
        print(f"Raw output: {var_output}")
        return

    # Step 3: Identify distribution type
    print("\n[Step 3/4] Identifying distribution type...")
    try:
        dist_output = get_distribution_type(confounder_name, confounder_explanation, client)
        dist_data = parse_llm_json(dist_output)

        distribution_type = dist_data['distribution_type']
        value_description = dist_data.get('value_description', '')

        print(f"Distribution type: {distribution_type}")
        print(f"  Value description: {value_description}")

    except Exception as e:
        print(f"Error in distribution identification: {e}")
        print(f"Raw output: {dist_output}")
        return

    # Step 4: Generate parameters for each individual (分批处理)
    print("\n[Step 4/4] Generating distribution parameters for each individual...")
    print(f"  配置: 每批 {BATCH_SIZE} 条数据，支持断点续传")
    try:
        # generate_parameters 现在返回已解析的列表，不再需要 parse_llm_json
        params_data, sample_size = generate_parameters(
            confounder_name, distribution_type, df_twins, client
        )

        print(f"\n生成参数完成，共 {len(params_data)} 条记录")

        # Combine all information into final data structure
        # 格式兼容 1220_final_sampler.py
        final_data = {
            "dataset": "twins",
            "confounder_name": confounder_name,
            "confounder_explanation": confounder_explanation,
            "distribution_type": distribution_type,
            "value_description": value_description,
            "sample_size": sample_size,
            "data": []
        }

        # 创建参数字典，以id为键，方便查找
        params_dict = {param_obj['id']: param_obj['parameters'] for param_obj in params_data}

        # Merge original data with generated parameters
        print("正在合并数据...")
        for i in range(sample_size):
            ## 每一行的每一个col的键值对应
            record = df_twins.iloc[i].to_dict()
            record[f'{confounder_name}_distribution_type'] = distribution_type

            # 使用id查找对应的参数
            if i in params_dict:
                record[confounder_name] = params_dict[i]
            else:
                print(f"  警告: 样本 {i} 没有找到对应的参数")
                record[confounder_name] = None

            record['id'] = i
            final_data['data'].append(record)

        # Save final data with parameters
        output_path = 'outcome/1220_outcome/twins_data_glm_output.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([final_data], f, indent=4, ensure_ascii=False)

        print(f"\n已保存到 {output_path}")
        print(f"文件格式兼容 1220_final_sampler.py")

        # Preview first few records
        print("\n" + "="*60)
        print("Preview of generated data (first 3 records):")
        print("="*60)
        for i in range(min(3, len(final_data['data']))):
            record = final_data['data'][i]
            print(f"\nRecord {i}:")
            print(f"  Mother Age Group: {record.get('mager8', 'N/A')}, "
                  f"Race: {record.get('mrace', 'N/A')}, "
                  f"Treatment: {record['treat']}, "
                  f"Outcome: {record['mort_1yr']}")
            print(f"  {confounder_name} parameters: {record[confounder_name]}")

    except Exception as e:
        print(f"\n[错误] 参数生成失败: {e}")
        print("如果是网络问题，可以重新运行脚本，程序会从断点继续")
        import traceback
        traceback.print_exc()
        return




if __name__ == '__main__':
    main()
