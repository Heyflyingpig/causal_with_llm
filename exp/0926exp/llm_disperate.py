from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def get_confounder_hypotheses(*variables: str, client: OpenAI):

    if len(variables) < 2:
        raise ValueError("请至少提供两个变量。")
    if len(variables) == 2:
        variables_str = f'“{variables[0]}”与“{variables[1]}”'
    else:
        variables_str = '、'.join([f'“{v}”' for v in variables[:]])
    
    # 分两步进行，变量单独进行判断，再给数据
    prompt = f"""
    你是一位医学领域的因果推断专家。

    **背景**：在一个医学研究中，我们分析了一组病人的数据，并给出两个变量{variables_str}

    **任务**：
    1. 首先你需要判断这两个变量之间是否有直接关联，如果你认为其中不存在隐变量，请输出“否”，如果存在隐变量，请输出“是”，并且按要求按格式给出混淆隐变量列表
    2. 如果存在混淆变量，请你提出最有可能的`3到5个`的**隐藏的、共同的原因**，这个原因能够同时导致{variables_str}这些现象。换句话说，什么样的潜在疾病或身体状况可以合理解释我们观察到的这种关联？请列出可能性，并简要说明理由。
    3. 对于你提出的每个混淆变量，需要输出其发生的**先验概率**。这个概率应该是一个具体的数值（例如0.05），其含义代表该混淆变量在普通人群中为“是”的概率。

    **要求**: 
    4. 首先需要判断是否存在混淆变量，有的话输出“是”，没有的话输出“否”
    5. 如果存在混淆变量，请对你提出的混淆变量进行排名，并且按要求输出先验概率并说明理由。
    6. 如果不存在混淆变量，请不需要完成后续步骤
    7. 必须以严格的JSON格式输出，不要包含任何JSON格式之外的解释性文字。输出的JSON对象应包含以下键：
       - "variables": 一个包含输入变量的列表。
       - "is_confounder": 一个布尔值，表示是否存在混淆变量。
       - "confounder_variables": 一个包含所有你认为的潜在原因的列表，每一个对象表示潜在原因。
       - "Probability": 一个对象列表，每个对象表示一个潜在原因的先验概率。每个对象应包含 "confounder" (str) 和 "probability" (float) 两个键。
       - "confounder_hypotheses": 一个对象列表，每个对象代表一个混淆变量假说，并包含以下键：
         - "rank": 排名 (整数)，排序从依据是以你认为的可能性百分比从高到低排序。
         - "confounder": 混淆变量的名称 (字符串)。
         - "reasoning": 简要说明理由 (字符串)。
         - "causal_graph": 一个描述因果图的字符串，例如 "混淆变量 -> 观察变量1, 混淆变量 -> 观察变量2"。

       

    **输出格式示例，其中数据均为举例**:
    ```json
    {{
      "variables": ["变量A", "变量B"],
      "is_confounder": true | false,
      "confounder_variables": ["你认为的潜在原因1", "你认为的潜在原因2"],
      "Probability": [
        {{
          "confounder": "潜在原因1",
          "probability": 0.15
        }},
        {{
          "confounder": "潜在原因2",
          "probability": 0.08
        }}
      ],
      "confounder_hypotheses": [
        {{
          "rank": 1,
          "confounder": "潜在原因1",
          "reasoning": "这是最可能的原因，因为..."
          "causal_graph": "潜在原因2 -> 变量A; 潜在原因2 -> 变量B"
          
        }},
        {{
          "rank": 2,
          "confounder": "潜在原因2",
          "reasoning": "这个原因的可能性较低，因为..."
          "causal_graph": "潜在原因2 -> 变量A; 潜在原因2 -> 变量B"
        }}
      ],

      
    }}
    ```
    """
    response = client.chat.completions.create(
        model="glm-4.5-air",
        messages=[
            {"role": "user", "content": prompt}
        ]
        
    )

    llm_hypotheses = response.choices[0].message.content
    return llm_hypotheses


def data_llm(*variables: str, confounder_variables: list, var_list: list , client: OpenAI):


    if len(variables) < 2:
        raise ValueError("请至少提供两个变量。")
    if len(variables) == 2:
        variables_str = f'“{variables[0]}”与“{variables[1]}”'
    else:
        variables_str = '、'.join([f'“{v}”' for v in variables[:]])

    var_list_str = json.dumps(var_list, ensure_ascii=False, indent=2)
    confounder_variables_str = str(confounder_variables)
    
    # 使用 .format() 方法代替f-string，以安全地处理包含JSON示例的提示文本
    prompt_data = """
    你是一位严谨的因果数据科学家，擅长根据概率模型生成高质量的合成数据
    **背景**: 我们正在研究一个因果假设，并需要你根据下面提供的概率信息，生成一个符合统计规律的合成数据集。

    **任务**: 
    生成一个包含所有混杂隐变量需要的数据集。这个数据集必须严格反映以下变量之间的 **概率性因果关系** ，而不是确定性规则。
    **变量与因果假设**:
    - 观察变量: {var_str}
    - 隐混杂变量及其先验概率: {conf_vars_str}
    - 观察变量对应的数据：{var_list_str}
    你需要根据我们推测出的混淆隐变量以及其先验概率(Probability)。生成一组新的、包含所有对应记录的数据，这组新数据要符合这些变量与混淆变量之间的因果关系。

    **要求**: 
    **关键生成原则 (必须严格遵守)**:
    1.  **关系是概率性的，不是确定性的**: 例如，当某一列的数据`{conf_vars_str}` 为 "true" 时，`{var_str}` 有 `90%` 的概率为 "Abnormal"，但仍有 `10%` 的概率为 "Normal"。你生成的数据必须能体现这种随机性。**绝对禁止**让它们100%对应！
    2. **模拟随机抽样**: 请想象你正在进行100次独立的随机实验。在每一次实验中，你首先根据 `{conf_vars_str}` 中的先验概率决定其状态，然后按照观察变量生成可能性的混淆变量数据。
    2. 生成的数据需要根据先验概率(Probability)生成，并且需要符合因果关系。
    3. 你必须以严格的JSON格式输出，不要包含任何JSON格式之外的解释性文字，对于每一个混淆变量单独生成数据输出,要求如下：
            - "variables": 一个包含输入变量的列表。
            - "confounder_variables": 一个包含混淆隐变量的列表。
            - "data": 一个包含所有数据对象的列表，每个对象包含所有相关变量（观察变量和混淆变量）的值。
            - "id": 一个包含数据集id的列表。

    **输出格式示例**:
    ```json
    [
        {{
            
        "variables": ["变量A", "变量B"],
        "confounder_variables": ["混淆变量1"],
        "data": [
            {{"变量A": "值", "变量B": "值", "混淆变量1": "值", "id": "值"}},
            {{"变量A": "值", "变量B": "值", "混淆变量1": "值", "id": "值"}}
        ]
        }},
            {{
        "variables": ["变量A", "变量B"],
        "confounder_variables": ["混淆变量2"],
        "data": [
            {{"变量A": "值", "变量B": "值", "混淆变量2": "值", "id": "值"}},
            {{"变量A": "值", "变量B": "值", "混淆变量2": "值", "id": "值"}}
        ]
        }}
    ]
    
    
    ```
    """.format(
        var_str=variables_str,
        conf_vars_str=confounder_variables_str,
        var_list_str=var_list_str
    )
    
    response_data = client.chat.completions.create(
        model="glm-4.5",
        messages=[
            {"role": "user", "content": prompt_data}
        ],
        temperature=0.8, #提高一点

    )
    llm_data = response_data.choices[0].message.content
    return llm_data


def chat_confounder(client, num_runs, first_results_list):    
    
    for i in range(num_runs):
        
        observed_variables = ["X-ray Result", "Dyspnea Symptom"] # 修正变量名
        # 使用 f-string 来格式化字符串，让输出更清晰
        print(f"Running LLM call {i + 1}/{num_runs}...")
        
        
        try:
            # 将API调用移入try块，以便捕获网络或API错误
            hypotheses_str = get_confounder_hypotheses(*observed_variables, client=client)
            
            # 同样需要处理LLM可能返回的代码块标记
            if hypotheses_str.strip().startswith("```json"):
                hypotheses_str = hypotheses_str.strip()[7:-3].strip()
            
            single_run_data = json.loads(hypotheses_str)
            
            # 检查LLM的判断，如果不存在混淆变量，则跳过本次结果
            if not single_run_data.get("is_confounder", False):
                print(f"第 {i + 1} 次调用：LLM判断不存在混淆变量，跳过记录。")
                continue
            
            single_run_data['id'] = i + 1
            
            first_results_list.append(single_run_data)
            print(f"第 {i + 1} 次调用成功并已记录混淆变量生成。")

        except json.JSONDecodeError as e:
            # 如果某一次调用失败，打印错误信息并跳过，继续下一次调用
            print(f"混淆变量生成第 {i + 1} 次调用时解析JSON失败: {e}")
            print("原始字符串:", hypotheses_str)
        except Exception as e:
            print(f"混淆变量生成第 {i + 1} 次调用时发生未知错误: {e}")

    

        

def chat_data(client, hypotheses_list, data_list):
    for i, hypothesis in enumerate(hypotheses_list):
        print(f"为第 {i + 1}/{len(hypotheses_list)} 个假设生成数据...")
        try:
            # 从假设中提取信息
            confounder_info = hypothesis['Probability'][0]
            
            ## 提取变量名称
            observed_vars = hypothesis['variables']
            
            # 读取原始数据
            df = pd.read_csv('oringnal_data/bnlearn_generate/generated_cancer_dataset.csv')
            df_subset = df[['Xray', 'Dyspnoea']].head(200)

            # 重命名列以匹配假设
            df_subset.columns = observed_vars

            # 将DataFrame转换为字典列表
            var_list = df_subset.to_dict(orient='records')
            
            # 调用LLM生成数据
            data_str = data_llm(*observed_vars, confounder_variables=confounder_info, var_list=var_list, client=client)
            
            if data_str.strip().startswith("```json"):
                data_str = data_str.strip()[7:-3].strip()
            json_run_data = json.loads(data_str)
            
            data_list.extend(json_run_data)
            print(f"为第 {i + 1} 个假设生成数据成功。")

        except Exception as e:
            print(f"为第 {i + 1} 个假设生成数据时发生未知错误: {e}")



if __name__ == '__main__':
    ## 所有假说列表
    all_hypotheses_data = [] 
    all_data = []
    
    client = OpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=os.getenv("OPENAI_API_KEY"),       
    )
    try:
        ## 运行数量
        chat_confounder(client, num_runs=1, first_results_list=all_hypotheses_data)
        
    except Exception as e:
        print(f"\n程序发生严重错误: {e}")

    finally:
        # finally块确保无论是否发生异常，都会执行这部分代码
        if all_hypotheses_data:
            output_filename = "outcome/926_outcome/var_glm_output_test.json"

            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(all_hypotheses_data, f, indent=4, ensure_ascii=False)
            
            
            print(f"\n所有 {len(all_hypotheses_data)} 次运行的结果已成功保存到文件: {output_filename}")
        else:
            print("\n没有成功获取到任何结果，不生成文件。")
    
    ## 第二次调用，进行数据集生成
    try:
        # 确保有假设数据后再进行
        if all_hypotheses_data:
            chat_data(client, hypotheses_list=all_hypotheses_data, data_list=all_data)
    except Exception as e:
        print(f"\n程序发生严重错误: {e}")

    finally:
        if all_data:
            output_data_filename = "outcome/926_outcome/data_glm_data_test.json"
            with open(output_data_filename, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=4, ensure_ascii=False)
            print(f"\n所有 {len(all_data)} 次运行的结果已成功保存到文件: {output_data_filename}")
