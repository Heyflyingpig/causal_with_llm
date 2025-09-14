from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

def get_confounder_hypotheses(*variables: str, client: OpenAI):

    if len(variables) < 2:
        raise ValueError("请至少提供两个变量。")
    if len(variables) == 2:
        variables_str = f'“{variables[0]}”与“{variables[1]}”'
    else:
        variables_str = '、'.join([f'“{v}”' for v in variables[:]])
    
    prompt = f"""
    你是一位医学领域的因果推断专家。

    **背景**：在一个医学研究中，我们分析了一组病人的数据，并给出两个变量{variables_str}

    **任务**：
    1. 首先你需要判断这两个变量之间是否有直接关联，如果你认为其中不存在混淆变量，请输出“否”，如果存在混淆变量，请输出“是”，并且按要求按格式给出混淆变量列表
    2. 如果存在混淆变量，请你提出最有可能的`3到5个`的**隐藏的、共同的原因**，这个原因能够同时导致{variables_str}这些现象。换句话说，什么样的潜在疾病或身体状况可以合理解释我们观察到的这种关联？请列出可能性，并简要说明理由。
    3. 对于该混淆变量需要输出先验概率
    4. 对于你提出的每一个混淆变量，请提供观察变量（"X光检查结果", "呼吸困难症状"）在该混淆变量影响下的条件概率。你需要为观察变量定义两种状态（例如：“阳性”/“阴性”，或“出现”/“未出现”），并给出在这两种状态下的概率。

    **要求**: 
    1. 首先需要判断是否存在混淆变量，有的话输出“是”，没有的话输出“否”
    2. 如果存在混淆变量，请对你提出的混淆变量进行排名，并且按要求输出先验概率并说明理由。
    3. 如果不存在混淆变量，请不需要完成后续步骤
    4. 必须以严格的JSON格式输出，不要包含任何JSON格式之外的解释性文字。输出的JSON对象应包含以下键：
       - "variables": 一个包含输入变量的列表。
       - "is_confounder": 一个布尔值，表示是否存在混淆变量。
       - "confounder_variables": 一个包含所有你认为的潜在原因的列表，每一个对象表示潜在原因。
       - "Probability": 一个包含所有你认为的潜在原因的先验概率的列表，每一个对象表示潜在原因的先验概率。
       - "confounder_hypotheses": 一个对象列表，每个对象代表一个混淆变量假说，并包含以下键：
         - "rank": 排名 (整数)，排序从依据是以你认为的可能性百分比从高到低排序。
         - "confounder": 混淆变量的名称 (字符串)。
         - "reasoning": 简要说明理由 (字符串)。
         - "causal_graph": 一个描述因果图的字符串，例如 "混淆变量 -> 观察变量1, 混淆变量 -> 观察变量2"。
       - "conditional_probabilities": 一个对象列表，每个对象代表一个混淆变量下的条件概率，并包含以下键：
         - "confounder": 混淆变量的名称 (字符串)。
         - "probabilities": 一个对象列表，每个对象包含一个观察变量的条件概率表(CPT)。
           - "observed_variable": 观察变量的名称 (字符串)。
           - "cpt": 一个对象，表示条件概率表，键为观察变量的状态，值为概率。
       

    **输出格式示例**:
    ```json
    {{
      "variables": ["变量A", "变量B"],
      "is_confounder": true,
      "confounder_variables": ["你认为的潜在原因1", "你认为的潜在原因2"],
      "Probability": [
        {{
          "confounder": "潜在原因1",
          "probability": 
        }},
        {{
          "confounder": "潜在原因2",
          "probability": 
        }}
      ],
      "confounder_hypotheses": [
        {{
          "rank": 1,
          "confounder": "潜在原因1",
          "reasoning": "这是最可能的原因，因为..."
          
        }},
        {{
          "rank": 2,
          "confounder": "潜在原因2",
          "reasoning": "这个原因的可能性较低，因为..."
          "causal_graph": "潜在原因2 -> 变量A; 潜在原因2 -> 变量B"
        }}
      ],
      "conditional_probabilities": [
        {{
            "confounder": "潜在原因1",
            "probabilities": [
                {{
                    "observed_variable": "变量A",
                    "cpt": {{ "状态1": 0.8, "状态2": 0.2 }}
                }},
                {{
                    "observed_variable": "变量B",
                    "cpt": {{ "状态1": 0.7, "状态2": 0.3 }}
                }}
            ]
        }}
      ]
      
    }}
    ```
    """
    response = client.chat.completions.create(
        model="claude-sonnet-4-20250514",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    llm_hypotheses = response.choices[0].message.content
    return llm_hypotheses

def chat_llm(client, num_runs, results_list):
    

    for i in range(num_runs):
        
        observed_variables = ["X光检查结果", "呼吸困难症状"]
        # 使用 f-string 来格式化字符串，让输出更清晰
        print(f"正在进行第 {i + 1}/{num_runs} 次LLM调用...")
        
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
            
            results_list.append(single_run_data)
            print(f"第 {i + 1} 次调用成功并已记录。")

        except json.JSONDecodeError as e:
            # 如果某一次调用失败，打印错误信息并跳过，继续下一次调用
            print(f"第 {i + 1} 次调用时解析JSON失败: {e}")
            print("原始字符串:", hypotheses_str)
        except Exception as e:
            print(f"第 {i + 1} 次调用时发生未知错误: {e}")
            

if __name__ == '__main__':
    all_hypotheses_data = [] # 在try块外初始化列表
    try:
        client = OpenAI(
            base_url="https://api.anglergap.org/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            default_headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}
        )
        
        
        chat_llm(client, num_runs=1, results_list=all_hypotheses_data)
        
    except Exception as e:
        print(f"\n程序发生严重错误: {e}")

    finally:
        # finally块确保无论是否发生异常，都会执行这部分代码
        if all_hypotheses_data:
            output_filename = "exp/914_outcome/ez_glm_output.json"
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(all_hypotheses_data, f, indent=4, ensure_ascii=False)
            
            print(f"\n所有 {len(all_hypotheses_data)} 次运行的结果已成功保存到文件: {output_filename}")
        else:
            print("\n没有成功获取到任何结果，不生成文件。")

