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

    **背景**：在一个医学研究中，我们分析了一组病人的数据，发现了一个有趣的现象：病人的{variables_str}之间存在很强的统计相关性。并且存在这样的因果关系

    **任务**：
    1. 请不要仅仅列出单一的疾病名称。：你的核心任务是生成多个独立的、详细的 **“因果路径假说” ** 来解释这个现象
    2. 请详细描述从一个或者多个潜在的根源性因素开始，如何交叉引发一系列的病理变化，最终同时导致{variables_str}这些现象。

    **要求**: 
    1. 对你提出的因果叙事进行排名，并说明理由。
    2.  **复杂性要求**：每一个叙事必须包含一个或者多个根源因素，以及**>2个不同的中间疾病或病理状态**。请清晰地描述这个因果链条。
    3. 必须以严格的JSON格式输出，不要包含任何JSON格式之外的解释性文字。输出的JSON对象应包含以下键：
       - "variables": 一个包含输入变量的列表。      
       - "confounder_hypotheses": 一个对象列表，每个对象代表一个混淆变量假说，并包含以下键：       
         - "rank": 排名 (整数)，排序从依据是以你认为的可能性百分比从高到低排序。
         - "confounder_variables": 一个包含所有你认为叙事中所有潜在原因的列表，每一个对象都表示在你生成故事中的潜在原因，叙事原因的多少依据你所生成的叙事数量而定。
         - "confounder": 你所生成的包含因果效应叙事。
         - "reasoning": 简要说明理由 (字符串)。
         - "causal_graph": 一个描述因果图的字符串，例如 "混淆变量 -> 观察变量1, 混淆变量 -> 观察变量2"。
       

    **输出格式示例**:
    ```json
    {{
      "variables": ["变量A", "变量B","..."],
      "confounder_hypotheses": [
        {{
          "rank": 1,
          "confounder_variables": ["推导中的潜在原因1", "推到中的潜在原因2"，“...”],
          "confounder": "潜在原因1导致了变量A，潜在原因2导致了变量B,...",
          "reasoning": "这是最可能的原因，因为..."
          "causal_graph": "潜在原因2 -> 变量A; 潜在原因2 -> 变量B;..."
         
        }},
        {{
          "rank": 2,
          "confounder_variables": ["推导中的潜在原因1", "推到中的潜在原因2"，“...”],
          "confounder": "...",
          "reasoning": "这个原因的可能性较低，因为..."
          "causal_graph": "潜在原因2 -> 变量A; 潜在原因2 -> 变量B;..."
        }}
      ],
      
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

def chat_llm(client, num_runs):
    
    results_list = []
    
    for i in range(num_runs):
        causal_graph = "长期吸烟 -> 慢性支气管炎; 长期吸烟 -> 肺气肿; 慢性支气管炎 -> COPD; 肺气肿 -> COPD; COPD -> X光检查结果; COPD -> 呼吸困难症状"
        causal_graph = "长期吸烟 -> 慢性支气管炎; 长期吸烟 -> 肺气肿; 慢性支气管炎 -> COPD; 肺气肿 -> COPD; COPD -> X光检查结果; COPD -> 呼吸困难症状"
        observed_variables = ["X光检查结果", "呼吸困难症状"]

        # 使用 f-string 来格式化字符串，让输出更清晰
        print(f"正在进行第 {i + 1}/{num_runs} 次LLM调用...")
        
        hypotheses_str = get_confounder_hypotheses(*observed_variables, client=client)
        
        try:
            # 同样需要处理LLM可能返回的代码块标记
            if hypotheses_str.strip().startswith("```json"):
                hypotheses_str = hypotheses_str.strip()[7:-3].strip()
            
            single_run_data = json.loads(hypotheses_str)
            
            single_run_data['id'] = i + 1
            
            results_list.append(single_run_data)
            print(f"第 {i + 1} 次调用成功并已记录。")

        except json.JSONDecodeError as e:
            # 如果某一次调用失败，打印错误信息并跳过，继续下一次调用
            print(f"第 {i + 1} 次调用时解析JSON失败: {e}")
            print("原始字符串:", hypotheses_str)
        except Exception as e:
            print(f"第 {i + 1} 次调用时发生未知错误: {e}")
            
    return results_list

if __name__ == '__main__':
    try:
        client = OpenAI(
            base_url="https://api.anglergap.org/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
           
        )
               
        # 调用函数执行10次，并获取包含所有结果的列表
        all_hypotheses_data = chat_llm(client, num_runs=5)
    except Exception as e:
        print(f"\n程序发生严重错误: {e}")
    
    finally:
        if all_hypotheses_data:
            output_filename = "exp/outcome/mid_glm_output.json"
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(all_hypotheses_data, f, indent=4, ensure_ascii=False)
            
            print(f"\n所有 {len(all_hypotheses_data)} 次运行的结果已成功保存到文件: {output_filename}")
        else:
            print("\n没有成功获取到任何结果，不生成文件。")
    

