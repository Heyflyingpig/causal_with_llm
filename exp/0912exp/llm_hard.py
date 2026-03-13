from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

def get_confounder_hypotheses(*variables: str, causal_graph: str, client: OpenAI):

    if len(variables) < 2:
        raise ValueError("请至少提供两个变量。")
    if len(variables) == 2:
        variables_str = f'“{variables[0]}”与“{variables[1]}”'
    else:
        variables_str = '、'.join([f'“{v}”' for v in variables[:]])
    
    prompt = f"""
    你是一位因果推断领域的专家，并且拥有丰富的医学知识。你的任务是根据一个已知的、但不完整的因果模型，帮助我找到一个潜在的、未被观测到的混淆变量。

    **背景**：
    我们正在研究一个关于呼吸系统疾病的因果模型。其已知的结构以贝叶斯交换格式（.bif）的部分内容描述如下
    {causal_graph}

    **任务**：
    1. 在我们的数据分析中，我们发现 {variables_str} 之间存在很强的统计相关性。
    然而，在上面给出的因果结构中，{variables_str}之间没有任何因果路径可以解释这种关联。它们是两个独立的根节点。
    请根据你的专业医学知识进行推断，什么是最有可能的、同时导致 {variables_str} 的 **共同原因（混淆变量）**？请给出你的答案并简要解释为什么。

    **要求**: 
    1. 对你提出的因果叙事进行排名，并说明理由。
    2. 必须以严格的JSON格式输出，不要包含任何JSON格式之外的解释性文字。输出的JSON对象应包含以下键：
       - "variables": 一个包含输入变量的列表。      
       - "confounder_hypotheses": 一个对象列表，每个对象代表一个混淆变量假说，并包含以下键：       
         - "rank": 排名 (整数)，排序从依据是以你认为的可能性百分比从高到低排序。
         - "confounder_variables": 一个包含所有你认为叙事中所有潜在原因的列表，每一个对象都表示在你生成故事中的潜在原因，叙事原因的多少依据你所生成的叙事数量而定。
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
        
        causal_graph = """
        ```bif
        variable asia { type discrete [ 2 ] { yes, no }; }
        variable tub { type discrete [ 2 ] { yes, no }; }
        variable lung { type discrete [ 2 ] { yes, no }; }
        variable bronc { type discrete [ 2 ] { yes, no }; }
        variable either { type discrete [ 2 ] { yes, no }; }
        variable xray { type discrete [ 2 ] { yes, no }; }
        variable dysp { type discrete [ 2 ] { yes, no }; }

        probability ( tub | asia ) { ... }
        probability ( either | lung, tub ) { ... }
        probability ( xray | either ) { ... }
        probability ( dysp | bronc, either ) { ... }
        ```
     
        """
        
        
        observed_variables = ["肺癌", "支气管炎"]

        # 使用 f-string 来格式化字符串，让输出更清晰
        print(f"正在进行第 {i + 1}/{num_runs} 次LLM调用...")
        
        hypotheses_str = get_confounder_hypotheses(*observed_variables,causal_graph=causal_graph, client=client)
        
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
            output_filename = "exp/outcome/hd_glm_output.json"
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(all_hypotheses_data, f, indent=4, ensure_ascii=False)
            
            print(f"\n所有 {len(all_hypotheses_data)} 次运行的结果已成功保存到文件: {output_filename}")
        else:
            print("\n没有成功获取到任何结果，不生成文件。")
    

