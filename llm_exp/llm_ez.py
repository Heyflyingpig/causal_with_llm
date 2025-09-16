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
    You are an expert in causal inference in the medical field.

    **Background**: In a medical study, we analyzed data from a group of patients and are given two variables: {variables_str}.

    **Task**:
    1. First, you need to determine if there is a direct association between these two variables. If you believe no confounding variables exist, please output "No". If confounding variables exist, please output "Yes" and provide a list of the confounders as requested.
    2. If confounding variables exist, please propose the most likely `3 to 5` **hidden, common causes** that could simultaneously lead to the phenomena observed in {variables_str}. In other words, what underlying diseases or physical conditions could reasonably explain the observed association? Please list the possibilities and briefly explain your reasoning.
    3. For each confounding variable you propose, you need to output its **prior probability**. This probability should be a specific numerical value (e.g., 0.05), representing the probability of the confounder being "Yes" in the general population.
    4. For each confounding variable you propose, provide a **complete conditional probability table (CPT)** for the observed variables (e.g., "X-ray Result", "Dyspnea Symptom") under the influence of that confounder. This means you need to provide the probabilities of the different states of the observed variables when the confounder is "Yes" and when it is "No".

    **Requirements**:
    1. First, you must determine if a confounding variable exists. If so, output "Yes"; otherwise, output "No".
    2. If confounding variables exist, please rank your proposed confounders and output their prior probabilities with reasoning as requested.
    3. If no confounding variables exist, you do not need to complete the subsequent steps.
    4. The output must be in strict JSON format. Do not include any explanatory text outside the JSON structure. The output JSON object should contain the following keys:
       - "variables": A list containing the input variables.
       - "is_confounder": A boolean value indicating whether a confounding variable exists.
       - "confounder_variables": A list containing all the potential causes you believe exist.
       - "Probability": A list of objects, where each object represents the prior probability of a potential cause. Each object should contain "confounder" (str) and "probability" (float) as keys.
       - "confounder_hypotheses": A list of objects, where each object represents a confounder hypothesis and includes the following keys:
         - "rank": The rank (integer), ordered from highest to lowest based on what you believe to be the percentage of likelihood.
         - "confounder": The name of the confounding variable (string).
         - "reasoning": A brief explanation of the reasoning (string).
         - "causal_graph": A string describing the causal graph, e.g., "Confounder -> Observed Variable 1, Confounder -> Observed Variable 2".
       - "conditional_probabilities": A list of objects, where each object represents the conditional probabilities under a confounder and includes the following keys:
         - "confounder": The name of the confounding variable (string).
         - "probabilities": A list of objects, where each object contains the conditional probability table (CPT) for an observed variable.
           - "observed_variable": The name of the observed variable (string).
           - "cpt": An object that clearly shows the probabilities under two conditions, including the following keys:
             - "when_confounder_true": An object representing the probabilities of the observed variable's states when the confounder is "Yes".
             - "when_confounder_false": An object representing the probabilities of the observed variable's states when the confounder is "No".


    **Example Output Format (all data is for illustration purposes)**:
    ```json
    {{
      "variables": ["Variable A", "Variable B"],
      "is_confounder": true,
      "confounder_variables": ["Potential Cause 1", "Potential Cause 2"],
      "Probability": [
        {{
          "confounder": "Potential Cause 1",
          "probability": 0.15
        }},
        {{
          "confounder": "Potential Cause 2",
          "probability": 0.08
        }}
      ],
      "confounder_hypotheses": [
        {{
          "rank": 1,
          "confounder": "Potential Cause 1",
          "reasoning": "This is the most likely cause because...",
          "causal_graph": "Potential Cause 1 -> Variable A; Potential Cause 1 -> Variable B"
        }},
        {{
          "rank": 2,
          "confounder": "Potential Cause 2",
          "reasoning": "This cause is less likely because...",
          "causal_graph": "Potential Cause 2 -> Variable A; Potential Cause 2 -> Variable B"
        }}
      ],
      "conditional_probabilities": [
        {{
            "confounder": "Potential Cause 1",
            "probabilities": [
                {{
                    "observed_variable": "Variable A",
                    "cpt": {{
                        "when_confounder_true": {{ "State 1": 0.8, "State 2": 0.2 }},
                        "when_confounder_false": {{ "State 1": 0.05, "State 2": 0.95 }}
                    }}
                }},
                {{
                    "observed_variable": "Variable B",
                    "cpt": {{
                        "when_confounder_true": {{ "State 1": 0.7, "State 2": 0.3 }},
                        "when_confounder_false": {{ "State 1": 0.02, "State 2": 0.98 }}
                    }}
                }}
            ]
        }}
      ]
    }}
    ```
    """
    response = client.chat.completions.create(
        model="BigModel/GLM-4.5 [free]",
        messages=[
            {"role": "user", "content": prompt}
        ]
        
    )

    llm_hypotheses = response.choices[0].message.content
    return llm_hypotheses

def chat_llm(client, num_runs, results_list):
    

    for i in range(num_runs):
        
        observed_variables = ["X-ray Result", "Dyspnea Symptom"] # Translating variables to English
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
            base_url="https://demo.awa1.fun/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            default_headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}
        )
        
        
        chat_llm(client, num_runs=5, results_list=all_hypotheses_data)
        
    except Exception as e:
        print(f"\n程序发生严重错误: {e}")

    finally:
        # finally块确保无论是否发生异常，都会执行这部分代码
        if all_hypotheses_data:
            output_filename = "llm_exp/914_outcome/ez_glm_output_en.json"
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(all_hypotheses_data, f, indent=4, ensure_ascii=False)
            
            print(f"\n所有 {len(all_hypotheses_data)} 次运行的结果已成功保存到文件: {output_filename}")
        else:
            print("\n没有成功获取到任何结果，不生成文件。")

