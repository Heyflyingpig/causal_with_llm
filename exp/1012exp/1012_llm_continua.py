from openai import OpenAI
import os
import json
import re
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def load_jobs_dataset():
    """
    Load Jobs dataset from .dta file and extract relevant variables.
    Returns DataFrame with treatment, outcome, and covariates.
    """
    df = pd.read_stata('oringnal_data/bnlearn/jobs/nsw.dta')
    
    # Select relevant columns based on paper
    # Treatment: treat, Outcome: re78, Covariates: age, education, black, hispanic, married, nodegree, re75
    selected_cols = ['treat', 're78', 'age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're75']
    df_jobs = df[selected_cols].copy()
    
    print(f"Loaded Jobs dataset: {df_jobs.shape[0]} samples, {df_jobs.shape[1]} variables")
    print(f"Variables: {list(df_jobs.columns)}")
    
    return df_jobs


def get_prefix_prompt():
    """
    Generate prefix prompt following Appendix E.1 from the paper.
    This provides dataset introduction and variable descriptions.
    """
    prefix = """Brief introduction of the Jobs dataset:
The Jobs dataset is widely used in causal inference research for evaluating the performance of treatment effect estimation methods. It is constructed by combining experimental and observational data from the National Supported Work (NSW) demonstration and comparison group data (e.g., PSID or CPS).

This observational dataset contains:
(1) Treatment - Participation in Job Training: T ∈ {0,1} indicating whether the individual participated or did not participate in the job training program.
(2) Outcome - Real Earnings in 1978 (re78): The individual's earnings observed after the treatment decision. Continuous value representing employment outcome.
(3) Confounders - age, education, black, hispanic, married, nodegree, re75: Features affecting both the treatment and the outcome, such as age, education level, race (black, hispanic), marital status (married), educational attainment (nodegree = no high school degree), and prior earnings in 1975 (re75)."""
    
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


def generate_parameters(confounder_name: str, distribution_type: str, df_jobs: pd.DataFrame, client: OpenAI):
    """
    P_param(x_i, t_i, y_i): Generate distribution parameters for each individual.
    Following Appendix E.4 from the paper.
    """
    prefix_prompt = get_prefix_prompt()
    
    # Convert first 50 samples to list of dicts for the prompt (to manage token limits)
    sample_size = min(500, len(df_jobs))
    df_sample = df_jobs.head(sample_size)
    data_list = df_sample.to_dict(orient='records')
    data_str = json.dumps(data_list, ensure_ascii=False, indent=2)
    
    # Determine parameter description based on distribution type
    if "normal" in distribution_type.lower() or "gaussian" in distribution_type.lower():
        param_desc = "mean and standard deviation (std)"
        example_params = '{"mean": 0.5, "std": 0.2}'
    elif "bernoulli" in distribution_type.lower():
        param_desc = "probability p (between 0 and 1)"
        example_params = '{"p": 0.7}'
    elif "uniform" in distribution_type.lower():
        param_desc = "lower bound (low) and upper bound (high)"
        example_params = '{"low": 0, "high": 1}'
    else:
        param_desc = "appropriate distribution parameters"
        example_params = '{"param1": value1, "param2": value2}'
    
    prompt = f"""{prefix_prompt}

The values of existing confounders, treatments, and outcomes are given by:
{data_str}

For the confounder '{confounder_name}', which follows a {distribution_type} distribution, please specify {param_desc} for each individual from which we can sample the confounder value.

Base your parameter estimates on:
- The individual's observed features (age, education, race, marital status, prior earnings)
- The treatment assignment (treat)
- The outcome value (re78)
- Your world knowledge about how '{confounder_name}' relates to these variables

You must output in strict JSON format as a list. Each object in the list should contain:
- "id": The index of the individual (integer, 0-based)
- "parameters": The distribution parameters (object with parameter names and values)

Output format example:
```json
[
  {{"id": 0, "parameters": {example_params}}},
  {{"id": 1, "parameters": {example_params}}},
  ...
]
```"""

    response = client.chat.completions.create(
        model="glm-4.5",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    result = response.choices[0].message.content
    return result, sample_size


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
    Main function implementing ProCI confounder generation for Jobs dataset.
    Steps: Variable Generation -> Distribution Identification -> Parameter Inference
    """
    print("="*60)
    print("ProCI Framework - Jobs Dataset Confounder Generation")
    print("="*60)
    
    # Initialize OpenAI client
    client = OpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=os.getenv("OPENAI_API_KEY"),       
    )
    
    # Step 1: Load Jobs dataset
    print("\n[Step 1/4] Loading Jobs dataset...")
    df_jobs = load_jobs_dataset()
    
    # Step 2: Generate confounder variable
    print("\n[Step 2/4] Generating confounder variable...")
    try:
        var_output = get_confounder_variable(client)
        var_data = parse_llm_json(var_output)
        
        confounder_name = var_data['confounder_name']
        confounder_explanation = var_data['explanation']
        
        print(f"✓ Generated confounder: {confounder_name}")
        print(f"  Explanation: {confounder_explanation}")
        
        # Save variable generation result
        with open('outcome/1012_outcome/var_glm_output_test.json', 'w', encoding='utf-8') as f:
            json.dump(var_data, f, indent=4, ensure_ascii=False)
        print("✓ Saved to outcome/1012_outcome/var_glm_output_test.json")
        
    except Exception as e:
        print(f"✗ Error in variable generation: {e}")
        print(f"Raw output: {var_output}")
        return
    
    # Step 3: Identify distribution type
    print("\n[Step 3/4] Identifying distribution type...")
    try:
        dist_output = get_distribution_type(confounder_name, confounder_explanation, client)
        dist_data = parse_llm_json(dist_output)
        
        distribution_type = dist_data['distribution_type']
        value_description = dist_data.get('value_description', '')
        
        print(f"✓ Distribution type: {distribution_type}")
        print(f"  Value description: {value_description}")
        
    except Exception as e:
        print(f"✗ Error in distribution identification: {e}")
        print(f"Raw output: {dist_output}")
        return
    
    # Step 4: Generate parameters for each individual
    print("\n[Step 4/4] Generating distribution parameters for each individual...")
    try:
        params_output, sample_size = generate_parameters(
            confounder_name, distribution_type, df_jobs, client
        )
        params_data = parse_llm_json(params_output)
        
        print(f"✓ Generated parameters for {len(params_data)} individuals")
        
        # Combine all information into final data structure
        final_data = {
            "confounder_name": confounder_name,
            "confounder_explanation": confounder_explanation,
            "distribution_type": distribution_type,
            "value_description": value_description,
            "sample_size": sample_size,
            "data": []
        }
        
        # Merge original data with generated parameters
        for i, param_obj in enumerate(params_data):
            if i >= sample_size:
                break
            
            ## 每一行的每一个col的键值对应
            record = df_jobs.iloc[i].to_dict()
            record[f'{confounder_name}_distribution_type'] = distribution_type
            record[confounder_name] = param_obj['parameters']
            record['id'] = i
            
            final_data['data'].append(record)
        
        # Save final data with parameters
        with open('outcome/1012_outcome/data_glm_data_test.json', 'w', encoding='utf-8') as f:
            json.dump([final_data], f, indent=4, ensure_ascii=False)
        
        print(f"✓ Saved to outcome/1012_outcome/data_glm_data_test.json")
        
        # Preview first few records
        print("\n" + "="*60)
        print("Preview of generated data (first 3 records):")
        print("="*60)
        for i in range(min(3, len(final_data['data']))):
            record = final_data['data'][i]
            print(f"\nRecord {i}:")
            print(f"  Age: {record['age']}, Education: {record['education']}, Treatment: {record['treat']}, Outcome: {record['re78']}")
            print(f"  {confounder_name} parameters: {record[confounder_name]}")
        
    except Exception as e:
        print(f"✗ Error in parameter generation: {e}")
        print(f"Raw output: {params_output}")
        return
    
    print("\n" + "="*60)
    print("✓ ProCI confounder generation completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run 1012_final_sampler.py to sample values from distributions")
    print("2. Run 1012_analyze_llm_data.py to analyze the final dataset")


if __name__ == '__main__':
    main()
