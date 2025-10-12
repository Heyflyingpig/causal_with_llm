# ProCI Jobs Dataset Replication

This directory contains the implementation of the ProCI (Progressive Confounder Imputation) framework applied to the Jobs dataset, following the methodology from the paper "Mitigating hidden confounding by progressive confounder imputation via large language models".

## Overview

The implementation follows the paper's three-stage confounder generation process:
1. **Variable Generation** (P_var): LLM proposes a hidden confounder based on domain knowledge
2. **Distribution Identification** (P_dist): LLM determines the distribution type for the confounder
3. **Parameter Inference** (P_param): LLM generates personalized distribution parameters for each individual

## Files

- `1012_llm_continua.py` - Main script for LLM-based confounder generation
- `1012_final_sampler.py` - Samples actual values from the generated distributions
- `1012_analyze_llm_data.py` - Analyzes the final dataset with generated confounders

## Dataset

**Jobs Dataset** (`oringnal_data/bnlearn/jobs/nsw.dta`)
- Source: National Supported Work (NSW) demonstration
- Treatment: Job training participation (treat: 0/1)
- Outcome: Real earnings in 1978 (re78)
- Covariates: age, education, black, hispanic, married, nodegree, re75

## Usage

### Step 1: Generate Confounder with LLM

```bash
python exp/1012exp/1012_llm_continua.py
```

This script:
- Loads the Jobs dataset from .dta format
- Uses LLM to generate one hidden confounder variable
- Identifies the appropriate distribution type
- Generates distribution parameters for each individual (first 50 samples)
- Outputs:
  - `outcome/1012_outcome/var_glm_output_test.json` - Confounder metadata
  - `outcome/1012_outcome/data_glm_data_test.json` - Distribution parameters

### Step 2: Sample from Distributions

```bash
python exp/1012exp/1012_final_sampler.py
```

This script:
- Reads the distribution parameters from Step 1
- Samples actual values from each individual's personalized distribution
- Replaces parameter dictionaries with sampled values
- Output:
  - `outcome/1012_outcome/final_data.json` - Final dataset with sampled confounder values

### Step 3: Analyze Results

```bash
python exp/1012exp/1012_analyze_llm_data.py
```

This script:
- Loads the final dataset
- Performs descriptive statistics
- Analyzes correlations between confounder, treatment, and outcome
- Checks data quality
- Output:
  - `outcome/1012_outcome/jobs_with_confounder.csv` - CSV format for further analysis

## Implementation Details

### Prompt Templates

Following Appendix E of the paper, we use three prompt templates:

1. **Prefix Prompt** (E.1): Dataset introduction with treatment, outcome, and covariate descriptions
2. **Variable Generation** (E.2): Asks LLM to propose one confounder affecting both treatment and outcome
3. **Distribution Inference** (E.3): Identifies distribution type (Normal/Bernoulli/Uniform/Categorical)
4. **Parameter Estimation** (E.4): Generates personalized parameters based on individual features

### Supported Distributions

- **Normal/Gaussian**: Parameters: mean, std
- **Bernoulli**: Parameters: p (probability)
- **Uniform**: Parameters: low, high
- **Exponential**: Parameters: lambda (rate parameter)
- **Categorical**: Parameters: categories, probabilities

## Example Output

```
ProCI Framework - Jobs Dataset Confounder Generation
================================================================

[Step 1/4] Loading Jobs dataset...
Loaded Jobs dataset: 722 samples, 9 variables

[Step 2/4] Generating confounder variable...
✓ Generated confounder: Transportation Access
  Explanation: Access to reliable transportation can influence both 
  participation in job training and employment outcomes...

[Step 3/4] Identifying distribution type...
✓ Distribution type: Bernoulli
  Value description: 0 = no reliable transportation, 1 = has transportation

[Step 4/4] Generating distribution parameters for each individual...
✓ Generated parameters for 50 individuals
✓ Saved to outcome/1012_outcome/data_glm_data_test.json
```

## Notes

- This implementation generates **1 confounder** as a demonstration
- The full ProCI framework includes iterative generation with unconfoundedness testing
- Counterfactual outcome imputation (Section 3.4) is not included in this demo
- Uses GLM-4.5-air and GLM-4.5 models via OpenAI-compatible API

## Requirements

```python
pandas
numpy
python-dotenv
openai
pyreadstat  # for reading .dta files
```

## References

Paper: "Mitigating hidden confounding by progressive confounder imputation via large language models"
- Appendix E: Prompt Templates
- Section 3.3: Confounder Imputation via Prompting LLMs

