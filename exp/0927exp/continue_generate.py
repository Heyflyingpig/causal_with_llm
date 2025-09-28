from causallearn.utils.Dataset import load_dataset
from pandas import DataFrame

data, true_dag = load_dataset('sachs')

# data 是一个 numpy 数组
print("数据(numpy array):")
print(data[:5, :])

print("\n数据的维度 (样本数, 变量数):")
print(data.shape)


# true_dag 现在直接是一个包含真实因果图边的列表
# 我们可以直接打印这个列表
print("\n真实的因果样本:")
print(true_dag)

output_dir = "oringnal_data/bnlearn/Sachs"
output_filename = f"{output_dir}/sachs_dataset.csv"

data = DataFrame(data)
data.columns = true_dag
data.to_csv(output_filename, index=False)