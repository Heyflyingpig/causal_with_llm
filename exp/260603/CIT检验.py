import pandas as pd
import numpy as np
from causallearn.utils.cit import CIT


df_1 = pd.read_csv("jobs_with_counterfactuals.csv")


def add_jitter(x, noise=1e-4):
    return x + np.random.normal(0, noise, size=x.shape)

df_1["Y_cf0"] = add_jitter(df_1["Y_cf0"])
df_1["Y_cf1"] = add_jitter(df_1["Y_cf1"])


full_data_1 = np.hstack([
    df_1["T"].values.reshape(-1, 1),
    df_1["Y_cf0"].values.reshape(-1, 1),
    df_1["Y_cf1"].values.reshape(-1, 1),
    df_1[["age", "education", "black", "hispanic", "married", "re75", "nodegree",'loss_time','legal_issue'
]].values,
])



def create_kci_for_binary(data):

    return CIT(
        data=data,
        method="kci",
        kernelX="Gaussian",
        kernelY="Gaussian",
        kernelZ="Gaussian",
        width_x=0.1,
        width_y=0.1,
        width_z=0.5,
        reg=1e-3,
        num_eig=50
    )



kci_test_1 = create_kci_for_binary(full_data_1)



def kci_cond_test(cit_obj, x_col, y_cols, z_cols):

    p_values = []

    for y_col in y_cols:

        if len(z_cols) == 0:
            p = cit_obj(x_col, y_col, [])
        else:
            p = cit_obj(x_col, y_col, z_cols)
        p_values.append(p)
        

    return np.mean(p_values), np.min(p_values)



p_mean_baseline, p_min_baseline = kci_cond_test(
    kci_test_1,
    0,  # T的列索引
    [1, 2],  # Y0和Y1的列索引
    [3, 4, 5, 6, 7, 8,9]  # X的列索引
)
print(f"平均p值: {p_mean_baseline:.4f}")


