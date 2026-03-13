"""
DTA to CSV Converter Script
将 Stata .dta 文件转换为 CSV 格式
"""

import pandas as pd

# ============ 在这里修改文件路径 ============
INPUT_FILE = "oringnal_data/bnlearn/jobs/nsw.dta"      # 输入的 .dta 文件路径
OUTPUT_FILE = "oringnal_data/bnlearn/jobs/nsw.csv"     # 输出的 .csv 文件路径
# ==========================================

if __name__ == '__main__':
    # 读取 .dta 文件
    print(f"正在读取: {INPUT_FILE}")
    df = pd.read_stata(INPUT_FILE)
    
    # 显示数据基本信息
    print(f"数据形状: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"列名: {list(df.columns)}")
    
    # 保存为 CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"已保存到: {OUTPUT_FILE}")

