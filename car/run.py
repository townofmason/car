import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def run_python_file(filename):
    full_path = os.path.join(os.getcwd(), filename)
    with open(full_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, globals())  # 使用 globals() 传递当前的全局环境

# 按顺序运行三个Python文件，并在每个文件运行后等待用户按 Enter
print("正在运行 '统计性描述.py'...")
run_python_file("统计性描述.py")
input("\n按 Enter 键继续运行下一个文件...")  # 等待用户按 Enter

print("正在运行 '方差分析.py'...")
run_python_file("方差分析.py")
input("\n按 Enter 键继续运行下一个文件...")  # 等待用户按 Enter

print("正在运行 '回归分析.py'...")
run_python_file("回归分析.py")
input("\n按 Enter 键结束程序...")  # 等待用户按 Enter
