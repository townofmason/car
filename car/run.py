import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys
import subprocess  # 导入 subprocess 模块


def install_requirements():
    """安装 requirement.txt 中的依赖库"""
    requirements_file = os.path.join(os.getcwd(), "requirement.txt")  # 确保找到 requirements.txt
    if os.path.exists(requirements_file):
        print(f"正在安装 {requirements_file} 中的依赖库...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("依赖库安装成功！")
        except subprocess.CalledProcessError as e:
            print(f"依赖库安装失败，请手动运行以下命令安装：")
            print(f"pip install -r {requirements_file}")
            sys.exit(1)  # 退出程序
    else:
        print(f"未找到 {requirements_file} 文件，请确保文件存在。")
        sys.exit(1)  # 退出程序

def run_python_file(filename):
    """运行指定的 Python 文件"""
    full_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(code, globals())  # 使用 globals() 传递当前的全局环境
    else:
        print(f"未找到 {filename} 文件，请确保文件存在。")
        sys.exit(1)  # 退出程序

# 安装依赖库
install_requirements()

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
