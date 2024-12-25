import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import os

from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal, boxcox
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.multicomp import MultiComparison

rcParams['font.family'] = ('SimHei')

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接文件路径
file_path = os.path.join(script_dir,  '数据.xlsx')
from termcolor import colored
print(colored("**************************step1************************************\n", attrs=["bold"]))

# 打印大字（粗体）
print(colored("进行统计性描述，绘制箱线图，联合分布图，QQ图，进行多重比较检验\n\n", attrs=["bold"]))


# 读取文件
df = pd.read_excel(file_path)

new_data,lamb = boxcox(df['首月零售量'])
df['首月零售量']=new_data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)
df.head()
def check_df(dataframe, head = 5):
    print("############## Shape ##############")
    print(dataframe.shape)
    print("############## Types ##############")
    print(dataframe.dtypes)
    print("############## Head ##############")
    print(dataframe.head(head))
    print("############## Tail ##############")
    print(dataframe.tail(head))
    print("############## NA ##############")
    print(dataframe.isnull().sum())
    print("############## Quantiles ##############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
print(colored("检查数据集\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")

check_df(df)




sns.boxplot(x="车型", y="首月零售量", data=df)
plt.xticks(rotation=45)
plt.show()

sns.boxplot(x="母品牌", y="首月零售量", data=df)
plt.xticks(rotation=45)
plt.show()

sns.jointplot(x = "尺寸", y = "首月零售量", data = df, kind = "reg")
plt.show()

sns.jointplot(x = "价格", y = "首月零售量", data = df, kind = "reg")
plt.show()

fig , axs = plt.subplots(1,3,figsize=(15,5))
qqplot(np.array(df.loc[(df["车型"] == '轿车'), "首月零售量"]), line="s", ax=axs[0])
qqplot(np.array(df.loc[(df["车型"] == 'SUV'), "首月零售量"]), line="s", ax=axs[1])
qqplot(np.array(df.loc[(df["车型"] == 'MPV'), "首月零售量"]), line="s", ax=axs[2])
axs[0].set_title("轿车")
axs[1].set_title("SUV")
axs[2].set_title("MPV")
plt.show()


print(colored("************多重比较检验\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")
comparison = MultiComparison(df["首月零售量"], df["车型"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

comparison = MultiComparison(df["首月零售量"], df["母品牌"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())