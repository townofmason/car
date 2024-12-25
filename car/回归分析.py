import statsmodels.api as sm
import pandas as pd
import os

from termcolor import colored


print(colored("*********step3*********", attrs=["bold"]))

print(colored("进行回归分析", attrs=["bold"]))





script_dir = os.path.dirname(os.path.abspath(__file__))


file_path = os.path.join(script_dir, '数据.xlsx')




df = pd.read_excel(file_path, usecols=range(5))

##*****************数据预处理
print(colored("\n************将品牌与车型转换为虚拟变量*************",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")


df_dummies = pd.get_dummies(df, columns=['母品牌', '车型'], drop_first=True)


y = df_dummies['首月零售量']  # 因变量
X = df_dummies.drop(columns=['首月零售量'])  # 自变量


X = sm.add_constant(X)


print("\n处理后的自变量（X）：")
print(X.head())

print("\n因变量（y）：")
print(y.head())

##*****************共线性检验
print(colored("\n************进行共线性检验*****************\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

X = X.astype(int)  
X_without_const = X.drop(columns='const')  # 去掉常数项
vif_data = pd.DataFrame()
vif_data["Variable"] = X_without_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_without_const.values, i) for i in range(X_without_const.shape[1])]
print(vif_data)

high_vif = vif_data[vif_data["VIF"] >= 10] # 筛选出 VIF 大于等于 10 的变量
if high_vif.empty:
    print("\n所有 VIF 值都小于 10，模型中没有严重的共线性问题。")
else:
    print("\n可能存在共线性问题的变量：")
    print(high_vif)



print("\n所以去除尺寸\n")



X_1= X.drop(columns=['尺寸'])
X_without_const = X_1.drop(columns='const')  # 去掉常数项

vif_data = pd.DataFrame()
vif_data["Variable"] = X_without_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_without_const.values, i) for i in range(X_without_const.shape[1])]
print(vif_data)

high_vif = vif_data[vif_data["VIF"] >= 10] # 筛选出 VIF 大于等于 10 的变量
if high_vif.empty:
    print("\n去除尺寸后，所有 VIF 值都小于 10，模型中没有严重的共线性问题。")
else:
    print("\nVIF 值大于等于 10 的变量可能存在共线性问题：")
    print(high_vif)





##*****************无交互项的线性回归
print(colored("\n************进行无交互项的线性回归*****************\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")

# OLS
model = sm.OLS(y, X).fit()
import matplotlib.pyplot as plt
import numpy as np



plt.rcParams['font.family'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

residuals = model.resid
standardized_residuals = residuals / np.std(residuals)
plt.scatter(model.fittedvalues, standardized_residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("取对数前线性回归的残差图")
plt.xlabel("Fitted Values")
plt.ylabel("Standardized Residuals")
plt.show()
##输出R^2和F
r_squared = model.rsquared
adj_r_squared = model.rsquared_adj
print(f"\nR²: {r_squared:.4f}")
print(f"调整后的 R²: {adj_r_squared:.4f}")

print(f"f_pvalue: {model.f_pvalue:.4f}")





print("\n从残差图和R^2可以看出，回归效果并不好，注意到残差随着拟合值增大而变得更加分散，表明可能存在异方差性，考虑进行对数变换")






##********************进行对数变换
print(colored("\n************进行对数变换******************\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")

import numpy as np

y_log = np.log(y)
X_log = X.copy()
X_log['尺寸'] = np.log(X_log['尺寸'])
X_log['价格'] = np.log(X_log['价格'])
X_log = sm.add_constant(X_log)  
model_log = sm.OLS(y_log, X_log).fit()
##输出R^2和F
r_squared = model_log.rsquared
adj_r_squared = model_log.rsquared_adj
print(f"\nR²: {r_squared:.4f}")
print(f"调整后的 R²: {adj_r_squared:.4f}")

print(f"f_pvalue: {model.f_pvalue:.4f}")

residuals_log = model_log.resid
standardized_residuals_log = residuals_log / np.std(residuals_log)
fitted_values_log = model_log.fittedvalues

plt.scatter(fitted_values_log, standardized_residuals_log)
plt.axhline(0, color='red', linestyle='--')
plt.title("对数变换后线性回归的残差图")
plt.xlabel("Fitted Values")
plt.ylabel("Standardized Residuals")
plt.show()
print("\n从R^2和残差图可以看出，对数变换后回归效果大幅提升。")




##*****************引入交互项
print(colored("\n************引入交互项，观察R^2变化*****************\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


df_dummies = pd.get_dummies(df, columns=['母品牌', '车型'], drop_first=True)


y = df_dummies['首月零售量']  # 因变量
X = df_dummies.drop(columns=['首月零售量'])  # 自变量


if '价格' in X.columns and '尺寸' in X.columns:
    X['log_价格'] = np.log(X['价格'])
    X['log_尺寸'] = np.log(X['尺寸'])
    X = X.drop(columns=['价格', '尺寸'])  # 移除原始的价格和尺寸列
else:
    raise ValueError("数据中缺少必要的'价格'或'尺寸'列")


for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)


X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()  
y_log = np.log(y[X.index])  


X_with_size = X.copy()


base_X = X.drop(columns='log_尺寸').copy()
base_X = sm.add_constant(base_X)
base_model = sm.OLS(y_log, base_X).fit()


r_squared_values = [base_model.rsquared]
adj_r_squared_values = [base_model.rsquared_adj]


def add_interaction_terms(X, interaction_type):
    X_with_interactions = X.copy()
    
    if interaction_type == 'brand_price':
        for col in X.columns:
            if col.startswith('母品牌_'):
                interaction_col = f'{col}_log_价格'
                X_with_interactions[interaction_col] = X[col] * X['log_价格']
                
    elif interaction_type == 'brand_size':
        for col in X.columns:
            if col.startswith('母品牌_'):
                interaction_col = f'{col}_log_尺寸'
                X_with_interactions[interaction_col] = X[col] * X['log_尺寸']
                
    elif interaction_type == 'size_price':
        interaction_col = 'log_尺寸_log_价格'
        X_with_interactions[interaction_col] = X['log_尺寸'] * X['log_价格']
        
    elif interaction_type == 'price_model':
        for model_col in [c for c in X.columns if c.startswith('车型_')]:
            interaction_col = f'log_价格_{model_col}'
            X_with_interactions[interaction_col] = X['log_价格'] * X[model_col]
            
    elif interaction_type == 'size_model':
        for model_col in [c for c in X.columns if c.startswith('车型_')]:
            interaction_col = f'log_尺寸_{model_col}'
            X_with_interactions[interaction_col] = X['log_尺寸'] * X[model_col]
            
    elif interaction_type == 'brand_model':
        for brand_col in [c for c in X.columns if c.startswith('母品牌_')]:
            for model_col in [c for c in X.columns if c.startswith('车型_')]:
                interaction_col = f'{brand_col}_{model_col}'
                X_with_interactions[interaction_col] = X[brand_col] * X[model_col]
                
    return  X_with_interactions.drop(columns=['log_尺寸'], errors='ignore')  # 确保最终模型中没有 log_尺寸


interaction_steps = [
    ('brand_price', 'With Brand & Price Interaction'),
    ('brand_size', 'With Brand-Size Interactions'),
    ('size_price', 'With Size-Price Interactions'),
    ('price_model', 'With Price-Model Interactions'),
    ('size_model', 'With Size-Model Interactions'),
    ('brand_model', 'With Brand-Model Interactions')
]


r_squared_values = [base_model.rsquared]
adj_r_squared_values = [base_model.rsquared_adj]


results_df = pd.DataFrame(columns=['Step', 'R-squared', 'Adjusted R-squared'])


for interaction_type, label in interaction_steps:

    X_with_interaction = X_with_size.copy()

    
    X_with_interaction = add_interaction_terms(X_with_interaction, interaction_type)
    
   
    X_with_interaction = sm.add_constant(X_with_interaction)
    
    
    model_with_interaction = sm.OLS(y_log, X_with_interaction).fit()

    

  
    r_squared_values.append(model_with_interaction.rsquared)
    adj_r_squared_values.append(model_with_interaction.rsquared_adj)

    
    results_df = pd.concat([
        results_df,
        pd.DataFrame([{
            'Step': label,
            'R-squared': model_with_interaction.rsquared,
            'Adjusted R-squared': model_with_interaction.rsquared_adj
        }])
    ], ignore_index=True)




print("\n模型性能对比表：")
print(results_df)


labels = ['Without Interaction'] + [step[1] for step in interaction_steps]
x = np.arange(len(labels))  
width = 0.4 

fig, ax = plt.subplots(figsize=(8, 3))
rects1 = ax.bar(x - width/2, r_squared_values, width, label='R-squared')
rects2 = ax.bar(x + width/2, adj_r_squared_values, width, label='Adjusted R-squared')


ax.set_ylabel('Scores')
ax.set_title('Comparison of R-squared and Adjusted R-squared')
ax.set_xticks(x)

ax.legend()

fig.tight_layout()

plt.show()

print("\n 可以看出，每个交互项引入后都不会使调整后R²值明显减少，所以选择都引入，再根据系数大小和t检验显著性剔除不显著的交互项")



##*****************引入所有交互项
print(colored("\n************引入所有交互项*****************\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")

def add_interaction_terms(X, interaction_types):
    X_with_interactions = X.copy()
    
    if 'brand_price' in interaction_types:
        for col in X.columns:
            if col.startswith('母品牌_'):
                interaction_col = f'{col}_log_价格'
                X_with_interactions[interaction_col] = X[col] * X['log_价格']
                
    if 'brand_size' in interaction_types:
        for col in X.columns:
            if col.startswith('母品牌_'):
                interaction_col = f'{col}_log_尺寸'
                X_with_interactions[interaction_col] = X[col] * X['log_尺寸']
                
    if 'brand_model' in interaction_types:
        for brand_col in [c for c in X.columns if c.startswith('母品牌_')]:
            for model_col in [c for c in X.columns if c.startswith('车型_')]:
                interaction_col = f'{brand_col}_{model_col}'
                X_with_interactions[interaction_col] = X[brand_col] * X[model_col]
                
    if 'price_model' in interaction_types:
        for model_col in [c for c in X.columns if c.startswith('车型_')]:
            interaction_col = f'log_价格_{model_col}'
            X_with_interactions[interaction_col] = X['log_价格'] * X[model_col]
    
    if 'size_price' in interaction_types:
        interaction_col = 'log_尺寸_log_价格'
        X_with_interactions[interaction_col] = X['log_尺寸'] * X['log_价格']
        
    
            
    if 'size_model' in interaction_types:
        for model_col in [c for c in X.columns if c.startswith('车型_')]:
            interaction_col = f'log_尺寸_{model_col}'
            X_with_interactions[interaction_col] = X['log_尺寸'] * X[model_col]
            
    
            
    return X_with_interactions.drop(columns=['log_尺寸'], errors='ignore')  # 确保最终模型中没有 log_尺寸


interaction_types = ['brand_price', 'brand_size', 'brand_model', 'price_model', 'size_price', 'size_model']


X_with_all_interactions = add_interaction_terms(X_with_size, interaction_types)


X_with_all_interactions = sm.add_constant(X_with_all_interactions)



final_model = sm.OLS(y_log, X_with_all_interactions).fit()


print("\n包含所有交互项的模型的详细摘要：")
print(final_model.summary())


print(colored("\n观察到价格-车型的系数低，显著性差，所以去掉价格-车型交互项。",attrs=["bold"]))

# **********************去除价格-车型交互项
print(colored("\n************去除价格-车型交互项*****************\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")

def add_interaction_terms(X, interaction_types):
    X_with_interactions = X.copy()
    
    if 'brand_price' in interaction_types:
        for col in X.columns:
            if col.startswith('母品牌_'):
                interaction_col = f'{col}_log_价格'
                X_with_interactions[interaction_col] = X[col] * X['log_价格']
                
    if 'brand_size' in interaction_types:
        for col in X.columns:
            if col.startswith('母品牌_'):
                interaction_col = f'{col}_log_尺寸'
                X_with_interactions[interaction_col] = X[col] * X['log_尺寸']
                
    if 'brand_model' in interaction_types:
        for brand_col in [c for c in X.columns if c.startswith('母品牌_')]:
            for model_col in [c for c in X.columns if c.startswith('车型_')]:
                interaction_col = f'{brand_col}_{model_col}'
                X_with_interactions[interaction_col] = X[brand_col] * X[model_col]
                
    
    if 'size_price' in interaction_types:
        interaction_col = 'log_尺寸_log_价格'
        X_with_interactions[interaction_col] = X['log_尺寸'] * X['log_价格']
        
    
            
    if 'size_model' in interaction_types:
        for model_col in [c for c in X.columns if c.startswith('车型_')]:
            interaction_col = f'log_尺寸_{model_col}'
            X_with_interactions[interaction_col] = X['log_尺寸'] * X[model_col]
            
    
            
    return X_with_interactions.drop(columns=['log_尺寸'], errors='ignore')  # 确保最终模型中没有 log_尺寸

interaction_types = ['brand_price', 'brand_size', 'brand_model',  'size_price', 'size_model']


X_with_all_interactions = add_interaction_terms(X_with_size, interaction_types)


X_with_all_interactions = sm.add_constant(X_with_all_interactions)

final_model = sm.OLS(y_log, X_with_all_interactions).fit()

print("\n去除价格-车型后模型的详细摘要：")
print(final_model.summary())

print(colored("\n观察到尺寸-suv和尺寸-轿车的系数近似，且显著性低，所以去除尺寸-车型交互项。",attrs=["bold"]))





##*****************去除尺寸-车型交互项
print(colored("\n************去除尺寸-车型交互项*****************\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")

def add_interaction_terms(X, interaction_types):
    X_with_interactions = X.copy()
    
    if 'brand_price' in interaction_types:
        for col in X.columns:
            if col.startswith('母品牌_'):
                interaction_col = f'{col}_log_价格'
                X_with_interactions[interaction_col] = X[col] * X['log_价格']
                
    if 'brand_size' in interaction_types:
        for col in X.columns:
            if col.startswith('母品牌_'):
                interaction_col = f'{col}_log_尺寸'
                X_with_interactions[interaction_col] = X[col] * X['log_尺寸']
                
    if 'brand_model' in interaction_types:
        for brand_col in [c for c in X.columns if c.startswith('母品牌_')]:
            for model_col in [c for c in X.columns if c.startswith('车型_')]:
                interaction_col = f'{brand_col}_{model_col}'
                X_with_interactions[interaction_col] = X[brand_col] * X[model_col]
                
    
    if 'size_price' in interaction_types:
        interaction_col = 'log_尺寸_log_价格'
        X_with_interactions[interaction_col] = X['log_尺寸'] * X['log_价格']
        
    
            
    
            
    
            
    return X_with_interactions.drop(columns=['log_尺寸'], errors='ignore')  # 确保最终模型中没有 log_尺寸


interaction_types = ['brand_price', 'brand_size', 'brand_model',  'size_price', ]


X_with_all_interactions = add_interaction_terms(X_with_size, interaction_types)

X_with_all_interactions = sm.add_constant(X_with_all_interactions)

final_model = sm.OLS(y_log, X_with_all_interactions).fit()

print("\n最终模型的详细摘要：")
print(final_model.summary())


r_squared = final_model.rsquared
adj_r_squared = final_model.rsquared_adj
print(f"\nR²: {r_squared:.4f}")
print(f"调整后的 R²: {adj_r_squared:.4f}")
print(f"f_pvalue: {final_model.f_pvalue:.4f}")


residuals = final_model.resid
fitted_values = final_model.fittedvalues

plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()






