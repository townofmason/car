import pandas as pd
from scipy.stats import f_oneway, boxcox, levene
import os
from termcolor import colored


print(colored("********************step2********************************", attrs=["bold"]))

print(colored("进行方差分析（我们通过幂函数变换数据使数据符合正态分布）", attrs=["bold"]))
print(colored("p1用于检验方差是否相等，p2用于检验均值是否不同\n", attrs=["bold"]))



script_dir = os.path.dirname(os.path.abspath(__file__))


file_path = os.path.join(script_dir, '数据.xlsx')

data = pd.read_excel(file_path)

new_data,lamb=boxcox(data['首月零售量'])#变换数据使其符合正态分布
data['首月零售量']=new_data



##*****************车型的方差分析
data_car=data[data['车型']=='轿车']['首月零售量']
data_SUV=data[data['车型']=='SUV']['首月零售量']
data_MPV=data[data['车型']=='MPV']['首月零售量']
w,p1=levene(data_car,data_SUV,data_MPV)
print(colored("************车型的方差分析****************\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")

f,p2=f_oneway(data_car,data_SUV,data_MPV)
print(f"p1:{p1}")#检验其方差是否相等
print(f"p2:{p2}\n")#检验其方差是否相等

print("p2<0.05 说明车型是一个有影响的因素\n")

##*************尺寸的方差分析
data_size1=data[data['尺寸']<=2000]['首月零售量']
data_size2=data[data['尺寸']>2000][data['尺寸']<=2250]['首月零售量']
data_size3=data[data['尺寸']>2250][data['尺寸']<=2500]['首月零售量']
data_size4=data[data['尺寸']>2500][data['尺寸']<=2750]['首月零售量']
data_size5=data[data['尺寸']>2750][data['尺寸']<=3000]['首月零售量']
data_size6=data[data['尺寸']>3000]['首月零售量']
w,p1=levene(data_size1,data_size2,data_size3,data_size4,data_size5,data_size6)
print(colored("***************尺寸的方差分析**************\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")


f,p2=f_oneway(data_size1,data_size2,data_size3,data_size4,data_size5,data_size6)
print(f"p1:{p1}")#检验其方差是否相等
print(f"p2:{p2}\n")#检验其方差是否相等

print("p2>0.05 说明尺寸不是一个有影响的因素\n")



##******************母品牌的方差分析
print(colored("************品牌的方差分析********************\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")

data_brand1=data[data['母品牌']=='奥迪']['首月零售量']
data_brand2=data[data['母品牌']=='宝马']['首月零售量']
data_brand3=data[data['母品牌']=='北汽集团']['首月零售量']
data_brand4=data[data['母品牌']=='奔驰']['首月零售量']
data_brand5=data[data['母品牌']=='比亚迪']['首月零售量']
data_brand6=data[data['母品牌']=='别克']['首月零售量']
data_brand7=data[data['母品牌']=='大众']['首月零售量']
data_brand8=data[data['母品牌']=='东风汽车']['首月零售量']
data_brand9=data[data['母品牌']=='丰田']['首月零售量']
data_brand10=data[data['母品牌']=='广汽集团']['首月零售量']
data_brand11=data[data['母品牌']=='红旗']['首月零售量']
data_brand12=data[data['母品牌']=='吉利汽车']['首月零售量']
data_brand13=data[data['母品牌']=='江汽集团']['首月零售量']
data_brand14=data[data['母品牌']=='零跑汽车']['首月零售量']
data_brand15=data[data['母品牌']=='哪吒汽车']['首月零售量']
data_brand16=data[data['母品牌']=='奇瑞']['首月零售量']
data_brand17=data[data['母品牌']=='赛力斯']['首月零售量']
data_brand18=data[data['母品牌']=='上汽集团']['首月零售量']
data_brand19=data[data['母品牌']=='特斯拉']['首月零售量']
data_brand20=data[data['母品牌']=='蔚来']['首月零售量']
data_brand21=data[data['母品牌']=='五菱汽车']['首月零售量']
data_brand22=data[data['母品牌']=='小米汽车']['首月零售量']
data_brand23=data[data['母品牌']=='小鹏汽车']['首月零售量']
data_brand24=data[data['母品牌']=='长安汽车']['首月零售量']
data_brand25=data[data['母品牌']=='长城汽车']['首月零售量']
w,p1=levene(data_brand1,data_brand2,data_brand3,data_brand4,data_brand5,data_brand6,data_brand7,data_brand8,data_brand9,data_brand10,data_brand11,data_brand12,data_brand13,data_brand14,data_brand15,data_brand16,data_brand17,data_brand18,data_brand19,data_brand20,data_brand21,data_brand22,data_brand23,data_brand24,data_brand25)

f,p2=f_oneway(data_brand1,data_brand2,data_brand3,data_brand4,data_brand5,data_brand6,data_brand7,data_brand8,data_brand9,data_brand10,data_brand11,data_brand12,data_brand13,data_brand14,data_brand15,data_brand16,data_brand17,data_brand18,data_brand19,data_brand20,data_brand21,data_brand22,data_brand23,data_brand24,data_brand25)
print(f"p1:{p1}")#检验其方差是否相等
print(f"p2:{p2}\n")#检验其方差是否相等

print("p2<0.05 说明母品牌是一个有影响的因素\n")



##******************价格的方差分析
print(colored("************价格的方差分析********************\n",attrs=["bold"]))
input("\n按 Enter 键继续运行...\n")

data_price1=data[data['价格']<=10]['首月零售量']
data_price2=data[data['价格']>10][data['价格']<=15]['首月零售量']
data_price3=data[data['价格']>15][data['价格']<=20]['首月零售量']
data_price4=data[data['价格']>20][data['价格']<=25]['首月零售量']
data_price5=data[data['价格']>25][data['价格']<=30]['首月零售量']
data_price6=data[data['价格']>30][data['价格']<=35]['首月零售量']
data_price7=data[data['价格']>35][data['价格']<=40]['首月零售量']
data_price8=data[data['价格']>40][data['价格']<=50]['首月零售量']
data_price9=data[data['价格']>50]['首月零售量']
w,p1=levene(data_price1,data_price2,data_price3,data_price4,data_price5,data_price6,data_price7,data_price8,data_price9)

f,p2=f_oneway(data_price1,data_price2,data_price3,data_price4,data_price5,data_price6,data_price7,data_price8,data_price9)
print(f"p1:{p1}")#检验其方差是否相等
print(f"p2:{p2}\n")#检验其方差是否相等

print("p2<0.05 说明价格是一个有影响的因素\n")





