# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:36:50 2019

@author: 刘永聪
"""

'''
赛题：挖掘幸福感（精简版）
解题流程：1. 初始数据特征
         2. 数据预处理：数据清洗（缺失值处理/离群点处理/噪声处理）-数据集成-构造特征-数据变换（离散化/稀疏化/规范化）-数据规约（维度规约/维度变换）
         3. 构造模型（正则化 多分类 逻辑回归 xgboost）(训练集，验证集，归一化，正则化)
         4. 验证集预测并评分
         5. 预测并生成结果文件（id,prediction,csv文件）
         6. 特征与模型改进（参考计量经济学，在建模时遇到的问题,one-hot编码优化(或采用众数),分箱种类，学习方法,根据特征的相关性，删除无关特征）
备注：该代码中仅对精简版数据进行处理，因此不再强调“精简版”关键词
'''
#导入相关库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

#相关函数定义
'''
1. 初始数据特征 相关函数
'''
#替换通用数据
def rep_data(data,a,*b):
    #data为需要进行替换的数据集，a为新内容，b为旧内容，返回新的结果
    for i in b:
        data=data.replace(i,a)
    return data

'''
2. 数据预处理 相关函数
'''
#绘制连续型数据柱状图
def draw_bar(data,col):
    data.loc[:,[col]].plot.bar()
    plt.xticks([]) 
    plt.show()

#对列填充相应的中位数
def fill(data,*b):
    for col in b:
        data[col].fillna(data[col].median(),inplace=True)

#判断哪些列拥有通用值
def Judge_value(data,ls,*b):
    #data为数据集，ls为通用值列表，b为列名
    res=[]
    for col in b:
        for value in ls:
            if value in data[col].values:
                res.append(col)
                break
    return res

#将收入分区
def cut_income(row):
    if row<=0:
        return 0
    elif 0<row<=5000:
        return 1
    elif 5000<row<=10000:
        return 2
    elif 10000<row<=50000:
        return 3
    elif 50000<row<=100000:
        return 4
    elif 100000<row<=200000:
        return 5
    else:
        return 6

#将家庭收入分区
def cut_family_income(row):
    if row<=0:
        return 0
    elif 0<row<=10000:
        return 1
    elif 10000<row<=50000:
        return 2
    elif 50000<row<=100000:
        return 3
    elif 100000<row<=150000:
        return 4
    elif 150000<row<=200000:
        return 5
    else:
        return 6
    
#年龄分区
def cut_age(row):
    if row<=30:
        return 0
    elif 30<row<=40:
        return 1
    elif 40<row<=50:
        return 2
    elif 50<row<=60:
        return 3
    else:
        return 4

#将时间数据仅保留年月日
def change_time(row):
    time=str(row)[:10]
    time=dt.datetime.strptime(time,'%Y-%m-%d')
    return time

#判断是否周末
def weekday_type(row):
    if row in [6,7]:
        return 1
    else:
        return 0

#导入相关数据并查看数据集信息
happiness_train_abbr=pd.read_csv('happiness_train_abbr.csv',header=0)
happiness_test_abbr=pd.read_csv('happiness_test_abbr.csv',header=0)
#print('训练集信息\n',happiness_train_abbr.info())
#print('测试集信息\n',happiness_test_abbr.info())
#得知训练集与测试集分别有42，41类特征；其中仅'survey_time'为object类型数据，其他均为数值型数据;并且训练集有5列有缺失值，测试集有四列具缺失值
#要注意通用的数字：-1 = 不适用; -2 = 不知道; -3 = 拒绝回答; -8 = 无法回答（这些值都是需要转化的）

#将label列提取出来，并将label列从数据集中删除，合并训练集与测试集方便数据处理
label_happiness=pd.DataFrame(happiness_train_abbr.loc[:,'happiness'],columns=['happiness'])#提取label
happiness_train_abbr.drop(columns=['happiness'],inplace=True)#训练集中删除label
data=pd.concat([happiness_train_abbr,happiness_test_abbr],axis=0).reset_index(drop=True)#合并训练集与测试



#缺失值处理

#删除缺失值比例大于20%的变量'work_status','work_yr','work_type','work_manage'
data.drop(columns=['work_status','work_yr','work_type','work_manage'],inplace=True)

#将'family_income'中的一位缺失值填充，用中位数填补
fill(data,'family_income')

#对通用含义数值进行处理
#判断哪些列有通用值
res=Judge_value(data,[-1,-2,-3,-8],'id', 'survey_type', 'province', 'city', 'county', 'survey_time', 'gender',
            'birth', 'nationality', 'religion', 'religion_freq', 'edu', 'income', 'political', 'floor_area', 
            'height_cm', 'weight_jin', 'health', 'health_problem', 'depression', 'hukou', 'socialize', 'relax', 
            'learn', 'equity', 'class', 'work_exper', 'family_income', 'family_m', 'family_status', 'house', 'car', 
            'marital', 'status_peer', 'status_3_before', 'view', 'inc_ability')
#print(res)
#需要进行通用值处理的列有'nationality', 'religion', 'religion_freq', 'edu', 'income', 
#'political', 'health', 'health_problem', 'depression', 'socialize', 'relax', 'learn', 
#'equity', 'class', 'family_income', 'family_m', 'family_status', 'house', 'car', 'status_peer', 
#'status_3_before', 'view', 'inc_ability'



#对通用数据进行处理，并对该分箱的数据进行分箱
#survey_type：将1，2变为0，1
data['survey_type']=data['survey_type']-1

#gender：将1，2变为0，1
data['gender']=data['gender']-1

#nationality:因为汉族远远大于其他族的人数，所以仅分为非汉0，汉1;对于通用值，基本不是汉族默认为非汉
#print(data['nationality'].value_counts())
data['nationality'][data['nationality']!=1]=0

#religion:从分布可以看出通用值仅有-8，无法回答，此类情况认为是信仰宗教的1
#print(data['religion'].value_counts())
data['religion'][data['religion']==-8]=1

#religion_freq:通用值仅有-8，无法回答，此种情况默认是参加过的，选择除1，外次数最多的3
#print(data['religion_freq'].value_counts())
data['religion_freq'][data['religion_freq']==-8]=3

#edu:通用值仅有-8，无法回答，此种情况默认是用众数4代替
#print(data['edu'].value_counts())
data['edu'][data['edu']==-8]=4

#income:为了避免极端值的影响，将其分箱，将通用值组放入众数区间收入3
#draw_bar(data,'income')
data['income']=data['income'].apply(cut_income)
#print(data['income'].value_counts())
data['income'][data['income']==0]=3

#political:根据常识可知，一般政治关系可以分为党员，与非党员，-8默认为非党员
#print(data['political'].value_counts())
data['political'][data['political']!=4]=0
data['political'][data['political']==4]=1

#floor_area/height_cm/weight_jin:无极端值不用分箱
#print(data['floor_area'].max())
#print(data['floor_area'].min())
#print(data['height_cm'].max())
#print(data['height_cm'].min())
#print(data['weight_jin'].max())
#print(data['weight_jin'].min())

#health:对于-8，默认为3一般
#print(data['health'].value_counts())
data['health'][data['health']==-8]=3

#health_problem：对于-8，认为存在影响，默认其为存在影响中的众数4
#print(data['health_problem'].value_counts())
data['health_problem'][data['health_problem']==-8]=4

#depression：对于-8，认为存在影响，默认其为存在影响中的众数4
#print(data['depression'].value_counts())
data['depression'][data['depression']==-8]=4

#hukou：种类变量，仅分为农业户口与非农业户口即可
#print(data['hukou'].value_counts())
data['hukou'][data['hukou']!=1]=0

#socialize:-8默认为答题人不确定，因此排除1，5，选择其他种类中的最多的种类替换
#print(data['socialize'].value_counts())
data['socialize'][data['socialize']==-8]=2

#relax：-8默认为答题人不确定，因此排除1，5，选择其他种类中的最多的种类替换
#print(data['relax'].value_counts())
data['relax'][data['relax']==-8]=4

#learn：-8默认为答题人不确定，因此排除1，5，选择其他种类中的最多的种类替换
#print(data['learn'].value_counts())
data['learn'][data['learn']==-8]=2

#equity：-8默认答题人不确定，因此排除1，5，选择其他种类中的最多的种类替换
#print(data['equity'].value_counts())
data['equity'][data['equity']==-8]=4

#class：-8默认答题人不确定，因此排除1，10，选择其他种类中的最多的种类替换
#print(data['class'].value_counts())
data['class'][data['class']==-8]=5

#work_exper：无序种类变量，不适合变为二值类型，之后采用one-hot编码
#print(data['work_exper'].value_counts())

#family_income：为了避免极端值的影响，将其分箱，将通用值组放入众数区间收入2
#draw_bar(data,'family_income')
data['family_income']=data['family_income'].apply(cut_family_income)
#print(data['family_income'].value_counts())
data['family_income'][data['family_income']==0]=2

#family_m:出现明显异常值50，应该将其更改为5，-1替换为0，-2替换为2，-3替换为9
#print(data['family_m'].value_counts())
data['family_m'][data['family_m']==50]=5
data['family_m'][data['family_m']==-1]=0
data['family_m'][data['family_m']==-2]=2
data['family_m'][data['family_m']==-3]=9

#family_status：-8默认排除极端情况那个1，5，选择剩余的众数3
#print(data['family_status'].value_counts())
data['family_status'][data['family_status']==-8]=3

#house：因差距较大，需要进行分箱处理；-3认为高房产，-1认为无房产，-2认为至少2套房产
#print(data['house'].value_counts())
data['house'][data['house']>5]=8
data['house'][data['house']==-1]=0
data['house'][data['house']==-2]=2
data['house'][data['house']==-3]=8

#car:-8默认为众数2，并转化为0，1
#print(data['car'].value_counts())
data['car'][data['car']==-8]=2
data['car'][data['car']==1]=0
data['car'][data['car']==2]=1

#marital：该无序种类变量，各种类数目没有相差过大不能分成二值变量，因此之后转化为one-hot编码
#print(data['marital'].value_counts())

#status_peer：-8默认为众数2
#print(data['status_peer'].value_counts())
data['status_peer'][data['status_peer']==-8]=2

#status_3_before：-8默认为众数2
#print(data['status_3_before'].value_counts())
data['status_3_before'][data['status_3_before']==-8]=2

#view：-8，默认为3
#print(data['view'].value_counts())
data['view'][data['view']==-8]=3

#inc_ability：-8默认为众数2
#print(data['inc_ability'].value_counts())
data['inc_ability'][data['inc_ability']==-8]=2



#扩充特征（年龄，采访的年，月份，星期，星期类型，BMI,人均住房面积）

#age:为了提高计算速度，需要进行分箱
data['survey_time']=pd.to_datetime(data['survey_time'])
data['survey_time']=data['survey_time'].apply(change_time)
data['birth']=pd.to_datetime(data['birth'],format="%Y").dt.year
data['age']=data['survey_time'].dt.year-data['birth']
#print(data['age'].min())
data['age']=data['age'].apply(cut_age)

#year/month/weekday/weekday_type
data['year']=data['survey_time'].dt.year
data['month']=data['survey_time'].dt.month
data['weekday']=data['survey_time'].dt.weekday+1
data['weekday_type']=data['weekday'].apply(weekday_type)

#BMI:注意单位换算
data['BMI']=(data['weight_jin']/2)/((data['height_cm']/100)*(data['height_cm']/100))

#人均住房面积
data['per_floor_area']=data['floor_area']/data['family_m']



#为了避免one-hot导致数据集暴增，仅对'work_exper','marital','province'
#one-hot编码

#work_exper
cols=['work_exper_1','work_exper_2','work_exper_3','work_exper_4','work_exper_5','work_exper_6']
data[cols]=pd.get_dummies(data['work_exper'])

#marital
cols=['marital_1','marital_2','marital_3','marital_4','marital_5','marital_6','marital_7']
data[cols]=pd.get_dummies(data['marital'])

#province
#print(list(pd.get_dummies(data['province'])))#查看有哪些列
cols=['pro_1','pro_2','pro_3','pro_4','pro_5','pro_6','pro_7','pro_8','pro_9','pro_10','pro_11','pro_12',
      'pro_13','pro_15','pro_16','pro_17','pro_18','pro_19','pro_21','pro_22','pro_23',
      'pro_24','pro_26','pro_27','pro_28','pro_29','pro_30','pro_31']
data[cols]=pd.get_dummies(data['province'])

#删除目前数据集中该删除的变量'id','province','city','county','survey_time','birth','work_exper','marital'
data.drop(columns=['id','province','city','county','survey_time','birth','work_exper','marital'],inplace=True)



#对label_happiness进行处理:-8除开极端值，用众数4填充，并且编号要从0开始
res_1=Judge_value(label_happiness,[-1,-2,-3,-8],'happiness')#判断是否有通用值
#print(res_1)
#print(label_happiness['happiness'].value_counts())#判断通用值是哪个
label_happiness['happiness'][label_happiness['happiness']==-8]=4
label_happiness['happiness']=label_happiness['happiness']-1



#得到训练集特征、测试集特征以及训练集标签
feature_train=data.iloc[:8000,:]
feature_test=data.iloc[8000:,:]
feature_label=label_happiness.copy()



#特征归一化（使用均值归一化,考虑std=0时）
#训练集特征
feature_train_norm=feature_train.apply(lambda x: 0 if np.std(x,ddof=1)==0 else (x-np.mean(x))/(np.std(x,ddof=1)))
#测试集特征
feature_test_norm=feature_test.apply(lambda x: 0 if np.std(x,ddof=1)==0 else (x-np.mean(x))/(np.std(x,ddof=1)))



#模型构建

#训练集和验证集划分
X_train,X_test,y_train,y_test=train_test_split(feature_train_norm,feature_label,test_size=0.3,random_state=1000)

#将数据都转化为Dmatrix格式
dtrain=xgb.DMatrix(X_train,y_train)
dval=xgb.DMatrix(X_test)
dtest=xgb.DMatrix(feature_test_norm)

#设置模型参数
params={
    'booster':'gbtree',
    'objective':'multi:softmax',#多分类问题
    'num_class':5,#类别数，与'multi:softmax'并用
    'gamma':0.1,#用于控制是否剪枝的参数，越大越保守，一般0.1，0.2
    'max_depth':7,#构建树的深度，越大越容易过拟合
    'lambda':2,#控制模型复杂度的权重值L2正则化项的参数，越大越不容易过拟合
    'subsample':0.7,#随机采样训练样本
    'colsample_bytree':0.8,#生成树时的列采样
    'min_child_weight':1,
    'silent':1,#设置成1，则没有运行信息输出，最好设置为0
    'eta':0.005,#如同学习率
    'seed':1000,#随机数种子
    'nthread':8,#线程数     
        }

plst=params.items()#查看键值对
num_rounds=2000#迭代次数

#训练模型
model=xgb.train(plst,dtrain,num_rounds)

#计算训练集的准确率
pre_train=model.predict(xgb.DMatrix(X_train))
print("训练集的准确率为：{:.3f}".format(accuracy_score(y_train,pre_train)))



#验证集预测并评分

#对验证集进行预测
pre_val=model.predict(dval)

#计算准确率
print("验证集预测的准确率为：{:.3f}".format(accuracy_score(y_test,pre_val)))

#计算得分（实际上为损失）
y_test=y_test.values
n=y_test.shape[0]
pre_val=np.reshape(pre_val,(n,1))
score=np.dot((y_test-pre_val).T,(y_test-pre_val))/n
print("得分为：",score)
'''
#显示重要特征
plot_importance(model)
plt.show()
'''


#5. 预测并生成结果文件

#预测dtest数据，并将结果保存为csv文件（注意预测结果要加1，进行还原）
pre_test=model.predict(dtest)
pre_test=pre_test+1#还原类别形式1-5
pre_test.reshape(-1,1)#转换为列向量
res=pd.DataFrame(columns=[])
res['id']=happiness_test_abbr['id']
res['happiness']=pd.DataFrame(pre_test)
print(res)
res.to_csv('happiness_submit.csv',index=False)#储存为csv










