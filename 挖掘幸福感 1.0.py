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

#将时间数据仅保留年月日
def change_time(row):
    time=str(row)[:10]
    time=dt.datetime.strptime(time,'%Y-%m-%d')
    return time

#利用分位数分箱
def cut(data,col,*b):
    dots=data[col].quantile(b)
    return dots

#根据分位数来映射
def change_num(row,data,col,*b):
    n=len(b)
    dots=cut(data,col,*b)
    for i in range(n):
        if dots.iloc[i+1]>=row>dots.iloc[i]:
            return i+2
        elif row>dots.iloc[n-1]:
            return n+1
        elif row<=dots.iloc[0]:
            return 1
        else:
            continue

#1. 初始数据特征
#导入相关数据并查看数据集信息
happiness_train_abbr=pd.read_csv('happiness_train_abbr.csv',header=0)
happiness_test_abbr=pd.read_csv('happiness_test_abbr.csv',header=0)
#print('查看训练集\n',happiness_train_abbr.head(2))
#print('查看测试集\n',happiness_test_abbr.head(2))
#print('训练集信息\n',happiness_train_abbr.info())
#print('测试集信息\n',happiness_test_abbr.info())
#得知训练集与测试集分别有42，41类特征；其中仅'survey_time'为object类型数据，其他均为数值型数据;并且训练集有5列有缺失值，测试集有四列具缺失值
#要注意通用的数字：-1 = 不适用; -2 = 不知道; -3 = 拒绝回答; -8 = 无法回答（这些值都是需要转化的）

#建立数据集副本，方便之后的特征处理
train=happiness_train_abbr.copy()
test=happiness_test_abbr.copy()

#将通用值全部先转化为缺失值
train=rep_data(train,np.nan,-1,-2,-3,-8)
test=rep_data(test,np.nan,-1,-2,-3,-8)
#print(train.info())
#print(test.info())
#从信息中可以得知有在训练集中'work_status','work_yr','work_type','work_manage'这几列的缺失值超过了50%应予以删除



#2. 数据预处理

#2.1 数据清洗

#删除训练集中缺失值比例超过50%的特征列
train.drop(columns=['work_status','work_yr','work_type','work_manage'],inplace=True)
test.drop(columns=['work_status','work_yr','work_type','work_manage'],inplace=True)
#print(train.info())
#print(test.info())

#绘制含连续型缺失值的特征的图形并对相应的缺失值进行填充
#连续型：均匀分布，用均值填充；倾斜分布，用中位数填充；离散型：用哑变量进行填充；label列用中位数填充

#income:都不是均匀分布，取中位数填充缺失值
#draw_bar(train,'income')
#draw_bar(test,'income')
train['income'].fillna(train['income'].median(),inplace=True)
test['income'].fillna(test['income'].median(),inplace=True)
#print(train.info())
#print(test.info())

#family_income:都不是均匀分布，取中位数填充缺失值
#draw_bar(train,'family_income')
#draw_bar(test,'family_income')
train['family_income'].fillna(train['family_income'].median(),inplace=True)
test['family_income'].fillna(test['family_income'].median(),inplace=True)
#print(train.info())
#print(test.info())

#family_m:取中位数填充
#draw_bar(train,'family_m')
#draw_bar(test,'family_m')
train['family_m'].fillna(train['family_m'].median(),inplace=True)
test['family_m'].fillna(test['family_m'].median(),inplace=True)
#print(train.info())
#print(test.info())

#house:取中位数填充
#draw_bar(train,'house')
#draw_bar(test,'house')
train['house'].fillna(train['house'].median(),inplace=True)
test['house'].fillna(test['house'].median(),inplace=True)
#print(train.info())
#print(test.info())

#离散型数据
#有大小关系的离散变量填充为中位数（'religion_freq','edu','political','health','health_problem','depression','socialize',
#'relax','learn','equity','class','family_status','status_peer','status_3_before','view','inc_ability'）
fill(train,'religion_freq','edu','political','health','health_problem','depression','socialize',
'relax','learn','equity','class','family_status','status_peer','status_3_before','view','inc_ability')
fill(test,'religion_freq','edu','political','health','health_problem','depression','socialize',
'relax','learn','equity','class','family_status','status_peer','status_3_before','view','inc_ability')
#print(train.info())
#print(test.info())

#无大小关系的离散变量填充为哑变量（'nationality','religion','car'）

#nationality进行one-hot编码(注意测试集中没有民族7，需要补上)
#训练集
cols=['nationality_1','nationality_2','nationality_3','nationality_4','nationality_5','nationality_6','nationality_7','nationality_8','nationality_NaN']
train[cols]=pd.get_dummies(train['nationality'],dummy_na=True)
train.drop(columns=['nationality'],inplace=True)
#print(train.info())
#测试集
cols=['nationality_1','nationality_2','nationality_3','nationality_4','nationality_5','nationality_6','nationality_8','nationality_NaN']
test[cols]=pd.get_dummies(test['nationality'],dummy_na=True)
test['nationality_9']=test['nationality_NaN']
test['nationality_NaN']=test['nationality_8']
test['nationality_8']=0
test.rename(columns={'nationality_8':'nationality_7','nationality_NaN':'nationality_8','nationality_9':'nationality_NaN'},inplace=True)
test.drop(columns=['nationality'],inplace=True)
#print(test.info())

#religion进行one-hot编码
#训练集
cols=['religion_is','religion_not','religion_NaN']
train[cols]=pd.get_dummies(train['religion'],dummy_na=True)
train.drop(columns=['religion'],inplace=True)
#print(train.info())
#测试集
cols=['religion_is','religion_not','religion_NaN']
test[cols]=pd.get_dummies(test['religion'],dummy_na=True)
test.drop(columns=['religion'],inplace=True)
#print(test.info())

#car进行one-hot编码
#训练集
cols=['car_have','car_not','car_NaN']
train[cols]=pd.get_dummies(train['car'],dummy_na=True)
train.drop(columns=['car'],inplace=True)
#print(train.info())
#测试集
cols=['car_have','car_not','car_NaN']
test[cols]=pd.get_dummies(test['car'],dummy_na=True)
test.drop(columns=['car'],inplace=True)
#print(test.info())

#对label的缺失值进行填充(用中位数进行填充),要从0开始编码，因为XGBoost模型默认为该形式
train['happiness'].fillna(train['happiness'].median(),inplace=True)
train['happiness']=train['happiness']-1
#print(train['happiness'].unique())
#print(train.info())



#2.2 构造特征

#关于时间与年龄特征的初始化（注意最后要将时间删除）
#训练集
train['survey_time']=pd.to_datetime(train['survey_time'])
train['survey_time']=train['survey_time'].apply(change_time)
train['birth']=pd.to_datetime(train['birth'],format="%Y").dt.year
train['age']=train['survey_time'].dt.year-train['birth']
train.drop(columns=['survey_time','birth'],inplace=True)
#print(train.head(2))
#测试集
test['survey_time']=pd.to_datetime(test['survey_time'])
test['survey_time']=test['survey_time'].apply(change_time)
test['birth']=pd.to_datetime(test['birth'],format="%Y").dt.year
test['age']=test['survey_time'].dt.year-test['birth']
test.drop(columns=['survey_time','birth'],inplace=True)
#print(test.head(2))


#2.3 数据变换

#2.3.1 数据稀疏化（对需要进行one-hot编码，但还没有进行相应编码的变量进行编码）
#('hukou','work_exper','marital')

#hukou:注意test集合中缺少类别7
#训练集
cols=['hukou_1','hukou_2','hukou_3','hukou_4','hukou_5','hukou_6','hukou_7','hukou_8']
train[cols]=pd.get_dummies(train['hukou'])
train.drop(columns=['hukou'],inplace=True)
#print(train.info())
#测试集
cols=['hukou_1','hukou_2','hukou_3','hukou_4','hukou_5','hukou_6','hukou_8']
test[cols]=pd.get_dummies(test['hukou'])
test['hukou_9']=test['hukou_8']
test['hukou_8']=0
test.rename(columns={'hukou_8':'hukou_7','hukou_9':'hukou_8'},inplace=True)
test.drop(columns=['hukou'],inplace=True)
#print(test.info())

#work_exper
#训练集
cols=['work_exper_1','work_exper_2','work_exper_3','work_exper_4','work_exper_5','work_exper_6']
train[cols]=pd.get_dummies(train['work_exper'])
train.drop(columns=['work_exper'],inplace=True)
#print(train.info())
#测试集 
cols=['work_exper_1','work_exper_2','work_exper_3','work_exper_4','work_exper_5','work_exper_6']
test[cols]=pd.get_dummies(test['work_exper'])
test.drop(columns=['work_exper'],inplace=True)
#print(test.info())

#marital
#训练集
cols=['marital_1','marital_2','marital_3','marital_4','marital_5','marital_6','marital_7']
train[cols]=pd.get_dummies(train['marital'])
train.drop(columns=['marital'],inplace=True)
#print(train.info())
#测试集
cols=['marital_1','marital_2','marital_3','marital_4','marital_5','marital_6','marital_7']
test[cols]=pd.get_dummies(test['marital'])
test.drop(columns=['marital'],inplace=True)
#print(test.info())

#2.3.2 数据离散化：对跨度较大的的连续变量进行分箱处理(采用分位数分箱,全部分为10份)
#'income','floor_area','height_cm','weight_jin','family_income','age'

#训练集
train['income']=train['income'].apply(change_num,args=(train,'income',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
train['floor_area']=train['floor_area'].apply(change_num,args=(train,'floor_area',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
train['height_cm']=train['height_cm'].apply(change_num,args=(train,'height_cm',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
train['weight_jin']=train['weight_jin'].apply(change_num,args=(train,'weight_jin',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
train['family_income']=train['family_income'].apply(change_num,args=(train,'family_income',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
train['age']=train['age'].apply(change_num,args=(train,'age',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
#print(train[['birth','income','floor_area','height_cm','weight_jin','family_income','age']].head(5))
#测试集
test['income']=test['income'].apply(change_num,args=(test,'income',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
test['floor_area']=test['floor_area'].apply(change_num,args=(test,'floor_area',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
test['height_cm']=test['height_cm'].apply(change_num,args=(test,'height_cm',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
test['weight_jin']=test['weight_jin'].apply(change_num,args=(test,'weight_jin',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
test['family_income']=test['family_income'].apply(change_num,args=(test,'family_income',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
test['age']=test['age'].apply(change_num,args=(test,'age',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
#print(test[['birth','income','floor_area','height_cm','weight_jin','family_income','age']].head(5))

#将训练集和测试集的特征提取出来(去掉label和id)
#训练集特征
feature_train=train.copy()
feature_label=pd.DataFrame(columns=[])
feature_train.drop(columns=['happiness','id'],inplace=True)
feature_label['happiness']=train['happiness']
#print(feature_train.info())
#print(feature_label.info())
#测试集特征
feature_test=test.copy()
feature_test.drop(columns=['id'],inplace=True)
#print(feature_test.info())

#2.3.3 特征归一化（使用均值归一化,考虑std=0时）
#训练集特征
feature_train_norm=feature_train.apply(lambda x: 0 if np.std(x,ddof=1)==0 else (x-np.mean(x))/(np.std(x,ddof=1)))
#print(feature_train_norm.head(2))
#print(feature_train_norm.info())
#测试集特征
feature_test_norm=feature_test.apply(lambda x: 0 if np.std(x,ddof=1)==0 else (x-np.mean(x))/(np.std(x,ddof=1)))
#print(feature_test_norm.head(2))
#print(feature_test_norm.info())



#3. 模型构建

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
    'max_depth':30,#构建树的深度，越大越容易过拟合
    'lambda':4,#控制模型复杂度的权重值L2正则化项的参数，越大越不容易过拟合
    'subsample':0.7,#随机采样训练样本
    'colsample_bytree':0.7,#生成树时的列采样
    'min_child_weight':3,
    'silent':1,#设置成1，则没有运行信息输出，最好设置为0
    'eta':0.1,#如同学习率
    'seed':1000,#随机数种子
    'nthread':4,#线程数     
        }

plst=params.items()#查看键值对
num_rounds=500#迭代次数

#训练模型
model=xgb.train(plst,dtrain,num_rounds)

#计算训练集的准确率
pre_train=model.predict(xgb.DMatrix(X_train))
print("训练集的准确率为：{:.3f}".format(accuracy_score(y_train,pre_train)))



#4. 验证集预测并评分

#对验证集进行预测
pre_val=model.predict(dval)

#计算准确率
print("验证集预测的准确率为：{:.3f}".format(accuracy_score(y_test,pre_val)))

#计算得分（实际上为损失）
y_test=y_test.values
n=y_test.shape[0]
pre_val=np.reshape(pre_val,(n,1))
score=np.dot((y_test-pre_val).T,(y_test-pre_val))/n
print(score)

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
res['id']=test['id']
res['happiness']=pd.DataFrame(pre_test)
print(pre_test[0:100])
print(res)
res.to_csv('happiness_submit.csv',index=False)#储存为csv










