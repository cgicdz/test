# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:38:14 2019

@author: 刘永聪
"""


'''
赛题：挖掘幸福感（完整版）
解题流程：1. 初始数据特征
         2. 数据预处理：数据清洗（缺失值处理/离群点处理/噪声处理）-数据集成-构造特征-数据变换（离散化/稀疏化/规范化）-数据规约（维度规约/维度变换）
         3. 构造模型（正则化 多分类 逻辑回归 xgboost）(训练集，验证集，归一化，正则化)
         4. 验证集预测并评分
         5. 预测并生成结果文件（id,prediction,csv文件）
         6. 特征与模型改进（参考计量经济学，在建模时遇到的问题,one-hot编码优化(或采用众数),分箱种类，学习方法,根据特征的相关性，删除无关特征）
备注：在该实例中产生缺失值的原因是，因为前面的题的选项，而不用作答的题（重点重点重点）；对连续变量进行分区时要考虑通用值的存在；对通用值进行处理
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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold,RepeatedKFold


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
def Judge_value(data,ls,b):
    #data为数据集，ls为通用值列表，b为列名
    res=[]
    for col in b:
        for value in ls:
            if value in data[col].values:
                res.append(col)
                break
    return res

#查看具体的通用值是什么
def check_value(data,col):
    print(data[col][data[col]<0].value_counts())
    
#查看最大值，最小值
def min_max(data,col):
    Min=data[col][data[col]>=0].min()
    Max=data[col][data[col]>=0].max()
    print("（{}，{}）".format(Min,Max))

#关于年份分区(考虑缺失值和通用值)
def cut_yr(row):
    if pd.isnull(row)==True:
        return 0
    elif 1900<row<=1949:
        return 1
    elif 1949<row<=1978:
        return 2
    elif 1978<row<=2000:
        return 3
    elif row>=2000:
        return 4
#关于父母出生年份分区(考虑缺失值和通用值)
def cut_fm_yr(row):
    if pd.isnull(row)==True:
        return 0
    elif 10<row<=1900:
        return 1
    elif 1900<row<=1930:
        return 2
    elif 1930<row<=1960:
        return 3
    elif 1960<row<=1990:
        return 3
    elif row>=1990:
        return 4
    
#将收入分区:包括个人收入和家庭收入
def cut_income(row):
    if 0<=row<=5000:
        return 1
    elif 5000<row<=10000:
        return 2
    elif 10000<row<=50000:
        return 3
    elif 50000<row<=100000:
        return 4
    elif 100000<row<=200000:
        return 5
    elif row>200000:
        return 6
    elif pd.isnull(row)==True:
        return 0



#将家庭收入分区
def cut_family_income(row):   
    if 0<=row<=10000:
        return 1
    elif 10000<row<=50000:
        return 2
    elif 50000<row<=100000:
        return 3
    elif 100000<row<=150000:
        return 4
    elif 150000<row<=200000:
        return 5
    elif row>200000:
        return 6
    elif pd.isnull(row)==True:
        return 0

#对服务评分分区
def cut_service(row):
    if 0<=row<10:
        return 1
    elif 10<=row<20:
        return 2
    elif 20<=row<30:
        return 3
    elif 30<=row<40:
        return 4
    elif 40<=row<50:
        return 5
    elif 50<=row<60:
        return 6
    elif 60<=row<70:
        return 7
    elif 70<=row<80:
        return 8
    elif 80<=row<90:
        return 9
    elif 90<=row<=100:
        return 10
    
    
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

#把一天的时间划分
def hour_cut(row):
    if 0<=row<6:
        return 0
    elif 6<=row<8:
        return 1
    elif 8<=row<12:
        return 2
    elif 12<=row<14:
        return 3
    elif 14<=row<18:
        return 4
    elif 18<=row<21:
        return 5
    elif 21<=row<24:
        return 6

#出生的年代划分
def birth_split(row):
    if 1920<=row<=1930:
        return 0
    elif  1930<row<=1940:
        return 1
    elif  1940<row<=1950:
        return 2
    elif  1950<row<=1960:
        return 3
    elif  1960<row<=1970:
        return 4
    elif  1970<row<=1980:
        return 5
    elif  1980<row<=1990:
        return 6
    elif  1990<row<=2000:
        return 7

#将收入分组
def income_cut(row):
    if row<0:
        return 0
    elif  0<=row<1200:
        return 1
    elif  1200<row<=10000:
        return 2
    elif  10000<row<24000:
        return 3
    elif  24000<row<40000:
        return 4
    elif  40000>=row:
        return 5

#自定义评价函数
def myFeval(preds,xgbtrain):
    label=xgbtrain.get_label()
    score=mean_squared_error(label,preds)
    return 'myFeval',score

#根据得分分类
def pre(predictions_xgb):
    if predictions_xgb<=0.5:
        return 1
    elif 0.5<predictions_xgb<=1.5:
        return 2
    elif 1.5<predictions_xgb<=2.5:
        return 3
    elif 2.5<predictions_xgb<=3.5:
        return 4
    elif predictions_xgb>3.5:
        return 5
    
    
#导入相关数据，并查看数据集信息
happiness_train=pd.read_csv('happiness_train_complete.csv',header=0,encoding='ISO-8859-1')
happiness_test=pd.read_csv('happiness_test_complete.csv',header=0,encoding='ISO-8859-1')
#print(happiness_train.info(verbose=True,null_counts=True))
#print(happiness_test.info(verbose=True,null_counts=True))

#将训练集的标签列剔除出来，并将训练集与测试集合并
label_happiness=pd.DataFrame(happiness_train.loc[:,'happiness'],columns=['happiness'])#提取label
happiness_train.drop(columns=['happiness'],inplace=True)#训练集中删除label
data=pd.concat([happiness_train,happiness_test],axis=0).reset_index(drop=True)#合并训练集与测试
#print(label_happiness.info())
#print(data.info(verbose=True,null_counts=True))

#对标签列的通用值进行处理:-8默认为3(对于特征变量的通用值先不进行处理),并从0开始编号
#print(label_happiness['happiness'].value_counts())
label_happiness['happiness'][label_happiness['happiness']==-8]=3
label_happiness['happiness']=label_happiness['happiness']-1

#处理时间特征
data['survey_time']=pd.to_datetime(data['survey_time'],format='%Y-%m-%d %H:%M:%S')
data['year']=data['survey_time'].dt.year
data['month']=data['survey_time'].dt.month
data['weekday']=data['survey_time'].dt.weekday+1
data['quarter']=data['survey_time'].dt.quarter
data['hour']=data['survey_time'].dt.hour
data['hour_cut']=data['hour'].apply(hour_cut)
data['age']=data['year']-data['birth']

#删除不需要的变量'edu_other','survey_time'
data.drop(columns=['edu_other','survey_time'],inplace=True)

'''是否入党(与样例不一样)'''
data['join_party'][data['join_party']!=4]=0
data['join_party'][data['join_party']==4]=1

#出生的年代
data['birth_s']=data['birth'].apply(birth_split)

#收入分组
data['income_cut']=data['income'].apply(income_cut)

#填充数据
data['edu_status'].fillna(5,inplace=True)
data['edu_yr'].fillna(-2,inplace=True)
data['property_other']=data['property_other'].map(lambda x: 0 if pd.isnull(x) else 1)
data['hukou_loc'].fillna(1,inplace=True)
data['social_neighbor'].fillna(8,inplace=True)
data['social_friend'].fillna(8,inplace=True)
data['work_status'].fillna(0,inplace=True)
data['work_yr'].fillna(0,inplace=True)
data['work_type'].fillna(0,inplace=True)
data['work_manage'].fillna(0,inplace=True)
data['family_income'].fillna(-2,inplace=True)
data['invest_other']=data['invest_other'].map(lambda x: 0 if pd.isnull(x) else 1)
data['minor_child'].fillna(0,inplace=True)
data['marital_1st'].fillna(0,inplace=True)
data['s_birth'].fillna(0,inplace=True)
data['marital_now'].fillna(0,inplace=True)
data['s_edu'].fillna(0,inplace=True)
data['s_political'].fillna(0,inplace=True)
data['s_hukou'].fillna(0,inplace=True)
data['s_income'].fillna(0,inplace=True)
data['s_work_exper'].fillna(0,inplace=True)
data['s_work_status'].fillna(0,inplace=True)
data['s_work_type'].fillna(0,inplace=True)

#删除变量
data.drop(columns=['id'],inplace=True)

#得到训练集特征、测试集特征以及训练集标签
feature_train=data.iloc[:8000,:]
feature_test=data.iloc[8000:,:]
feature_label=label_happiness.copy()



#模型构建
X_train=np.array(feature_train)
y_train=np.array(feature_label)
X_test=np.array(feature_test)

#设置模型参数
xgb_params={
    'booster':'gbtree',
    'eta':0.005,
    'max_depth':5,
    'subsample':0.7,
    'colsample_bytree':0.8,
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'silent':True,
    'nthread':8 
        }

folds=KFold(n_splits=3,shuffle=True,random_state=2018)
oof_xgb=np.zeros(len(feature_train))
predictions_xgb=np.zeros(len(feature_test))

for fold_,(trn_idx,val_idx) in enumerate(folds.split(X_train,y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data=xgb.DMatrix(X_train[trn_idx],y_train[trn_idx])
    val_data=xgb.DMatrix(X_train[val_idx],y_train[val_idx])
    watchlist=[(trn_data,'train'),(val_data,'valid_data')]
    clf=xgb.train(dtrain=trn_data,num_boost_round=4000,params=xgb_params,evals=watchlist,early_stopping_rounds=200,verbose_eval=100,feval=myFeval)
    oof_xgb[val_idx]=clf.predict(xgb.DMatrix(X_train[val_idx]),ntree_limit=clf.best_ntree_limit)
    predictions_xgb +=(clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit)/3)
print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train)))  

#5. 预测并生成结果文件

#预测dtest数据，并将结果保存为csv文件（注意预测结果要加1，进行还原）
pre_test=predictions_xgb.reshape(-1,1)#还原类别形式1-5
res=pd.DataFrame(columns=[])
res['id']=happiness_test['id']
res['happiness']=pd.DataFrame(pre_test)
res['happiness']=res['happiness'].apply(pre)
print(res['happiness'])

res.to_csv('happiness_submit.csv',index=False)#储存为csv
