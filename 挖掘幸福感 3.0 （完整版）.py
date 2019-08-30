# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:58:50 2019

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
    elif row==-1:
        return 0
    elif row in [-2,-3]:
        return 2   
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
    elif row==-1:
        return 0
    elif row in [-2,-3]:
        return 2  
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
    elif row==-3:
        return 6
    elif row in [-1,-2]:
        return 2

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
    elif row==-3:
        return 6
    elif row in [-1,-2]:
        return 2

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
    elif row==-3:
        return 4
    elif row in [-1,-2,-8]:
        return 7
    
    
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

#删除缺失值较多的变量：'edu_other','property_other','invest_other'
data.drop(columns=['edu_other','property_other','invest_other'],inplace=True)

#查看哪些变量具有通用值
res=Judge_value(data,[-1,-2,-3,-8],list(data))
#print(res)


#对缺失值进行处理（若有需要分区的变量可以先进行分区，同时注意相关变量的通用值处理）
#含缺失值变量：'edu_status','edu_yr','join_party','hukou_loc','social_neighbor','social_friend','work_status',
#'work_yr','work_type','work_manage','family_income','minor_child','marital_1st','s_birth',
#'marital_now','s_edu','s_political','s_hukou','s_income','s_work_exper','s_work_status','s_work_type'

#离散型(缺失值均用0替换)：'edu_status','hukou_loc','social_neighbor','social_friend','work_status','work_yr','work_type',
#'work_manage','minor_child','s_edu','s_political','s_hukou','s_work_exper','s_work_status','s_work_type'
'''
#查看相应的通用值
print(data['edu_status'].value_counts())
print(data['hukou_loc'].value_counts())
print(data['social_neighbor'].value_counts())
print(data['social_friend'].value_counts())
print(data['work_status'].value_counts())
print(data['work_yr'].value_counts())
print(data['work_type'].value_counts())
print(data['work_manage'].value_counts())
print(data['minor_child'].value_counts())
print(data['s_edu'].value_counts())
print(data['s_political'].value_counts())
print(data['s_hukou'].value_counts())
print(data['s_work_exper'].value_counts())
print(data['s_work_status'].value_counts())
print(data['s_work_type'].value_counts())
'''
#替换
data['edu_status'].fillna(5,inplace=True)
data['edu_status'][data['edu_status']==-8]=2

data['hukou_loc'].fillna(1,inplace=True)

data['social_neighbor'].fillna(8,inplace=True)
data['social_neighbor'][data['social_neighbor']==-8]=2

data['social_friend'].fillna(8,inplace=True)
data['social_friend'][data['social_friend']==-8]=3

data['work_status'].fillna(0,inplace=True)
data['work_status'][data['work_status']==-8]=3

data['work_yr'].fillna(0,inplace=True)
data['work_yr'][data['work_yr']==-1]=0
data['work_yr'][data['work_yr']==-2]=20
data['work_yr'][data['work_yr']==-3]=20

data['work_type'].fillna(0,inplace=True)
data['work_type'][data['work_type']==-8]=1

data['work_manage'].fillna(0,inplace=True)
data['work_manage'][data['work_manage']==-8]=3

data['minor_child'].fillna(0,inplace=True)
data['minor_child'][data['minor_child']==-8]=0

data['s_edu'].fillna(0,inplace=True)
data['s_edu'][data['s_edu']==-8]=4

data['s_political'].fillna(0,inplace=True)
data['s_political'][data['s_political']==-8]=1

data['s_hukou'].fillna(0,inplace=True)
data['s_hukou'][data['s_hukou']==-8]=1

data['s_work_exper'].fillna(0,inplace=True)

data['s_work_status'].fillna(0,inplace=True)
data['s_work_status'][data['s_work_status']==-8]=3

data['s_work_type'].fillna(0,inplace=True)
data['s_work_type'][data['s_work_type']==-8]=1


#连续型:'edu_yr','join_party','family_income','marital_1st','s_birth','marital_now','s_income'
'''
#先查看区间
min_max(data,'edu_yr')
min_max(data,'join_party')
min_max(data,'family_income')
min_max(data,'marital_1st')
min_max(data,'s_birth')
min_max(data,'marital_now')
min_max(data,'s_income')

#查看通用值
check_value(data,'edu_yr')
check_value(data,'join_party')
check_value(data,'family_income')
check_value(data,'marital_1st')
check_value(data,'s_birth')
check_value(data,'marital_now')
check_value(data,'s_income')
'''
#将各连续变量分区（注意考虑缺失值和通用值）
data['edu_yr']=data['edu_yr'].apply(cut_yr)
data['join_party']=data['join_party'].apply(cut_yr)
data['marital_1st']=data['marital_1st'].apply(cut_yr)
data['s_birth']=data['s_birth'].apply(cut_yr)
data['marital_now']=data['marital_now'].apply(cut_yr)
data['family_income']=data['family_income'].apply(cut_family_income)
data['s_income']=data['s_income'].apply(cut_income)


#需要转换为二值离散变量的值
data['survey_type']=data['survey_type']-1
data['gender']=data['gender']-1
data['nationality'][data['nationality']!=1]=0
data['religion'][data['religion']!=0]=1
data['political'][data['political']!=4]=0
data['hukou'][data['hukou']!=1]=0
data['work_type']=data['work_type']-1
data['insur_1'][data['insur_1']!=2]=0
data['insur_1'][data['insur_1']==2]=1
data['insur_2'][data['insur_2']!=2]=0
data['insur_2'][data['insur_2']==2]=1
data['insur_3'][data['insur_3']!=2]=0
data['insur_3'][data['insur_3']==2]=1
data['insur_4'][data['insur_4']!=2]=0
data['insur_4'][data['insur_4']==2]=1
data['car'][data['car']!=2]=0
data['car'][data['car']==2]=1
data['s_political'][data['s_political']!=4]=0
data['s_hukou'][data['s_hukou']!=1]=0
data['s_work_type']=data['s_work_type']-1
data['f_political'][data['f_political']!=4]=0
data['m_political'][data['m_political']!=4]=0

#跨度大的连续变量分区（提高计算速度）
data['income']=data['income'].apply(cut_income)

data['family_m'][data['family_m']==50]=5
data['family_m'][data['family_m']==-1]=0
data['family_m'][data['family_m']==-2]=2
data['family_m'][data['family_m']==-3]=9

data['house'][data['house']>5]=8
data['house'][data['house']==-1]=0
data['house'][data['house']==-2]=2
data['house'][data['house']==-3]=8

data['f_birth']=data['f_birth'].apply(cut_fm_yr)
data['m_birth']=data['m_birth'].apply(cut_fm_yr)
data['inc_exp']=data['inc_exp'].apply(cut_income)

data['public_service_1']=data['public_service_1'].apply(cut_service)
data['public_service_2']=data['public_service_2'].apply(cut_service)
data['public_service_3']=data['public_service_3'].apply(cut_service)
data['public_service_4']=data['public_service_4'].apply(cut_service)
data['public_service_5']=data['public_service_5'].apply(cut_service)
data['public_service_6']=data['public_service_6'].apply(cut_service)
data['public_service_7']=data['public_service_7'].apply(cut_service)
data['public_service_8']=data['public_service_8'].apply(cut_service)
data['public_service_9']=data['public_service_9'].apply(cut_service)


#对其他通用值数据进行处理（对于-8，选择中间2，3，5位的众数）
'''
#查看通用值的值,并决定替换值：全为-8
print(data['religion_freq'].value_counts())#4
print(data['edu'].value_counts())#4
print(data['health'].value_counts())#4
print(data['health_problem'].value_counts())#4
print(data['depression'].value_counts())#4
print(data['media_1'].value_counts())#2
print(data['media_2'].value_counts())#2
print(data['media_3'].value_counts())#2
print(data['media_4'].value_counts())#4
print(data['media_5'].value_counts())#4
print(data['media_6'].value_counts())#2
print(data['leisure_1'].value_counts())#2
print(data['leisure_2'].value_counts())#3
print(data['leisure_3'].value_counts())#3
print(data['leisure_4'].value_counts())#4
print(data['leisure_5'].value_counts())#4
print(data['leisure_6'].value_counts())#4
print(data['leisure_7'].value_counts())#4
print(data['leisure_8'].value_counts())#2
print(data['leisure_9'].value_counts())#4
print(data['leisure_10'].value_counts())#4
print(data['leisure_11'].value_counts())#4
print(data['leisure_12'].value_counts())#2
print(data['socialize'].value_counts())#2
print(data['relax'].value_counts())#4
print(data['learn'].value_counts())#2
print(data['socia_outing'].value_counts())#3
print(data['equity'].value_counts())#4
print(data['class'].value_counts())#5
print(data['class_10_before'].value_counts())#5
print(data['class_10_after'].value_counts())#5
print(data['class_14'].value_counts())#5
print(data['family_status'].value_counts())#3
print(data['son'].value_counts())#1
print(data['daughter'].value_counts())#1
print(data['f_edu'].value_counts())#1
print(data['f_work_14'].value_counts())#2
print(data['m_edu'].value_counts())#1
print(data['m_work_14'].value_counts())#2
print(data['status_peer'].value_counts())#2
print(data['status_3_before'].value_counts())#2
print(data['view'].value_counts())#4
print(data['inc_ability'].value_counts())#2
print(data['trust_1'].value_counts())#4
print(data['trust_2'].value_counts())#4
print(data['trust_3'].value_counts())#4
print(data['trust_4'].value_counts())#4
print(data['trust_5'].value_counts())#4
print(data['trust_6'].value_counts())#3
print(data['trust_7'].value_counts())#3
print(data['trust_8'].value_counts())#4
print(data['trust_9'].value_counts())#3
print(data['trust_10'].value_counts())#3
print(data['trust_11'].value_counts())#3
print(data['trust_12'].value_counts())#3
print(data['trust_13'].value_counts())#2
print(data['neighbor_familiarity'].value_counts())#4
'''
#替换通用值
data['religion_freq'][data['religion_freq']==-8]=4
data['edu'][data['edu']==-8]=4
data['health'][data['health']==-8]=4
data['health_problem'][data['health_problem']==-8]=4
data['depression'][data['depression']==-8]=4
data['media_1'][data['media_1']==-8]=2
data['media_2'][data['media_2']==-8]=2
data['media_3'][data['media_3']==-8]=2
data['media_4'][data['media_4']==-8]=4
data['media_5'][data['media_5']==-8]=4
data['media_6'][data['media_6']==-8]=2
data['leisure_1'][data['leisure_1']==-8]=2
data['leisure_2'][data['leisure_2']==-8]=3
data['leisure_3'][data['leisure_3']==-8]=3
data['leisure_4'][data['leisure_4']==-8]=4
data['leisure_5'][data['leisure_5']==-8]=4
data['leisure_6'][data['leisure_6']==-8]=4
data['leisure_7'][data['leisure_7']==-8]=4
data['leisure_8'][data['leisure_8']==-8]=2
data['leisure_9'][data['leisure_9']==-8]=4
data['leisure_10'][data['leisure_10']==-8]=4
data['leisure_11'][data['leisure_11']==-8]=4
data['leisure_12'][data['leisure_12']==-8]=2
data['socialize'][data['socialize']==-8]=2
data['relax'][data['relax']==-8]=4
data['learn'][data['learn']==-8]=2
data['socia_outing'][data['socia_outing']==-8]=3
data['equity'][data['equity']==-8]=4
data['class'][data['class']==-8]=5
data['class_10_before'][data['class_10_before']==-8]=5
data['class_10_after'][data['class_10_after']==-8]=5
data['class_14'][data['class_14']==-8]=5
data['family_status'][data['family_status']==-8]=3
data['daughter'][data['daughter']==-8]=1
data['son'][data['son']==-8]=1
data['f_edu'][data['f_edu']==-8]=1
data['f_work_14'][data['f_work_14']==-8]=2
data['m_edu'][data['m_edu']==-8]=1
data['m_work_14'][data['m_work_14']==-8]=2
data['status_peer'][data['status_peer']==-8]=2
data['status_3_before'][data['status_3_before']==-8]=2
data['view'][data['view']==-8]=4
data['inc_ability'][data['inc_ability']==-8]=2
data['trust_1'][data['trust_1']==-8]=4
data['trust_2'][data['trust_2']==-8]=4
data['trust_3'][data['trust_3']==-8]=4
data['trust_4'][data['trust_4']==-8]=4
data['trust_5'][data['trust_5']==-8]=4
data['trust_6'][data['trust_6']==-8]=3
data['trust_7'][data['trust_7']==-8]=3
data['trust_8'][data['trust_8']==-8]=4
data['trust_9'][data['trust_9']==-8]=3
data['trust_10'][data['trust_10']==-8]=3
data['trust_11'][data['trust_11']==-8]=3
data['trust_12'][data['trust_12']==-8]=3
data['trust_13'][data['trust_13']==-8]=2
data['neighbor_familiarity'][data['neighbor_familiarity']==-8]=4

#扩充特征(年龄，采访的年，月份，星期，星期类型)
#age
data['survey_time']=pd.to_datetime(data['survey_time'])
data['survey_time']=data['survey_time'].apply(change_time)
data['birth']=pd.to_datetime(data['birth'],format="%Y").dt.year
data['age']=data['survey_time'].dt.year-data['birth']

#year/month/weekday/weekday_type
data['year']=data['survey_time'].dt.year
data['month']=data['survey_time'].dt.month
data['weekday']=data['survey_time'].dt.weekday+1
data['weekday_type']=data['weekday'].apply(weekday_type)

#one-hot编码:'hukou_loc','work_status','work_manage','s_edu','s_work_exper','s_work_type',
#'f_edu','f_work_14','m_edu','m_work_14','work_exper','marital','province'
#hukou_loc
cols=['hukou_loc_1','hukou_loc_2','hukou_loc_3','hukou_loc_4']
data[cols]=pd.get_dummies(data['hukou_loc'])
#work_status
cols=['work_status_0','work_status_1','work_status_2','work_status_3','work_status_4','work_status_5','work_status_6','work_status_7','work_status_8','work_status_9']
data[cols]=pd.get_dummies(data['work_status'])
#work_manage
cols=['work_manage_0','work_manage_1','work_manage_2','work_manage_3','work_manage_4']
data[cols]=pd.get_dummies(data['work_manage'])
#s_edu
cols=['s_edu_0','s_edu_1','s_edu_2','s_edu_3','s_edu_4','s_edu_5','s_edu_6','s_edu_7','s_edu_8','s_edu_9','s_edu_10','s_edu_11','s_edu_12','s_edu_13','s_edu_14']
data[cols]=pd.get_dummies(data['s_edu'])
#s_work_exper
cols=['s_work_exper_0','s_work_exper_1','s_work_exper_2','s_work_exper_3','s_work_exper_4','s_work_exper_5','s_work_exper_6']
data[cols]=pd.get_dummies(data['s_work_exper'])
#s_work_type
cols=['s_work_type_-1','s_work_type_0','s_work_type_1']
data[cols]=pd.get_dummies(data['s_work_type'])
#f_edu
cols=['f_edu_1','f_edu_2','f_edu_3','f_edu_4','f_edu_5','f_edu_6','f_edu_7','f_edu_8','f_edu_9','f_edu_10','f_edu_11','f_edu_12','f_edu_13','f_edu_14']
data[cols]=pd.get_dummies(data['f_edu'])
#f_work_14
cols=['f_work_14_1','f_work_14_2','f_work_14_3','f_work_14_4','f_work_14_5','f_work_14_6','f_work_14_7','f_work_14_8','f_work_14_9','f_work_14_10','f_work_14_11','f_work_14_12','f_work_14_13','f_work_14_14','f_work_14_15','f_work_14_16','f_work_14_17']
data[cols]=pd.get_dummies(data['f_work_14'])
#m_edu
cols=['m_edu_1','m_edu_2','m_edu_3','m_edu_4','m_edu_5','m_edu_6','m_edu_7','m_edu_8','m_edu_9','m_edu_10','m_edu_11','m_edu_12','m_edu_13','m_edu_14']
data[cols]=pd.get_dummies(data['m_edu'])
#m_work_14
cols=['m_work_14_1','m_work_14_2','m_work_14_3','m_work_14_4','m_work_14_5','m_work_14_6','m_work_14_7','m_work_14_8','m_work_14_9','m_work_14_10','m_work_14_11','m_work_14_12','m_work_14_13','m_work_14_14','m_work_14_15','m_work_14_16','m_work_14_17']
data[cols]=pd.get_dummies(data['m_work_14'])
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

#删除变量:'hukou_loc','work_status','work_manage','s_edu','s_work_exper','s_work_type',
#'f_edu','f_work_14','m_edu','m_work_14','work_exper','marital','province',
#'id','city','county','survey_time','birth'
data.drop(columns=['hukou_loc','work_status','work_manage','s_edu','s_work_exper','s_work_type','f_edu','f_work_14','m_edu','m_work_14','work_exper','marital','province','id','city','county','survey_time','birth'],inplace=True)



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
P=np.zeros(len(feature_train))

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
































