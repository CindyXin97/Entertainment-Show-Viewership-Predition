#/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Tue June 27 17:56:04 2019

@author: CindyXin
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import warnings
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

#忽视警告
warnings.filterwarnings('ignore')

#明星档次
#def star_level(v_star):
#    if float(v_star) <10000:
#        return('0')
#    elif float(v_star)>=10000 and float(v_star)<35000:
#        return('1')
#    elif float(v_star)>=35000 and float(v_star)<75000:
#        return('2')
#    else:
#        return('3')

#竞争因素
def compute_level(v_compute):
    if float(v_compute) <2.132576:
        return('0')
    elif float(v_compute)>=2.132576 and float(v_compute)<3.000000:
        return('1')
    elif float(v_compute)>=3.000000 and float(v_compute)<4.458042:
        return('2')
    else:
        return('3') 

def SVC_RandomForestClassifier(v_tvornet,v_type,v_appname,v_prod_cost,v_pub_cost,v_star,v_period,v_n,v_dupin,v_compete):
    df1 = pd.read_excel('/data2/research_dep01/svc/data1.xlsx')#数据读取路径

#训练模型
    df1[['tvornet']] = df1[['tvornet']].replace([u'电视综艺',u'纯网综艺'],['0','1'])
    df1[['type']] = df1[['type']].replace([u'才艺选秀类',u'美食类',u'明星竞演类',u'其他',u'亲子/儿童互动',u'生活观察类',u'谈话/脱口秀类',u'文化创意类',u'喜剧类',u'综合游戏类'],['0','1','2','3','4','5','6','7','8','9'])
    df1[['appname']] = df1[['appname']].replace([u'hunantv',u'iqiyi',u'qq',u'sohu',u'youku'],['0','1','2','3','4'])
    df1[['period']] = df1[['period']].replace([u'贺岁档',u'平时',u'暑期档'],['0','1','2'])
    df1[['dupin']] = df1[['dupin']].replace([u'独播',u'非独播'],['0','1'])


    df1.loc[df1['prod_cost'] < 3000,'prod_cost'] = 0
    df1.loc[(df1['prod_cost'] >= 3000)&(df1['prod_cost'] < 6000),'prod_cost'] = 1
    df1.loc[(df1['prod_cost'] >= 6000)&(df1['prod_cost'] < 8250),'prod_cost'] = 2
    df1.loc[df1['prod_cost'] >= 8250,'prod_cost'] = 3

    df1.loc[df1['pub_cost'] < 100,'pub_cost'] = 0
    df1.loc[(df1['pub_cost'] >= 100)&(df1['pub_cost'] < 300),'pub_cost'] = 1
    df1.loc[(df1['pub_cost'] >= 300)&(df1['pub_cost'] < 800),'pub_cost'] = 2
    df1.loc[df1['pub_cost'] >= 800,'pub_cost'] = 3

    df1.loc[df1['star'] < 10923,'star'] = 0
    df1.loc[(df1['star'] >= 10923)&(df1['star'] < 34221),'star'] = 1
    df1.loc[(df1['star'] >= 34221)&(df1['star'] < 74090),'star'] = 2
    df1.loc[df1['star'] >= 74090,'star'] = 3

    df1.loc[df1['compete'] < 2.132576,'compete'] = 0
    df1.loc[(df1['compete'] >= 2.132576)&(df1['compete'] < 3.000000),'compete'] = 1
    df1.loc[(df1['compete'] >= 3.000000)&(df1['compete'] < 4.458042),'compete'] = 2
    df1.loc[df1['compete'] >= 4.458042,'compete'] = 3

    x = df1[['appname','compete','dupin','n','period','prod_cost','pub_cost','star','tvornet','type']]
    y = df1[['class']]

    smo = SMOTE(random_state=42)

    X_smo, y_smo = smo.fit_sample(x, y)

    y_smo = DataFrame(y_smo,columns = ['class'])

    rf0 = RandomForestClassifier(oob_score=True, random_state=20, n_estimators = 80, max_depth=8)
    rf0.fit(X_smo, y_smo)

#x_test = df1.loc[df1['is_train'] != 1,['tvornet','type','appname','prod_cost','pub_cost','star','period','n','dupin','compete']]
# print(x_test)
    x_test=pd.DataFrame([{'tvornet':str(v_tvornet),'type':str(v_type),'appname':str(v_appname),'prod_cost':str(v_prod_cost),'pub_cost':str(v_pub_cost),'star':float(v_star),'period':str(v_period),'n':str(v_n),'dupin':str(v_dupin),'compete':float(v_compete)}])
# print(x_test)
    y_pred = list(rf0.predict(x_test))
    print(y_pred[0])


if __name__ == '__main__':
    v_tvornet=sys.argv[1].split("-")[0].replace(u'电视综艺','0').replace(u'纯网综艺','1')
    v_type=sys.argv[1].split("-")[1].replace(u'才艺选秀类','0').replace(u'美食类','1').replace(u'明星竞演类','2').replace(u'其他','3').replace(u'亲子/儿童互动','4').replace(u'生活观察类','5').replace(u'谈话/脱口秀类','6').replace(u'文化创意类','7').replace(u'喜剧类','8').replace(u'综合游戏类','9')
    v_appname=sys.argv[1].split("-")[2].replace('芒果','0').replace(u'爱奇艺','1').replace(u'腾讯视频','2').replace(u'搜狐','3').replace(u'优酷','4')
    v_prod_cost=sys.argv[1].split("-")[3].replace(u'小于3000万','0').replace(u'3000万~6000万','1').replace(u'6000万~8000万','2').replace(u'大于8000万','3')
    v_pub_cost=sys.argv[1].split("-")[4].replace(u'小于100万','0').replace(u'100万~300万','1').replace(u'300万~800万','2').replace(u'大于800万','3')
    v_star=sys.argv[1].split("-")[5].replace(u'小于10000','0').replace(u'10000~35000','1').replace(u'35000~75000','2').replace(u'大于75000','3')
    v_period=sys.argv[1].split("-")[6].replace(u'贺岁档','0').replace(u'平时','1').replace(u'暑期档','2')
    v_n=sys.argv[1].split("-")[7].replace(u'否','0').replace(u'是','1')
    v_dupin=sys.argv[1].split("-")[8].replace(u'独播','0').replace(u'非独播','1')
    v_compete=sys.argv[1].split("-")[9]
    SVC_RandomForestClassifier(v_tvornet,v_type,v_appname,v_prod_cost,v_pub_cost,v_star,v_period,v_n,v_dupin,v_compete)
