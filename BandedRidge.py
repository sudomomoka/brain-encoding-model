# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import pickle
import os
import sys
import math
import time
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from himalaya.ridge import GroupRidgeCV
from himalaya.ridge import ColumnTransformerNoStack
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

from scipy import stats


#from correlation import correlation_c

def correlation_c(y_test_T,y_pre_T):
    corr_list = []
    for j in range(y_test_T.shape[0]):
        corr = np.corrcoef(y_test_T[j, :],y_pre_T[j, :])[0,1]
        #n = y_test_T[j, :].shape[0]
        #t = np.dot(corr, np.sqrt(n - 2)) / np.sqrt(1 - np.power(corr, 2))
        #p = (1 - stats.t.cdf(np.absolute(t), n - 2)) * 2
        #if p<0.05:
        #    corr_list.append(corr)
        #else:
        #    corr_list.append(0)

    #print(corr_list)
        corr_list.append(corr)
    #ave_corr = np.average(corr_list)
    #print("相関係数: {}".format(corr_list))

    return corr_list


def pair_corr_with_p(s1, s2):
    cc=np.corrcoef(s1, s2)[0,1]
    n = s1.shape[0]
    t = np.dot(cc, np.sqrt(n - 2)) / np.sqrt(1 - np.power(cc, 2))
    p = (1 - stats.t.cdf(np.absolute(t), n - 2)) * 2
    if p<0.05:
        return cc
    else:
        return 0


begin = time.time()
r=4
width = 5
print("width:",width)
width = width - 1


data2=[]
data3=[]
bert_data1 = []

"""
with open("../../dvd-prepro/data/pickle_data/Crown_sentence.pickle",'rb') as f:
    crown_f = pickle.load(f)
    #data2 = np.vstack([data2,data1])
#    print("data shape:",data2.shape)
print("Crown shape:",crown_f.shape)
"""
with open("../../dvd-prepro/src/話し言葉_dvd/Glee_feature",'rb') as f:
#with open("../../dvd-prepro/data/sentence_pickle_data/Glee_sentence.pickle",'rb') as f:
    glee_f1 = pickle.load(f)
    #data2 = data1
print("Glee shape:",glee_f1.shape)

with open("../../dvd-prepro/src/話し言葉_dvd/DreamGirls_feature",'rb') as f:
#with open("../../dvd-prepro/data/sentence_pickle_data/DreamGirls_sentence.pickle",'rb') as f:
    dreamgirls_f1 = pickle.load(f)
    #data2 = np.vstack([data2,data1])
print("DreamGirls shape:",dreamgirls_f1.shape)

#with open("../../dvd-prepro/data/sentence_pickle_data/Mentalist_sentence.pickle",'rb') as f:
with open("../../dvd-prepro/src/話し言葉_dvd/Mentalist_feature",'rb') as f:
    mentalist_f1 = pickle.load(f)
    #data2 = np.vstack([data2,data1])
print("Mentalist shape:",mentalist_f1.shape)
"""
with open("../../dvd-prepro/src/話し言葉_dvd/Heroes_feature",'rb') as f:
#with open("../../dvd-prepro/data/sentence_pickle_data/Heroes_sentence.pickle",'rb') as f:
    heroes_f = pickle.load(f)[4:]
    #data2 = np.vstack([data2,data1])
print("Heroes shape:",heroes_f.shape)
"""

with open("../../dvd-prepro/src/話し言葉_dvd/GIS2_feature",'rb') as f:
#with open("../../dvd-prepro/data/sentence_pickle_data/GIS2_sentence.pickle",'rb') as f:
    gis2_f1 = pickle.load(f)
    #data2 = np.vstack([data2,data1])
print("GIS2 shape:",gis2_f1.shape)

with open("../../dvd-prepro/src/話し言葉_dvd/GIS1_feature",'rb') as f:
#with open("../../dvd-prepro/data/sentence_pickle_data/GIS1_sentence.pickle",'rb') as f:
    gis1_f1 = pickle.load(f)
    #data2 = np.vstack([data2,data1])
print("GIS1 shape:",gis1_f1.shape)



#テストデータ
x=dreamgirls_f1

#訓練データ
data2 = gis2_f1
data2 = np.vstack([data2,glee_f1])
data2 = np.vstack([data2,mentalist_f1])
data2 = np.vstack([data2,gis1_f1])
#data2 = np.vstack([data2,heroes_f1])
#data2 = np.vstack([data2,crown_f1])

data2 = np.array(data2)

for i in range(len(data2)-width):
    data3.append(data2[i])
    data3.append(data2[i+1])
    data3.append(data2[i+2])
    data3.append(data2[i+3])
    data3.append(data2[i+4])
    data3 = np.array(data3)
    data3 = data3.flatten()
    bert_data1.append(data3)
    #print(len(bert_data))
    data3=[]


test_x1=[]

for i in range(len(x)-width):
    data3.append(x[i])
    data3.append(x[i+1])
    data3.append(x[i+2])
    data3.append(x[i+3])
    data3.append(x[i+4])
    data3 = np.array(data3)
    data3 = data3.flatten()
    test_x1.append(data3)
    #print(len(bert_data))
    data3=[]

test_x1 = np.array(test_x1)
bert_data1 = np.array(bert_data1)
print(bert_data1.shape)
print(test_x1.shape)
#bert_data = bert_data.T
#bert_data = bert_data[:-2,:]#なんか微妙に合わない…
#print("test_bert: ",test_x.shape)
#print("bert_dataの形状:",bert_data.shape)
#print("word2vec_dataの形状:",bert_data.shape)



data2=[]
data3=[]
bert_data2 = []


with open("../../dvd-prepro/data/sentence_pickle_data/Glee_sentence.pickle",'rb') as f: 
    glee_f2 = pickle.load(f)
    #data2 = data1
print("Glee shape:",glee_f2.shape)

with open("../../dvd-prepro/data/sentence_pickle_data/DreamGirls_sentence.pickle",'rb') as f:
    dreamgirls_f2 = pickle.load(f)
    #data2 = np.vstack([data2,data1])
print("DreamGirls shape:",dreamgirls_f2.shape)

with open("../../dvd-prepro/data/sentence_pickle_data/Mentalist_sentence.pickle",'rb') as f:
    mentalist_f2 = pickle.load(f)
    #data2 = np.vstack([data2,data1])
print("Mentalist shape:",mentalist_f2.shape)

with open("../../dvd-prepro/data/sentence_pickle_data/GIS1_sentence.pickle",'rb') as f:
    gis1_f2 = pickle.load(f)
    #data2 = np.vstack([data2,data1])
print("GIS1 shape:",gis1_f2.shape)

with open("../../dvd-prepro/data/sentence_pickle_data/GIS2_sentence.pickle",'rb') as f:
    gis2_f2 = pickle.load(f)
    #data2 = np.vstack([data2,data1])
print("GIS2 shape:",gis2_f2.shape)


#テストデー
x=dreamgirls_f2
#訓練デー 
data2 = gis2_f2
data2 = np.vstack([data2,glee_f2])
data2 = np.vstack([data2,mentalist_f2])
data2 = np.vstack([data2,gis1_f2])
#data2 = np.vstack([data2,heroes_f2])
#data2 = np.vstack([data2,crown_f2])
data2 = np.array(data2)


for i in range(len(data2)-width):
    data3.append(data2[i])
    data3.append(data2[i+1])
    data3.append(data2[i+2])
    data3.append(data2[i+3])
    data3.append(data2[i+4])
    data3 = np.array(data3)
    data3 = data3.flatten()
    bert_data2.append(data3)
    #print(len(bert_data))
    data3=[]


test_x2=[]


for i in range(len(x)-width):
    data3.append(x[i])
    data3.append(x[i+1])
    data3.append(x[i+2])
    data3.append(x[i+3])
    data3.append(x[i+4])
    data3 = np.array(data3)
    data3 = data3.flatten()
    test_x2.append(data3)
    #print(len(bert_data))
    data3=[]

test_x2 = np.array(test_x2)
bert_data2 = np.array(bert_data2)
print(test_x2.shape)
print(bert_data2.shape)


print("Train::: time_features :",bert_data1.shape)
print("Train::: Sentece[CLS] :",bert_data2.shape)

print("Test::: time_features :",test_x1.shape)
print("Test::: Sentece[CLS] :",test_x2.shape)


bert_data = np.block([bert_data1, bert_data2])
#test_x = np.block([test_x1, test_x2])
t=np.zeros((test_x1.shape[0],test_x1.shape[1]))
c=np.zeros((test_x2.shape[0],test_x2.shape[1]))

print(t.shape)
print(c.shape)

test_x_time = np.block([test_x1,c])
test_x_cls = np.block([t,test_x2])
print(test_x_time.shape)
print(test_x_cls.shape)

#exit()
test_y =[]


for i in range(9):
    with open("../data/DM03/DreamGirls/DreamGirls_brain_run" + str(i+1)+".pickle",'rb') as f2:
        data = pickle.load(f2)
        #print(data.shape)

    if i+1 == 1:
        dreamgirls_b = data
        #print(brain_data.shape)
    #print(data.shape)
    else:
        dreamgirls_b = np.vstack([dreamgirls_b,data])
        #brain_data = np.vstack([brain_data,data])
#test_y = test_y[:-1]

print("DreamGirls brain",dreamgirls_b.shape)

a=0


for i in range(4):
    with open("../data/DM03/Mentalist/Mentalist_brain_run" + str(i+1) + ".pickle",'rb') as f2:
        data = pickle.load(f2)
    if i+1 == 1:
        mentalist_b = data
        #print(brain_data.shape)
    else:
        mentalist_b = np.vstack([mentalist_b,data])
        #print(brain_data.shape)
print("Mentalist brain",mentalist_b.shape)
"""
for i in range(5):
    with open("../data/DM01/Crown/Crown_brain_run" + str(i+1) + ".pickle",'rb') as f2:
        data = pickle.load(f2)
    if i+1 == 1:
        crown_b = data
    else:
        crown_b = np.vstack([crown_b,data])
print("Crown brain",crown_b.shape)
"""

for i in range(4):
    with open("../data/DM03/Glee/Glee_brain_run" + str(i+1) + ".pickle", 'rb') as f2:
        data=pickle.load(f2)
    if i+1 == 1:
        glee_b = data
    else:
        glee_b = np.vstack([glee_b,data])
print("Glee brain",glee_b.shape)
"""
for i in range(5):
    with open("../data/DM01/Heroes/Heroes_brain_run" + str(i+1) + ".pickle",'rb') as f2:
        data = pickle.lcoad(f2)
    if i+1 == 1:
        heroes_b = data
    else:
        heroes_b = np.vstack([heroes_b,data])
heroes_b = heroes_b[:-4]
print("Heroes brain",heroes_b.shape)
"""
for i in range(2):
    with open("../data/DM03/GIS2/GIS2_brain_run" + str(i+1) + ".pickle",'rb') as f2:
        data = pickle.load(f2)
    if i+1 == 1:
        gis2_b = data
    else:
        gis2_b = np.vstack([gis2_b,data])
print("GIS2 brain",gis2_b.shape)

for i in range(2):
    with open("../data/DM03/GIS1/GIS1_brain_run" + str(i+1) + ".pickle",'rb') as f2:
        data = pickle.load(f2)
    if i+1 == 1:
        gis1_b = data
    else:
        gis1_b = np.vstack([gis1_b,data])

print("GIS1 brain",gis1_b.shape)



#テストデータ
test_y = dreamgirls_b
#訓練データ
brain_data = gis2_b
brain_data = np.vstack([brain_data,glee_b])
brain_data = np.vstack([brain_data,mentalist_b])
brain_data = np.vstack([brain_data,gis1_b])
#brain_data = np.vstack([brain_data,heroes_b])
#brain_data = np.vstack([brain_data,crown_b])



#brain_data = brain_data.T
#brain_data = brain_data[23:-23,:]#なんか微妙に合わない…
brain_data = brain_data[4:,:]
test_y = test_y[4:,:]
print("brain_dataの形状:",brain_data.shape)


def set_trn_tst_cv_samples(nSamples, tstRatio, nCV):
    # Trn/ Tst
    nSamples_trn = int(np.round(nSamples* (1-tstRatio)))
    samples_trn = np.array(range(nSamples_trn), int)
    samples_tst = np.array(range(nSamples_trn, nSamples), int)
    # CV
    nSamples_cv = int(np.round(nSamples_trn/nCV))
    samples_cv = list(np.array_split(samples_trn, nCV))
    return samples_trn, samples_tst, samples_cv


#g1:time_features
#g2:sentence_features
g1 = []
g2 = []
for i in range(len(bert_data1[1])):
    g1.append(i)

#print(g1)
print(len(g1))

print(len(bert_data1[1]), len(bert_data[1]))

ct = ColumnTransformerNoStack([("group_1", StandardScaler(), g1),("group_2", StandardScaler(), slice(len(bert_data1[1]), len(bert_data[1])))])


y_train = brain_data
y_test = test_y
X_train = bert_data
X_test_time = test_x_time
X_test_cls = test_x_cls
#exit()
data_loading_end = time.time()

print("データロード時間:",data_loading_end - begin)
ridge_start = time.time()


# Train the ridge regression
def cross_validate(train_x_all,train_y_all,alpha,split_size=5):
  results = [0 for _ in range(train_y_all.shape[1])]
  kf = KFold(n_splits=split_size)
  for train_idx, val_idx in kf.split(train_x_all, train_y_all):
    train_x = train_x_all[train_idx]
    train_y = train_y_all[train_idx]
    val_x = train_x_all[val_idx]
    val_y = train_y_all[val_idx]

    reg = Ridge(alpha=a).fit(train_x,train_y)
    pre_y = reg.predict(val_x)

    y_val_T = val_y.T
    y_pre_T = pre_y.T

    #print("y_test_T shape:",y_val_T.shape)
    #print("y_pre_T shape:",y_pre_T.shape)

    k_fold_r = correlation_c(y_val_T,y_pre_T)
    #print("aaa")
    results = [x + y for (x, y) in zip(results, k_fold_r)]

  results = map(lambda x : x/5,results)
  results = list(results)
  #print(results)
  return results


model = GroupRidgeCV(groups="input")
pipe = make_pipeline(ct, model)
reg = pipe.fit(X_train,y_train)


filename = 'DM03_bandedridge_model_dreamgirls.sav'
pickle.dump(reg, open(filename, 'wb'))

#exit()
##################################################################################################

print(filename)
reg = pickle.load(open(filename, 'rb'))

y_pre_time =reg.predict(X_test_time)
y_pre_cls =reg.predict(X_test_cls)
print(y_pre_time.shape)
print(y_pre_cls.shape)

ridge_end = time.time()

print("回帰時間:",ridge_end - ridge_start)

calculation_start = time.time()



y_test_T = y_test.T

#y_pre_time = y_pre[0:len(bert_data1[1])]
#y_pre_cls = y_pre[len(bert_data1[1]):len(bert_data[1])]

print(y_pre_time.shape)
print(y_pre_cls.shape)

#exit()
y_pre_time_T = y_pre_time.T
y_pre_cls_T = y_pre_cls.T
print(y_pre_time_T.shape)
print(y_pre_cls_T.shape)


#print(len(best_parameters))
counter = 0
y_pred_all = []
corr_list_time = []
corr_list_cls = []

vectore = []

for counter in range(y_test.shape[1]):
    corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre_time_T[counter, :])
    vectore.append(y_pre_time_T[counter,:])
    corr_list_time.append(corr_with_p)

for counter in range(y_test.shape[1]):
    corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre_cls_T[counter, :])
    vectore.append(y_pre_cls_T[counter,:])
    corr_list_cls.append(corr_with_p)
    
    
corr_list_time= np.array(corr_list_time)
corr_list_cls= np.array(corr_list_cls)

vectore = np.array(vectore)
np.set_printoptions(threshold=1000)

#print(corr_list)
print("corr_time:",len(corr_list_time))
print("corr_cls:",len(corr_list_cls))

num_time = len(corr_list_time) - np.count_nonzero(corr_list_time)
num_cls = len(corr_list_cls) - np.count_nonzero(corr_list_cls)
print(num_time)
print(num_cls)

#corrs = list(corrs)
#corrs = np.array(corrs)
#corr_t = tuple(corr_list)

#alpha_graph(alphas,corr_t, corrs, b_alpha, 5, os.getcwd(), save=True)

calculation_end = time.time()
print("計算時間:",calculation_end - calculation_start)


save_start = time.time()

np.save('../話し言葉_dvd_result/DM03/Banded_DreamGirls_time_correlation',corr_list_time)
np.save('../話し言葉_dvd_result/DM03/Banded_DreamGirls_cls_correlation',corr_list_cls)

#np.save('../bert_sen_ridge_result/DM07/pre_GIS2_correlation',corr_list)
#np.save('../src/DM03/Y_pred/Mentalist_pred',  vectore)
print(corr_list_time.shape)
print(corr_list_cls.shape)
save_end = time.time()

#print("保存時間:",save_end - save_start)

#print("全工程時間:",save_end - begin)
