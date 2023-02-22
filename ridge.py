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
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

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

#alpha = 100.0

#a_list = [0.5,1.0,5.0]
a_list = [0.5,1.0,5.0,10.0,10.0**2,10.0**3,10.0**4,2.5*(10.0**4),5.0*(10.0**4),10.0**5,10.0**6,10.0**7]
alphas = (0.5,1.0,5.0,10.0,10.0**2,10.0**3,10.0**4,2.5*(10.0**4),5.0*(10.0**4),10.0**5,10.0**6,10.0**7)


alpha1 = (0.5)
alpha2= (1.0)
alpha3 = (5.0)
alpha4 = (10.0)
alpha5 = (10.0**2)
alpha6 = (10.0**3)
alpha7 = (10.0**4)
alpha8 = 2.5*(10.0**4)
alpha9 = 5.0*(10.0**4)
alpha10 = (10.0**5)
alpha11 = (10.0**6)
alpha12 = (10.0**7)

#print("alpha:",alpha)

data2=[]
data3=[]
bert_data = []

"""
with open("../../dvd-prepro/data/pickle_data/Crown_sentence.pickle",'rb') as f:
    crown_f = pickle.load(f)
    #data2 = np.vstack([data2,data1])
#    print("data shape:",data2.shape)
print("Crown shape:",crown_f.shape)
"""
#with open("../../dvd-prepro/data/話し言葉_dvd_funabiki/Glee_feature",'rb') as f:
#with open("../../dvd-prepro/data/sentence_pickle_data/Glee_sentence.pickle",'rb') as f:
with open("../../dvd-prepro/data/only_dvd_data/Glee_feature",'rb') as f:
    glee_f = pickle.load(f)
    #data2 = data1
print("Glee shape:",glee_f.shape)

#with open("../../dvd-prepro/data/話し言葉_dvd_funabiki/DreamGirls_feature",'rb') as f:
#with open("../../dvd-prepro/data/sentence_pickle_data/DreamGirls_sentence.pickle",'rb') as f:
with open("../../dvd-prepro/data/only_dvd_data/DreamGirls_feature",'rb') as f:
    dreamgirls_f = pickle.load(f)
    #data2 = np.vstack([data2,data1])
print("DreamGirls shape:",dreamgirls_f.shape)

#with open("../../dvd-prepro/data/sentence_pickle_data/Mentalist_sentence.pickle",'rb') as f:
#with open("../../dvd-prepro/data/話し言葉_dvd_funabiki/Mentalist_feature",'rb') as f:
with open("../../dvd-prepro/data/only_dvd_data/Mentalist_feature",'rb') as f:
    mentalist_f = pickle.load(f)
    #data2 = np.vstack([data2,data1])
print("Mentalist shape:",mentalist_f.shape)
"""
with open("../../dvd-prepro/src/話し言葉_dvd/Heroes_feature",'rb') as f:
#with open("../../dvd-prepro/data/sentence_pickle_data/Heroes_sentence.pickle",'rb') as f:
    heroes_f = pickle.load(f)[4:]
    #data2 = np.vstack([data2,data1])
print("Heroes shape:",heroes_f.shape)
"""

#with open("../../dvd-prepro/data/話し言葉_dvd_funabiki/GIS2_feature",'rb') as f:
#with open("../../dvd-prepro/data/sentence_pickle_data/GIS2_sentence.pickle",'rb') as f:
with open("../../dvd-prepro/data/only_dvd_data/GIS2_feature",'rb') as f:
    gis2_f = pickle.load(f)
    #data2 = np.vstack([data2,data1])
print("GIS2 shape:",gis2_f.shape)

#with open("../../dvd-prepro/data/話し言葉_dvd_funabiki/GIS1_feature",'rb') as f:
#with open("../../dvd-prepro/data/sentence_pickle_data/GIS1_sentence.pickle",'rb') as f:
with open("../../dvd-prepro/data/only_dvd_data/GIS1_feature",'rb') as f:
    gis1_f = pickle.load(f)
    #data2 = np.vstack([data2,data1])
print("GIS1 shape:",gis1_f.shape)



#テストデータ
x=gis2_f

#訓練データ
data2 = gis1_f
data2 = np.vstack([data2,glee_f])
data2 = np.vstack([data2,mentalist_f])
data2 = np.vstack([data2,dreamgirls_f])
#data2 = np.vstack([data2,heroes_f])
#data2 = np.vstack([data2,crown_f])

data2 = np.array(data2)

"""
for i in range(len(data2)-width):
    sum = data2[i] +  data2[i+1] +  data2[i+2] +  data2[i+3] +  data2[i+4]
    ave = sum / 5
    bert_data.append(ave)
"""
"""
#data1 = data1[:-4,:]
#print(data1[2])
#print(type(data1[2][0]))

"""
"""
for i in range(len(data2)-width):
    if np.all(data2[i]!=0.)== True and np.all(data2[i+1]!=0.)== True and np.all(data1[i+2]!=0.)== True and np.all(data1[i+3]!=0.)== True:
        data3.append(data2[i])
        data3.append(data2[i+1])
        data3.append(data2[i+2])
        data3.append(data2[i+3])
        data3.append(data2[i+4])
        data3 = np.array(data3)
        data2 = data2.flatten()
        #print(data2.shape)
        bert_data.append(data2)
        data2=[]
"""

for i in range(len(data2)-width):
    data3.append(data2[i])
    data3.append(data2[i+1])
    data3.append(data2[i+2])
    data3.append(data2[i+3])
    data3.append(data2[i+4])
    data3 = np.array(data3)
    data3 = data3.flatten()
    bert_data.append(data3)
    #print(len(bert_data))
    data3=[]


test_x=[]

for i in range(len(x)-width):
    data3.append(x[i])
    data3.append(x[i+1])
    data3.append(x[i+2])
    data3.append(x[i+3])
    data3.append(x[i+4])
    data3 = np.array(data3)
    data3 = data3.flatten()
    test_x.append(data3)
    #print(len(bert_data))
    data3=[]

test_x = np.array(test_x)
bert_data = np.array(bert_data)
print(bert_data.shape)
#bert_data = bert_data.T
#bert_data = bert_data[:-2,:]#なんか微妙に合わない…
#print("test_bert: ",test_x.shape)
#print("bert_dataの形状:",bert_data.shape)
#print("word2vec_dataの形状:",bert_data.shape)


test_y =[]


for i in range(9):
    with open("../data/DM01/DreamGirls/DreamGirls_brain_run" + str(i+1)+".pickle",'rb') as f2:
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
    with open("../data/DM01/Mentalist/Mentalist_brain_run" + str(i+1) + ".pickle",'rb') as f2:
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
    with open("../data/DM01/Glee/Glee_brain_run" + str(i+1) + ".pickle", 'rb') as f2:
        data=pickle.load(f2)
    if i+1 == 1:
        glee_b = data
    else:
        glee_b = np.vstack([glee_b,data])
print("Glee brain",glee_b.shape)
"""
for i in range(5):
    with open("../data/DM01/Heroes/Heroes_brain_run" + str(i+1) + ".pickle",'rb') as f2:
        data = pickle.load(f2)
    if i+1 == 1:
        heroes_b = data
    else:
        heroes_b = np.vstack([heroes_b,data])
heroes_b = heroes_b[:-4]
print("Heroes brain",heroes_b.shape)
"""
for i in range(2):
    with open("../data/DM01/GIS2/GIS2_brain_run" + str(i+1) + ".pickle",'rb') as f2:
        data = pickle.load(f2)
    if i+1 == 1:
        gis2_b = data
    else:
        gis2_b = np.vstack([gis2_b,data])
print("GIS2 brain",gis2_b.shape)

for i in range(2):
    with open("../data/DM01/GIS1/GIS1_brain_run" + str(i+1) + ".pickle",'rb') as f2:
        data = pickle.load(f2)
    if i+1 == 1:
        gis1_b = data
    else:
        gis1_b = np.vstack([gis1_b,data])

print("GIS1 brain",gis1_b.shape)



#テストデータ
test_y = gis2_b
#訓練データ
brain_data = gis1_b
brain_data = np.vstack([brain_data,glee_b])
brain_data = np.vstack([brain_data,mentalist_b])
brain_data = np.vstack([brain_data,dreamgirls_b])
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



tstRatio = 0.2
nCV = 5 # = 5-fold cross validation
nSamples = brain_data.shape[0]
samples_trn, samples_tst, samples_cv = set_trn_tst_cv_samples(nSamples, tstRatio, nCV)
# resp, stim
#y_train = brain_data[samples_trn, :]
y_train = brain_data
#y_test = brain_data[samples_tst, :]
y_test = test_y
#X_train = bert_data[samples_trn, :]
X_train = bert_data
#X_test = bert_data[samples_tst, :]
X_test = test_x

"""
d_n = bert_data.shape[0]
print(d_n)

X_train=bert_data[:(d_n//4)*3,:]
X_test=bert_data[-(d_n//4):,:]

y_train=brain_data[:(d_n//4)*3,:]
y_test =brain_data[-(d_n//4):,:]
"""
print("X_train shape:",X_train.shape)
print("y_train shape:",y_train.shape)
print("X_test shape:",X_test.shape)
print("y_test shape:",y_test.shape)
#print(type(X_test))

"""
X_train,X_test,y_train,y_test = train_test_split(bert_data, brain_data, train_size=0.75, random_state=0)

print("X_train shape:",X_train.shape)
print("y_train shape:",y_train.shape)
print("X_test shape:",X_test.shape)
print("y_test shape:",y_test.shape)
"""


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



best_co = [-100 for _ in range(y_train.shape[1])]
"""
best_parameters = [0 for _ in range(y_train.shape[1])]

for a in a_list:
    print("alpha:",a)
    c = cross_validate(X_train,y_train,a,5)
    print(len(c))
    for i in range(len(c)):
        if c[i] > best_co[i]:
            best_co[i] = c[i]
            best_parameters[i] = a
            #print(best_co[i])
            #print(best_parameters[i])

#print(best_parameters)
print(len(best_parameters))
print(len(best_co))

f = open('best_para.txt', 'wb')
pickle.dump(best_parameters, f)
#exit()
"""

f = open("./best_para.txt","rb")
best_parameters = pickle.load(f)

b_alpha = tuple(best_parameters)

def alpha_graph(alphas, corr_t,corrs, b_alpha, n_cv, save_path, save=False):
  """
  alphaのグラフを可視化する
  <<parameters>>
  alphas: (a, ) アルファ達
  corrs: (a, ) それぞれのアルファの平均の相関係数
  save: (bool) Trueならばグラフを保存する
  """
  fig = plt.figure()
  #mcc = corr_list[np.where(alphas==b_alpha)]
  #plt.title('average corr coef in training(CV:{})best alpha: {:.2e}, mean corr coef: {:.5f}'.format(n_cv, b_alpha, mcc))
  plt.xlabel('alpha')
  plt.ylabel('average correlation coefficient')
  #plt.plot(alphas, corrs, marker='o')
  plt.scatter(b_alpha, corr_t, color='red', marker='o')
  plt.plot(alphas, corrs, marker='o')
  plt.xscale('log')
  plt.minorticks_on()
  plt.grid(axis='x')
  plt.grid(which='both', axis='y', ls='--')
  if save:
    fig_path = save_path + '/Fig/alpha_view/'
    if not os.path.exists(fig_path):
      os.makedirs(fig_path)
    figfile = fig_path + 'alpha.png'
    plt.savefig(figfile)


    
reg1 = Ridge(alpha=alpha1).fit(X_train, y_train)
reg2 = Ridge(alpha=alpha2).fit(X_train, y_train)
reg3 = Ridge(alpha=alpha3).fit(X_train, y_train)
reg4 = Ridge(alpha=alpha4).fit(X_train, y_train)
reg5 = Ridge(alpha=alpha5).fit(X_train, y_train)
reg6 = Ridge(alpha=alpha6).fit(X_train, y_train)
reg7 = Ridge(alpha=alpha7).fit(X_train, y_train)
reg8 = Ridge(alpha=alpha8).fit(X_train, y_train)
reg9 = Ridge(alpha=alpha9).fit(X_train, y_train)
reg10 = Ridge(alpha=alpha10).fit(X_train, y_train)
reg11 = Ridge(alpha=alpha11).fit(X_train, y_train)
reg12 = Ridge(alpha=alpha12).fit(X_train, y_train)

#print("重み :",reg1.coef_)
print("重みの形状 :",reg1.coef_.shape)


y_pre1=reg1.predict(X_test)
y_pre2=reg2.predict(X_test)
y_pre3=reg3.predict(X_test)
y_pre4=reg4.predict(X_test)
y_pre5=reg5.predict(X_test)
y_pre6=reg6.predict(X_test)
y_pre7=reg7.predict(X_test)
y_pre8=reg8.predict(X_test)
y_pre9=reg9.predict(X_test)
y_pre10=reg10.predict(X_test)
y_pre11=reg11.predict(X_test)
y_pre12=reg12.predict(X_test)

print('Completed training regression.')

ridge_end = time.time()

print("回帰時間:",ridge_end - ridge_start)



calculation_start = time.time()



y_test_T = y_test.T

y_pre1_T = y_pre1.T
y_pre2_T = y_pre2.T
y_pre3_T = y_pre3.T
y_pre4_T = y_pre4.T
y_pre5_T = y_pre5.T
y_pre6_T = y_pre6.T
y_pre7_T = y_pre7.T
y_pre8_T = y_pre8.T
y_pre9_T = y_pre9.T
y_pre10_T = y_pre10.T
y_pre11_T = y_pre11.T
y_pre12_T = y_pre12.T


"""
c = []
print(y_test_T.shape)
print(y_pre1_T.shape)
for j in range(y_test_T.shape[0]):
    corr_val = pair_corr_with_p(y_test_T[j, :],y_pre1_T[j, :])
    c.append(corr_val)
#print(len(c))
mean1 = sum(c) / len(c)
print(mean1)

c = []
for j in range(y_test_T.shape[0]):
    corr_val = pair_corr_with_p(y_test_T[j, :],y_pre2_T[j, :])
    c.append(corr_val)
#print(len(c))                                                                                                                         
mean2 = sum(c) / len(c)
print(mean2)

c = []
for j in range(y_test_T.shape[0]):
    corr_val = pair_corr_with_p(y_test_T[j, :],y_pre3_T[j, :])
    c.append(corr_val)
#print(len(c))                                                                                                                        
mean3 = sum(c) / len(c)
print(mean3)

c = []
for j in range(y_test_T.shape[0]):
    corr_val = pair_corr_with_p(y_test_T[j, :],y_pre4_T[j, :])
    c.append(corr_val)
#print(len(c))                                                                                                                        
mean4 = sum(c) / len(c)
print(mean4)

c = []
for j in range(y_test_T.shape[0]):
    corr_val = pair_corr_with_p(y_test_T[j, :],y_pre5_T[j, :])
    c.append(corr_val)
#print(len(c))                                                                                                                        
mean5 = sum(c) / len(c)
print(mean5)

c = []
for j in range(y_test_T.shape[0]):
    corr_val = pair_corr_with_p(y_test_T[j, :],y_pre6_T[j, :])
    c.append(corr_val)
#print(len(c))                                                                                                                        
mean6 = sum(c) / len(c)
print(mean6)

c = []
for j in range(y_test_T.shape[0]):
    corr_val = pair_corr_with_p(y_test_T[j, :],y_pre7_T[j, :])
    c.append(corr_val)
#print(len(c))                                                                                                                        
mean7 = sum(c) / len(c)
print(mean7)

c = []
for j in range(y_test_T.shape[0]):
    corr_val = pair_corr_with_p(y_test_T[j, :],y_pre8_T[j, :])
    c.append(corr_val)
#print(len(c))                                                                                                                        
mean8 = sum(c) / len(c)
print(mean8)

c = []
for j in range(y_test_T.shape[0]):
    corr_val = pair_corr_with_p(y_test_T[j, :],y_pre9_T[j, :])
    c.append(corr_val)
#print(len(c))                                                                                                                        
mean9 = sum(c) / len(c)
print(mean9)

c = []
for j in range(y_test_T.shape[0]):
    corr_val = pair_corr_with_p(y_test_T[j, :],y_pre10_T[j, :])
    c.append(corr_val)
#print(len(c))                                                                                                                        
mean10 = sum(c) / len(c)
print(mean10)

c = []
for j in range(y_test_T.shape[0]):
    corr_val = pair_corr_with_p(y_test_T[j, :],y_pre11_T[j, :])
    c.append(corr_val)
#print(len(c))                                                                                                                        
mean11 = sum(c) / len(c)
print(mean11)

c = []
for j in range(y_test_T.shape[0]):
    corr_val = pair_corr_with_p(y_test_T[j, :],y_pre12_T[j, :])
    c.append(corr_val)
#print(len(c))                                                                                                                        
mean12 = sum(c) / len(c)
print(mean12)

corrs = mean1,mean2,mean3,mean4,mean5,mean6,mean7,mean8,mean9,mean10,mean11,mean12
"""
corrs = (0.0013698362492903565, 0.0013780414497261324, 0.0014432592706044598, 0.0015109349702812178, 0.002145482335203186, 0.002459846052478293, 0.001390619607712099, 0.0008511478238286443, 0.0005836889785510984, 0.00035367775959347275, -0.0007032847094089225, -0.0014277218030847366)
#print(corrs)

print(len(best_parameters))
counter = 0
y_pred_all = []
corr_list = []
vectore = []
for i in best_parameters:
    if i == alpha1:
      corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre1_T[counter, :])
      #corr = np.corrcoef(y_test_T[counter, :],y_pre1_T[counter, :])[0,1]
      vectore.append(y_pre1_T[counter,:])
      corr_list.append(corr_with_p)
      counter = counter + 1
    elif i == alpha2:
      corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre2_T[counter, :])
      corr_list.append(corr_with_p)
      vectore.append(y_pre2_T[counter,:])
      counter = counter + 1
    elif i == alpha3:
      corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre3_T[counter, :])
      #corr = np.corrcoef(y_test_T[counter, :],y_pre3_T[counter, :])[0,1]
      corr_list.append(corr_with_p)
      vectore.append(y_pre3_T[counter,:])
      counter = counter + 1

    elif i == alpha4:
      corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre4_T[counter, :])
      #corr = np.corrcoef(y_test_T[counter, :],y_pre4_T[counter, :])[0,1]
      corr_list.append(corr_with_p)
      vectore.append(y_pre4_T[counter-1,:])
      counter = counter + 1

    elif i == alpha5:
      corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre5_T[counter, :])
      #corr = np.corrcoef(y_test_T[counter, :],y_pre5_T[counter, :])[0,1]
      corr_list.append(corr_with_p)
      vectore.append(y_pre5_T[counter-1,:])
      counter = counter + 1
    elif i == alpha6:
      corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre6_T[counter, :])
      #corr = np.corrcoef(y_test_T[counter, :],y_pre6_T[counter, :])[0,1]
      corr_list.append(corr_with_p)
      vectore.append(y_pre6_T[counter,:])
      counter = counter + 1
    elif i == alpha7:
      corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre7_T[counter, :])
      #corr = np.corrcoef(y_test_T[counter, :],y_pre7_T[counter, :])[0,1]
      corr_list.append(corr_with_p)
      vectore.append(y_pre7_T[counter,:])
      counter = counter + 1
    elif i == alpha8:
      corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre8_T[counter, :])
      #corr = np.corrcoef(y_test_T[counter, :],y_pre8_T[counter, :])[0,1]
      corr_list.append(corr_with_p)
      vectore.append(y_pre8_T[counter,:])
      counter = counter + 1
    elif i == alpha9:
      corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre9_T[counter, :])
      #corr = np.corrcoef(y_test_T[counter, :],y_pre9_T[counter, :])[0,1]
      corr_list.append(corr_with_p)
      vectore.append(y_pre9_T[counter,:])
      counter = counter + 1
    elif i == alpha10:
      corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre10_T[counter, :])
      #corr = np.corrcoef(y_test_T[counter, :],y_pre10_T[counter, :])[0,1]
      corr_list.append(corr_with_p)
      vectore.append(y_pre10_T[counter,:])
      counter = counter + 1
    elif i == alpha11:
      corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre11_T[counter, :])
      #corr = np.corrcoef(y_test_T[counter, :],y_pre11_T[counter, :])[0,1]
      corr_list.append(corr_with_p)
      vectore.append(y_pre11_T[counter,:])
      counter = counter + 1
    elif i == alpha12:
      corr_with_p=pair_corr_with_p(y_test_T[counter, :],y_pre12_T[counter, :])
      #corr = np.corrcoef(y_test_T[counter, :],y_pre12_T[counter, :])[0,1]
      corr_list.append(corr_with_p)
      vectore.append(y_pre12_T[counter,:])
      counter = counter + 1
    elif i == 0: # no update:  koide insert
      corr = np.nan
      corr_list.append(corr)
      counter = counter + 1

corr_list= np.array(corr_list)
vectore = np.array(vectore)
np.set_printoptions(threshold=1000)
print(corr_list)
print(len(corr_list))
num = len(corr_list) - np.count_nonzero(corr_list)
print(num)

#corrs = list(corrs)
#corrs = np.array(corrs)
corr_t = tuple(corr_list)

#alpha_graph(alphas,corr_t, corrs, b_alpha, 5, os.getcwd(), save=True)

calculation_end = time.time()
print("計算時間:",calculation_end - calculation_start)


"""
ridge_cv = RidgeCV(alphas=[alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9,alpha10,alpha11,alpha12], cv=5)
reg = ridge_cv.fit(X_train, y_train)
print("alpha:",ridge_cv.alpha_)

pred = reg.predict(X_test)

corr_list = correlation_c(y_test.T,pred.T)
corr_list = np.array(corr_list)
print(corr_list.shape)
#print(reg.score(X_train,Y_train))
#print(reg.coef_)
#print(reg.intercept_)
"""
save_start = time.time()

np.save('../only_dvd_result/DM01/GIS2_correlation',corr_list)
#np.save('../bert_sen_ridge_result/DM07/pre_GIS2_correlation',corr_list)
#np.save('../src/DM03/Y_pred/Mentalist_pred',  vectore)
print(corr_list.shape)
save_end = time.time()

#print("保存時間:",save_end - save_start)

#print("全工程時間:",save_end - begin)
