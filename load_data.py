#このプログラムはmat形式の脳活動データの必要な部分のみを取り出し、pickle形式で保存します
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import torch
import torch.nn as nn
import sys
import numpy as np
from scipy.io import loadmat
import pickle
import h5py


print(torch.nn.utils.clip_grad_norm_)
print("cudnn version", torch.backends.cudnn.version())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
print(torch.cuda.is_available())
print(torch.cuda.device_count())



def Standardization(X, mean, sd):

	return (float(X - mean)/sd)


r = 5#ru_n数

subjectCode = 1#DM01=0,DM03=1,DM07=2,DM09=3
movid = 4
# 0)BreakingBad, 1)BigBangTheory1, 2)BigBangTheory2, 3)Crown, 4)Heroes, 5)SUITS,6)DreamGirls, 7)Glee, 8)Mentalist, 9)GIS1, 10)GIS2,
#runid = 1
#runid = runid - 1
#個人的run1から書いた方がわかりやすいから1引いてます
print("subject:",subjectCode)
print("movid",movid)

resp_data_combine = []

with open("../mask/DreamGirls_現在.pickle",'rb') as f:
    gm = pickle.load(f)

#マスキングデータで必要部分を取り出す
mask = loadmat("../bold/DM03/vset_100.mat")
mask = mask['tvoxels']
mask = [ flatten for inner in mask for flatten in inner ]
print("mask size:",len(mask))


item = loadmat("../loadItems.mat")

dt = 4#脳活動データの遅れる秒数
#width = 1#何秒の幅のデータをとるか
#print("width:",width)
#width = width -1

rm =[]
counter=0

for i in range(r):
        
        if i <= 4:
                data = h5py.File("../bold/DM07/" + "20200114S1.psn/trendRemoved-0" + str(i+1)  + ".mat",'r')
        else:
                data = h5py.File("../bold/DM07/" + "20200121S1.psn/trendRemoved-0" + str(i-4)  + ".mat",'r')
        print("aaa")
        
        #data = h5py.File("../bold/DM07/" + "20200325S3.psn/trendRemoved-0" + str(i+3)  + ".mat",'r')
        data = data['dataDT']
        print("row brain data size:",data.shape)
        print("run",i+1)

        rm = []
        a=[0]
        sen = ''
        no =0
        nm=0
        counter = 0
        sec=0

        """
	with open("../../dvd-prepro/data/Mentalist/Mentalist_run_id.txt",'r+') as f:
                for lines in f.readlines():
                        col = lines.split('\t')
                        run_tag = col[0]
                        #print(run_tag)
                        #print(run_tag == 'run' + str(i+1))
                        #print(col)
                        if run_tag == 'run' + str(i+1):
                                #print(col)
                                if sen == '..' and sen != col[2][:-1]:
                                        #print(col[1])
                                        sec = int(col[1])
                                        for k in range(width):
                                                #print("あ:",int(col[1])+k)
                                                rm.append(int(col[1])+dt+k-1)
                                if col[2][:-1] == '..':
                                        if int(col[1]) - sec <= width and sen != '..' :
                                                print(int(col[1]),sec,int(col[1]) - sec)
                                        #print(col[1])
                                        #print(int(col[1]))
                                        no += 1
                                        nm += 1
                                        rm.append(int(col[1])+dt-1)
                                else:
                                        counter += 1
                                        if a[-1] != no and no !=0:
                                                #print(no)
                                                a.append(no)
                                                no=0
                                                sen = col[2][:-1]
		if no!=0 and i+1 == r:
			a.append(no)
	rm = sorted(list(set(rm)))
	print(a)
	print((len(a)-1)*2 + sum(a))
	#print(counter)
	print(".. :",nm)"""

        subject = item['subject']
        mov = subject[0,subjectCode]['mov']
        run = mov[0,movid]['run']
        samples = run[0,i]['samples']
        #print(samples.shape)
        samples = [ flatten for inner in samples-1 for flatten in inner ]

        print("最初の20秒削っただけの長さ:",len(samples))
        rm = [k for k in rm if k < len(samples)+20]
        print("不要部分の長さ:",len(rm))
        print(gm[i])
        

        respsamples = []
        counter = 0

        for t in samples:
                #print(t)
                if counter < len(rm):
                        if t != rm[counter]:
                                #print(t)
                                respsamples.append(t)
                        else:
                                counter += 1
                else:
                        respsamples.append(t)

        print(respsamples)
        print("respsamples size:",len(respsamples))
        print(len(gm[i]))
        #print("resp_data:",resp_data)

        resp_data = data[:, mask]
        print("resp_data:",resp_data)
        print(resp_data.shape)


        #resp_data = resp_data[respsamples,:]

        #resp_data = resp_data[rm,:]
        resp_data = resp_data[gm[i],:]
        print(resp_data)
        print(resp_data.shape)
        print(type(resp_data))

        #print(np.array(resp_data).shape)

        # 平均0、分散1に正規化
        #mean = np.mean(resp_data)
        #sd = np.std(resp_data)
        #resp_data = np.array([[Standardization(X, mean, sd) for X in row] for row in resp_data])
        #転置
        #resp_data = resp_data.T
        #print("resp_data:",resp_data.shape)

        file_path = "../data/DM07/DreamGirls/DreamGirls_現在_run" + str(i+1) + ".pickle"
        print(file_path)
        with open(file_path,'wb') as f:
                pickle.dump(resp_data,f)
