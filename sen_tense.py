#①このプログラムは一文ごとに時制をつけるモデル制作のためのもの
#「Ninjalconll」のファイルと「~~~_run?.txt」(これはdvd-prepro/data/映画名/のところにある)を一文ごとに参照。
#その文の中に、時制がついてるイベントがあったら、その時制のタグをつける。
#時制のついてるイベントがなかったら、その分は[現在]のタグ？それか[その他]
#文がないところはそのまま(あとでゼロ埋めする)

import re

arr=[]
with open("../data/Heroes/Heroes_run5.txt",'r+') as f2:
    for l in f2.readlines():
        c = l[:-1]
        arr.append(c)

print("秒数：",len(arr))

arr2=[]
with open("../data/ninjal_conll/HEROES.conll",'r+') as f1:
    for lines in f1.readlines():
        col = lines.split('\t')
        #print(col)
        if lines.startswith('#'):
            #print(col[0])
            #print(col)
            line = str(col[0])
            line = line.split()
            sen = line[3]
            sen = ''.join(filter(str.isalnum, sen))
            #arr.append(line[3])
            #print(sen)
            #print(arr)
        elif not lines.startswith('\n'):
            #print(col[11])
            time = col[11]
            if time != "_":
                #print(time)
                arr2.append([sen,time])
                #print(sen)

#for l in arr2:
#    print(l)

#print("\n\n\n\n\n\n")

arr3 = []
for i in range(len(arr2)):
    if (i != len(arr2)-1) and (i != 0):
        if (arr2[i][0] == arr2[i+1][0]) and (arr2[i-1][0] != arr2[i][0]):
            #print(arr2[i][0])
            #print(arr2[i][1])
            #print(arr2[i+1][1])
            if not arr2[i][1] == arr2[i+1][1]:
                arr3.append([arr2[i][0],[arr2[i][1],arr2[i+1][1]]])
            else:
                arr3.append([arr2[i][0],[arr2[i][1]]])
        elif arr2[i-1][0] != arr2[i][0]:
            arr3.append([arr2[i][0],[arr2[i][1]]])

    else :
        arr3.append([arr2[i][0],[arr2[i][1]]])
#print(arr3)



result = []
flag = 0
for i in arr:
    flag = 0
    s1 = ''.join(filter(str.isalnum, i))
    if ".." in i :
        ss = str(i) + "\t" + "['その他']"
        result.append(ss)
    else:
        for t in arr3:
            s2 = ''.join(filter(str.isalnum, t[0]))
            if (len(s2) < len (s1)) and (s2 in s1):
                ss = str(i) + "\t" + str(t[1])
                result.append(ss)
                flag = 1
                break
            if (len(s2) >= len (s1)) and (s1 in s2):
                ss = str(i) + "\t" + str(t[1])
                result.append(ss)
                flag = 1
                break

        if flag != 1:
            ss = str(i) + "\t" + "['その他']"
            result.append(ss)
"""
for i in arr:
    if i != "..":
        s1 = ''.join(filter(str.isalnum, i))
        s2 = ''.join(filter(str.isalnum, r[counter][0]))
        #print(s)
        #print(r[counter][0])
        if s1 in s2 :
            ss = str(i) + "\t" + str(r[counter][1])
            result.append(ss)
            counter = counter + 1
            #print(s1)

        #if s2 in s1:
            #ss = str(i) + "\t" + str(r[counter][1])
            #counter = counter + 1
            #print(s1)

        else:
            #print(s1)
            #print(s2)
            ss = str(i) + "\t" + "['その他']"
            #print(ss)
            result.append(ss)
    else:
        ss = str(i) + "\t" + "['その他']"
        result.append(ss)
"""

#print(r)
text = ""
counter = 0
for a in result:
    counter += 1
    text += a
    text += '\n'

for l in arr3:
    print(l)
    print("\n\n\n")

#print(text)
print(counter)
#exit()

file = open('../data/Heroes/all_Heroes_tense_run5.txt', 'w')
file.write(text)
file.close()

