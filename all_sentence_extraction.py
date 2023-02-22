#①このプログラムは１秒ごとに入った文全てを取り出すもの
import re

arr=[]

#with open("../../data/ninjal_conll/ザクラウン.conll",'r+') as f:
with open("../../data/Suits/SUITS_run3_soundDescription_each1sec.txt",'r+') as f:
    for lines in f.readlines():
        col = lines.split('\t')
        #print(col)
        if lines.startswith('#'):
            #print(col[0])
            #print(col)
            line = str(col[0])
            line = line.split()
            #print(line)
            arr.append(line[3])
            #print(arr)

text = ""
for a in arr:
    text += a
    text += '\n'
print(text)

#file = open('../../data/text/Crown/all_Crown_sentence_conll.txt', 'w')
file = open('../../data/text/Suits/all_Suits_sentence_run3.txt', 'w')
file.write(text)
file.close()
