#③このプログラムは,作成したテキストファイルに１秒ごとにsentenceIDをつける
import re

arr=[]

counter = 1
run = 9#run数

for i in range(run):
    with open("../../data/DreamGirls/" + "all_DreamGirls_tense_run" + str(i+1) +".txt",'r+') as f1:
        counter = 1
        for lines in f1.readlines():
            line = lines.split()
            if counter > 20:
                line = str(lines)
                line = "run" + str(i+1) + '\t' + str(counter) + '\t' + line
                #line = line + '\n'
                arr.append(line)
                #print(arr)
                counter += 1
            else:
                counter += 1
text = ""
for a in arr:
    text = text + a
    #text += '\n'
print(text)

f2 = open("../../data/DreamGirls/DreamGirls_run_id.txt",'w')
f2.write(text)
f2.close()
