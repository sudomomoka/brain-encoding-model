#②このプログラムは"all_sentence_extraction.py"で取り出した文を１秒ごとに列にする
#微妙にうまくいかないとことか、データ自体がまちがってるところがあるから手作業でちょいちょい直さないとだめ
import re
import collections

arr=['a']
list=[]
counter=0
flag=0
k=''
l=''

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


f = open("../../data/text/Crown/all_Crown_sentence_run5.txt",'r+')
lines = f.readlines()
lines = [line.strip() for line in lines]
c = collections.Counter(lines)
keys = get_keys_from_value(c, 1)
print(keys)

print(len(lines))

for counter in range(len(lines)-1):
    if lines[counter] == lines[counter+1]:
        arr.append(k)
        k = ''
        arr.append(lines[counter])
        #print(lines[counter])
    elif (lines[counter] != lines[counter+1]) and (lines[counter] != lines[counter-1]):
        if lines[counter] not in k and k != arr[-1]:
            for i in keys:
                if i == k :
                    print(i)
                    arr.append(k)
                    k =''
            k = k + lines[counter]

        else:
            arr.append(k)
            #print(k)
            k = ''
            l = ''
            k = k  + lines[counter]
    else:
        k=''
        arr.append(lines[counter])

arr.append(arr[-1])
#l = [line for line in lines if line.startswith(col)]

#for counter in range(len(arr)):
f.close()
#print(re.match(b,a))


text = ""
for a in arr[1:]:
    if a != '':
        text = text + a
        text += '\n'

#print(text)

file = open('../../data/Crown/Crown_run5.txt', 'w')
file.write(text)
file.close()
