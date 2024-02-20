'''
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open(r"inference/images/19.png")
plt.imshow(im)
plt.show()

import tensorflow as tf

#查看tensorflow版本
print(tf.__version__)

print('GPU', tf.test.is_gpu_available())

a = tf.constant(2.0)
b = tf.constant(4.0)
print(a + b)

'''


# label 排序
import os 
dir = 'inference/label/'
path = os.listdir(dir)
path = sorted(path,key=lambda x:int(x[:-4]))
for file in path:
    with open(dir+file, 'r',encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(0, len(lines)):
        lines[i] = lines[i].rstrip('\n')
        lines[i] = lines[i].split()
    forward_player_cards,back_player_cards,my_cards,my_left_cards,lord_card=[],[],[],[],[]
    for line in lines:
        if float(line[2]) < 0.1961:
            lord_card.append(line)
        elif float(line[2]) > 0.3922 and float(line[2]) < 0.58824 and float(line[1]) < 0.4167:
            forward_player_cards.append(line)
        elif float(line[2]) > 0.3922 and float(line[2]) < 0.58824 and float(line[1]) > 0.5556:
            back_player_cards.append(line)
        elif float(line[2]) > 0.58824 and float(line[2]) < 0.82353:
            my_cards.append(line)
        elif float(line[2]) > 0.82353:
            my_left_cards.append(line)
    lord_card = sorted(lord_card,key=lambda x: x[1])
    forward_player_cards = sorted(forward_player_cards,key=lambda x: x[1])
    back_player_cards = sorted(back_player_cards,key=lambda x: x[1])
    my_cards = sorted(my_cards,key=lambda x: x[1])
    my_left_cards = sorted(my_left_cards,key=lambda x: x[1])
    with open(dir+file, 'w',encoding='utf-8') as f:
        for i in lord_card:
            f.write('地主牌%s:'% (lord_card.index(i)+1))
            f.write(str(i)+'\n')
        for i in forward_player_cards:
            f.write('上家出牌%s:'% (forward_player_cards.index(i)+1))
            f.write(str(i)+'\n')
        for i in back_player_cards:
            f.write('下家出牌%s:'% (back_player_cards.index(i)+1))
            f.write(str(i)+'\n')
        for i in my_cards:
            f.write('本家出牌%s:'% (my_cards.index(i)+1))
            f.write(str(i)+'\n')
        for i in my_left_cards:
            f.write('本家手牌%s:'% (my_left_cards.index(i)+1))
            f.write(str(i)+'\n')









































































