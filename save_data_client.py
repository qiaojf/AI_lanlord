
import json
import os
import socket
import struct
import sys
import threading

from PIL import Image, ImageGrab
# import win32api, win32con, win32gui
from skimage import io

players = ['player1','player2','player3']
def get_img():
    i = 1
    while True:
        while True:
            img = ImageGrab.grab((120, 0, 1560, 1020))  # x=1440 y=1020
            file = r"./tmp/{}.png".format(i)
            img.save(file)
            img = io.imread(file)
            if img[130,800,1] in (64,88): #没有地主牌，
                # if img[820,100,1] not in (64,88): #计时器在当前玩家
                #     pass
                # elif img[350,150,1] not in (64,88): #计时器在上家
                #     pass
                # elif img[350,1280,1] not in (64,88):   #计时器在下家
                #     pass
                # elif img[820,100,1] in (64,88) and img[350,150,1] in (64,88) and img[350,1280,1] in (64,88): #没有计时器
                pass
            elif img[130,800,1] not in (64,88): # 有地主牌
                if img[850,100,1] not in (64,88) and img[440,150,1] not in (64,88): #计时器在当前玩家
                    # print('本家出牌中...')
                    break
                elif img[380,150,1] not in (64,88) and img[450,1290,1] not in (64,88): #计时器在上家
                    # print('上家出牌中...')
                    break
                elif img[380,1280,1] not in (64,88) and img[770,700,1] not in (64,88):   #计时器在下家
                    # print('下家出牌中...')
                    break
                elif img[850,100,1] in (64,88) and img[380,150,1] in (64,88) and img[380,1280,1] in (64,88):  # 没有计时器
                    # print('游戏结束')
                    break
        if i == 1:
            pass
        else:
            old_img = io.imread('./tmp/{}.png'.format(i-1))
            if img[850,100,1]==old_img[850,100,1] and img[380,150,1]==old_img[380,150,1] and img[380,1280,1]==old_img[380,1280,1]:
                os.remove('./tmp/{}.png'.format(i-1))
        i += 1


def deal_data(dt_boxes,rec_res):
    lord_cards = []
    forward_player_cards = []
    back_player_cards = []
    my_cards = []
    my_left_cards = []
    bbox = sorted(dt_boxes,key=lambda x: x[0],reverse=True)
    for i in bbox:
        if float(i[1]) < 0.1961:  #地主牌
            lord_cards.append(rec_res[dt_boxes.index(i)])
        elif float(i[0]) < 0.4167 and float(i[1]) < 0.58824 and float(i[1]) > 0.3922:  #上家出牌
            forward_player_cards.append(rec_res[dt_boxes.index(i)])
        elif float(i[0]) > 0.5556 and float(i[1]) < 0.58824 and float(i[1]) > 0.3922:  #下家出牌
            back_player_cards.append(rec_res[dt_boxes.index(i)])
        elif float(i[1]) > 0.82353:    #当前玩家手牌
            my_left_cards.append(rec_res[dt_boxes.index(i)])
        elif float(i[1]) > 0.58824 and float(i[1]) < 0.82353: #当前玩家出牌
            my_cards.append(rec_res[dt_boxes.index(i)])
    return forward_player_cards,back_player_cards,my_cards,my_left_cards

def get_record(record,forward_player_cards,back_player_cards,my_cards):
    def get_player():
        if record == []:
            player = players[0]
        else:
            player_index = players.index(record[-1][0])+1
            if player_index > 2:
                player = players[0]
            else:
                player = players[player_index]
        return player

    def is_over():
        if record != []:
            records =[x for x in record if x[1]!=['PASS']]
            card_num1,card_num2,card_num3=0,0,0
            for i in records:
                if i[0]==players[0]:
                    card_num1 += len(i[1])
                if i[0]==players[1]:
                    card_num2 += len(i[1])
                if i[0]==players[2]:
                    card_num3 += len(i[1])
            if card_num1==20 or card_num2==17 or card_num3==17:
                is_over=True
            else:
                is_over=False
        return is_over

    if forward_player_cards != [] and back_player_cards == [] and my_cards == []: #上家首先出牌
        if record == []:
            record.append((players[0],forward_player_cards))
        else:
            player = get_player()
            record.append((player,forward_player_cards))
            if is_over:
                with open('./game_record.txt','a',encoding="utf-8") as f:
                    f.write(str(record)+','+'\n')
                record = []

    elif forward_player_cards == [] and back_player_cards != [] and my_cards == []: #下家首先出牌
        if record == []:
            record.append((players[0],back_player_cards))
        else:
            player = get_player()
            record.append((player,back_player_cards))
            if is_over:
                with open('./game_record.txt','a',encoding="utf-8") as f:
                    f.write(str(record)+','+'\n')
                record = []

    elif forward_player_cards == [] and back_player_cards == [] and my_cards != []: #当前玩家首先出牌
        if record == []:
            record.append((players[0],my_cards))
        else:
            player = get_player()
            record.append((player,my_cards))
            if is_over:
                with open('./game_record.txt','a',encoding="utf-8") as f:
                    f.write(str(record)+','+'\n')
                record = []

    if record != []:
        player = get_player()
        if forward_player_cards != [] and back_player_cards != [] and my_cards == []: # 当前玩家出牌
            if back_player_cards == record[-1][1]:
                record.append((player,forward_player_cards))
        elif forward_player_cards != [] and back_player_cards == [] and my_cards != []: # 下家出牌
            if forward_player_cards == record[-1][1]:
                record.append((player,my_cards))
        elif forward_player_cards == [] and back_player_cards != [] and my_cards != []: # 上家出牌
            if my_cards == record[-1][1]:
                record.append((player,back_player_cards))
        elif forward_player_cards != [] and back_player_cards != [] and my_cards != []:
            if forward_player_cards == record[-1][1]: #下家出牌
                record.append((player,my_cards))
            elif my_cards == record[-1][1]: #上家出牌
                record.append((player,back_player_cards))
            elif back_player_cards == record[-1][1]: #当前玩家出牌
                record.append((player,forward_player_cards))
    return record

def socket_client():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('192.168.1.6',9999))
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    record = []
    while True:
        pic_path = './tmp/'
        if len(os.listdir(pic_path)) > 2:
            filelist = sorted(os.listdir(pic_path),key=lambda x:int(x[:-4]))
            for i in filelist:
                filepath = pic_path + i
                if os.path.isfile(filepath):
                    # 定义定义文件信息。128s表示文件名为128bytes长，l表示一个int或log文件类型，在此为文件大小
                    fileinfo_size = struct.calcsize('128sl')
                    # 定义文件头信息，包含文件名和文件大小
                    fhead = struct.pack('128sl', bytes(os.path.basename(filepath).encode('utf-8')),os.stat(filepath).st_size)
                    s.send(fhead)
                    # print ('client filepath: {0}'.format(filepath))
                    fp = open(filepath, 'rb')
                    while True:
                        data = fp.read(10240)
                        if not data:
                            print ('{0} file send over...'.format(filepath))
                            fp.close()
                            break
                        s.send(data)
                server_reply = s.recv(10240).decode()
                # os.remove(filepath)
                recv_data = eval(server_reply)
                dt_boxes,rec_res = recv_data[0],recv_data[1]
                forward_player_cards,back_player_cards,my_cards,my_left_cards = deal_data(dt_boxes,rec_res)
                print(forward_player_cards,back_player_cards,my_cards)
                record = get_record(record,forward_player_cards,back_player_cards,my_cards)    
        else:
            pass

t1 = threading.Thread(target=get_img, args=())
t2 = threading.Thread(target=socket_client, args=())
# t1.start()
t2.start()


































