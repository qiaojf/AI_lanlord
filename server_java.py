
from poke import Game_env
from RL_brain import PolicyGradient,DQNPrioritizedReplay
from doudizhu import Card,engine

import random
import numpy as np
import tensorflow as tf

import socket
import json
import threading   

# tf.compat.v1.disable_eager_execution()
players = ['player1','player2','player3']

game = Game_env()
model_name1 = r'Model_pr/player2_0.656000_4000.ckpt'
RL_D = DQNPrioritizedReplay(
            game.n_actions, 
            game.n_features,
        )
RL_D.load_model(model_name1)
model_name = r'Model_/player2_0.680000_9000.ckpt'
RL_P = PolicyGradient(
    game.n_actions, 
    game.n_features,
    )
RL_P.load_model(model_name)

def get_player(mark):
    if mark == players[0]:
        player = players[1]
    elif mark == players[1]:
        player = players[2]
    elif mark == players[2]:
        player = players[0]
    return player

def get_act(act):
    acts = list(set(act))
    if 'BJ' in acts:
        acts.remove('BJ')
    if 'CJ' in acts:    
        acts.remove('CJ')
    for k in acts:
        idx = [i for i,x in enumerate(act) if x==k]
        if len(idx) == 1:
            act[idx[0]] = k+'c'
        elif len(idx) == 2:
            act[idx[0]] = k+'c'
            act[idx[1]] = k+'d'
        elif len(idx) == 3:
            act[idx[0]] = k+'c'
            act[idx[1]] = k+'d'
            act[idx[2]] = k+'h'
        elif len(idx) == 4:
            act[idx[0]] = k+'c'
            act[idx[1]] = k+'d'
            act[idx[2]] = k+'h'
            act[idx[3]] = k+'s'
    return act

def get_cards(cards):
    card = list(set(cards))
    if 'BJ' in card:
        card.remove('BJ')
    if 'CJ' in card:
        card.remove('CJ')
    for k in card:
        idx = [i for i,x in enumerate(cards) if x==k]
        if len(idx) == 1:
            cards[idx[0]] = k+'s'
        elif len(idx) == 2:
            cards[idx[0]] = k+'s'
            cards[idx[1]] = k+'h'
        elif len(idx) == 3:
            cards[idx[0]] = k+'s'
            cards[idx[1]] = k+'h'
            cards[idx[2]] = k+'d'
        elif len(idx) == 4:
            cards[idx[0]] = k+'s'
            cards[idx[1]] = k+'h'
            cards[idx[2]] = k+'d'
            cards[idx[3]] = k+'c'
    return cards

def get_mark(record):
    if record == []:
        act_mark = ('player1',None)
    else:
        records =[x for x in record if x[1]!='pass']
        act = records[-1][1]
        acts = get_act(act)
        act_mark = (records[-1][0],acts)
    return act_mark

def change_data(data):
    data = eval(data)
    for i in data.keys():
        if i != 'record':
            player_cards=i
    player_cards = data[player_cards].split(',')

    records = []
    for i in data['record']:
        i=str(i).replace('{','(')
        i=i.replace('}',')')
        i=i.replace(':',',')
        records.append(eval(i))

    record = []
    for i in records:
        x = i[0]
        y = i[1]
        if y != "pass":
            y = y.split(',')
            y = engine.sort_cards(y)
        record.append((x,y))
    return player_cards,record

def deal_recv(client_data):
    player_cards,record = change_data(client_data)
    state = game.get_observation(record)
    player_cards = get_cards(player_cards)
    player_cards = [Card.new(x) for x in player_cards]
    if record != []:
        player = get_player(record[-1][0])
    else:
        player = players[0]
    act_mark = get_mark(record)
    actions = game.get_actions(act_mark,player,player_cards)
    action = RL_P.choose_action(state,actions,game.action_space)
    if action not in actions:
        action = random.choice(actions)
    actions.clear()
    return action

def recv_handler(link, client):     
    print("开始接收[%s:%s]的请求...." % (client[0], client[1]))
    while True:     # 利用一个死循环，保持和客户端的通信状态
        try:
            client_data = link.recv(80960).decode()
            print("[%s:%s]向你发来信息：%s" % (client[0], client[1], client_data))
            action = deal_recv(client_data)
            # data = '信息已收到'
            link.sendall(str(action).encode())
        except:
            data='出错了！请检查传入数据格式！！！'
            link.sendall(data.encode())
    link.close()

ip_port = ('192.168.1.6', 9999)
sk = socket.socket()            # 创建套接字
sk.bind(ip_port)                # 绑定服务地址
sk.listen(5)                    # 监听连接请求

print('启动socket服务，等待客户端连接...')

while True:     # 一个死循环，不断的接受客户端发来的连接请求
    conn, address = sk.accept()  
    t = threading.Thread(target=recv_handler, args=(conn, address))
    t.start()


























 



