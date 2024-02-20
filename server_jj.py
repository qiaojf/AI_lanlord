
import json
import random
import socket
import threading

import numpy as np
import tensorflow as tf

from doudizhu import Card, engine
from poke import Game_env
from RL_brain1 import DQNPrioritizedReplay, PolicyGradient, PolicysGradient

tf.compat.v1.disable_eager_execution()
players = ['player1','player2','player3']

game = Game_env()
model_name1 = r'Model_last/player1_0.5_1000.ckpt'
RL_1 = PolicyGradient(
            game.n_actions, 
            game.n_features,
        )
RL_1.load_model(model_name1)
model_name2 = r'Model_last/player2_0.5_1000.ckpt'
RL_2 = PolicysGradient(
    game.n_actions, 
    game.n_features,
    )
RL_2.load_model(model_name2)
model_name3 = r'Model_last/player3_0.5_1000.ckpt'
RL_3 = DQNPrioritizedReplay(
    game.n_actions, 
    game.n_features,
    )
RL_3.load_model(model_name3)


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
        records =[x for x in record if x[1]!=['PASS']]
        act = records[-1][1]
        acts = get_act(act)
        act_mark = (records[-1][0],acts)
    return act_mark

def change_data(data):
    data = eval(data)
    player_cards = data[1]
    record = data[0]
    return player_cards,record

def get_actions_candy(cards):
    def append_act():
        while player_cards!=[]:
            act = game.list_candidate(player_cards)
            mark = len(max(act, key=len, default=''))
            for y in act:
                if len(y)==mark:
                    candy_list.append(y)
                    for j in y:
                        player_cards.remove(j)
                    break
            act.clear()
    player_cards = cards.copy()
    acts = game.list_candidate(player_cards)
    acts = [x for x in acts if len(x)>=5]
    candy_list = []
    if acts==[]:
        append_act()
        actions_candy = (len(candy_list),candy_list)
        return actions_candy
    else:
        len_mark = 17
        actions_candy = []
        for i in acts:
            candy_list.append(i)
            for z in i:
                player_cards.remove(z)
            append_act()
            act_count = len(candy_list)
            if act_count < len_mark:
                len_mark = act_count
                actions_candy.clear()
                actions_candy.append((len_mark,candy_list))
            candy_list = []        
            player_cards = cards.copy()
    return actions_candy

def rule_stradge(action,actions,actions_candy):
    new_actions = []
    for i in actions_candy:
        new_actions.extend(i[1])
    if action in set(new_actions):
        return action
    elif action not in set(new_actions):
        for i in actions:
            if i in set(new_actions):
                action = i
                return action
        action = ['PASS']
        return action

def deal_recv(client_data):
    player_cards,record = change_data(client_data)
    state = game.get_observation(record)
    player_cards = get_cards(player_cards)
    # actions_candy = get_actions_candy(player_cards)
    player_cards = [Card.new(x) for x in player_cards]
    if record != []:
        player = get_player(record[-1][0])
    else:
        player = players[0]
    act_mark = get_mark(record)
    actions = game.get_actions(act_mark,player,player_cards)
    if record == []:
        action = RL_1.choose_action(state,actions,game.action_space)
    elif record[-1][0] == players[0]: #地主下家出牌
        action = RL_2.choose_action(state,actions,game.action_space)
    elif record[-1][0] == players[1]: #地主上家出牌
        action = RL_3.choose_action(state,actions,game.action_space)
    elif record[-1][0] == players[2]:   #地主出牌
        action = RL_1.choose_action(state,actions,game.action_space)
    # rule_stradge(action,actions,actions_candy)
    if action not in actions:
        action = random.choice(actions)
    actions.clear()
    return action

def recv_handler(link, client):     
    print("开始接收[%s:%s]的请求...." % (client[0], client[1]))
    while True:     # 利用一个死循环，保持和客户端的通信状态
    # try:
        client_data = link.recv(80960).decode()
        print("[%s:%s]向你发来信息：%s" % (client[0], client[1], client_data))
        action = deal_recv(client_data)
        print(action)
        if action == ['PASS']:
            link.sendall(str(action).encode())
        else:
            card_type = engine.Doudizhu.check_card_type(action)[1][0][0]
            # data = '信息已收到'
            link.sendall(str((action,card_type)).encode())

    # except:
        # data='出错了！请检查传入数据格式！！！'
        # link.sendall(data.encode())
    link.close()

ip_port = ('192.168.1.6', 8888)
sk = socket.socket()            # 创建套接字
sk.bind(ip_port)                # 绑定服务地址
sk.listen(5)                    # 监听连接请求

print('启动socket服务，等待客户端连接...')

while True:     # 一个死循环，不断的接受客户端发来的连接请求
    conn, address = sk.accept()  
    t = threading.Thread(target=recv_handler, args=(conn, address))
    t.start()


























 



