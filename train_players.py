

import random

import numpy as np
import tensorflow as tf

from doudizhu import Card
from doudizhu.engine import Doudizhu, cards2str
from poke import Game_env
from RL_brain1 import DQNPrioritizedReplay, PolicyGradient, PolicysGradient

#玩家列表
players = ['player1','player2','player3']

tf.compat.v1.disable_eager_execution()

game = Game_env()
# model_name1 = r'Model_pr/player2_0.656000_4000.ckpt'
RL_1 = PolicyGradient(
    game.n_actions, 
    game.n_features,
    )

RL_2 = PolicysGradient(
    game.n_actions, 
    game.n_features,
    )

# model_name = r'Model_pr/player2_0.751667_6000.ckpt'
RL_p = DQNPrioritizedReplay(
        game.n_actions, 
        game.n_features,
    )
# RL_p.load_model(model_name)

def store_transition(record):
    winner = record[-1][0]
    transition1,transition2,transition3 = [],[],[]
    for i in record:
        if i[0] == players[0]:
            reward = game.get_reward(i[1])
            observation = game.get_observation(record[:record.index(i)])
            # observation_ = game.get_observation(record[:record.index(i)+1])
            if players[0] == winner:
                transition1.append((observation,game.action_space.index(i[1]),reward*2))
            else:
                transition1.append((observation,game.action_space.index(i[1]),-0.5*reward))
        elif i[0] == players[1]:
            reward = game.get_reward(i[1])
            observation = game.get_observation(record[:record.index(i)])
            # observation_ = game.get_observation(record[:record.index(i)+1])
            if players[1] == winner:
                transition2.append((observation,game.action_space.index(i[1]),reward*2))
            elif players[2] == winner:
                transition2.append((observation,game.action_space.index(i[1]),reward))
            elif players[0] == winner:
                transition2.append((observation,game.action_space.index(i[1]),-0.5*reward))
        elif i[0] == players[2]:
            reward = game.get_reward(i[1])
            observation = game.get_observation(record[:record.index(i)])
            observation_ = game.get_observation(record[:record.index(i)+1])
            if players[2] == winner:
                transition3.append((observation,game.action_space.index(i[1]),reward*2,observation_))
            elif players[1] == winner:
                transition3.append((observation,game.action_space.index(i[1]),reward,observation_))
            elif players[0] == winner:
                transition3.append((observation,game.action_space.index(i[1]),-0.5*reward,observation_))
    with open("./records/player1_record.txt","a",encoding="utf-8") as f1:
        transition1 = str(transition1).replace('array([','[',len(transition1))
        transition1 = transition1.replace('])',']',len(transition1))
        f1.write(transition1+',')
        # f1.write('\r\n')
    with open("./records/player2_record.txt","a",encoding="utf-8") as f2:
        transition2 = str(transition2).replace('array([','[',len(transition2))
        transition2 = transition2.replace('])',']',len(transition2))
        f2.write(transition2+',')
        # f2.write('\r\n')
    with open("./records/player3_record.txt","a",encoding="utf-8") as f3:
        transition3 = str(transition3).replace('array([','[',2*len(transition3))
        transition3 = transition3.replace('])',']',2*len(transition3))
        f3.write(transition3+',')
        # f3.write('\r\n')

def load_train():
    with open("./records/player1_record.txt","r",encoding="utf-8") as f1:
        f1 = eval(f1.read())
        for trainsitions in f1:
            for trainsition in trainsitions:
                RL_1.store_transition(np.array(trainsition[0]),trainsition[1],trainsition[2])
            RL_1.learn(1)
        RL_1.save_model('player1',str(0.5),5000)
    with open("./records/player2_record.txt","r",encoding="utf-8") as f2:
        f2 = eval(f2.read())
        for trainsitions in f2:
            for trainsition in trainsitions:
                RL_2.store_transition(np.array(trainsition[0]),trainsition[1],trainsition[2])
            RL_2.learn(1)
        RL_2.save_model('player2',str(0.5),5000)
    with open("./records/player3_record.txt","r",encoding="utf-8") as f3:
        f3 = eval(f3.read())
        for trainsitions in f3:
            for trainsition in trainsitions:
                RL_p.store_transition(np.array(trainsition[0]),trainsition[1],trainsition[2],np.array(trainsition[3]))
            RL_p.learn()
            if (f3.index(trainsitions)+1)%50==0:
                RL_p.learn()
        RL_p.save_model('player3',str(0.5),5000)

def game_process():
    win_count = 0
    step = 1 
    #游戏循环
    for i_episode in range(1,5001):
        game_over = False
        act_mark = ('player1',None)
        game = Game_env()
        record = []
        print('游戏开始！')
        state = np.zeros(21+18+18+2)
        round = 1
        while True:
            player,player_pokes,player_real_pokes = game.get_playerinfo(round)
            actions = game.get_actions(act_mark,player,player_pokes)
            if actions == []:
                act_cards = 'pass'
                record.append((player,act_cards))
            else:
                if player == players[0]:
                    act_cards = RL_1.choose_action(state,actions,game.action_space)
                elif player == players[1]:
                    act_cards = RL_2.choose_action(state,actions,game.action_space)
                elif player == players[2]:
                    act_cards = RL_p.choose_action(state,actions,game.action_space)
                if act_cards not in actions:
                    act_cards = random.choice(actions)
                actions.clear()
                if act_cards == 'pass':
                    record.append((player,act_cards))
                else:
                    action,player_real_pokes,player_pokes = game.get_real_action(act_cards,player_real_pokes,player_pokes)
                    if action == []:
                        break
                    else:
                        act_mark = (player,action)
                        record.append((player,act_cards))
                        state_ = game.get_observation(record)
            if len(player_real_pokes) == 0:         #游戏结束
                game_over = True
                winner = record[-1][0]
                store_transition(record)
                if winner in players[1:]:
                    win_count += 1
                win_rate = win_count/i_episode 
                print('本局游戏结束，{}首先出完手牌'.format(player))
                print('i_episode:',i_episode,'win_rate:','%6f'%(win_rate))
                if game_over:
                    break
            round += 1
            state = state_
            step +=1
    # return win_rate,i_episode
game_process()

# load_train()





























