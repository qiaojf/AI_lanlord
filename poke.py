
import random

import numpy as np

from doudizhu import Card, list_greater_cards, new_game
from doudizhu.engine import Doudizhu, cards2str, sort_cards

#玩家列表
players = ['player1','player2','player3']
#出牌规则
card_type = Doudizhu.CARD_TYPE
STR_RANKS = '3-4-5-6-7-8-9-10-J-Q-K-A-2-BJ-CJ'.split('-')
INT_RANKS = range(1,16)
CHAR_RANK_TO_INT_RANK = dict(zip(STR_RANKS, INT_RANKS))

class Game_env(object):
    cards_candidate = []
    #初始化
    def __init__(self):
        self.action_space = self.space()
        self.reward1 = 0
        self.reward2 = 0
        self.reward3 = 0
        self.n_actions = len(self.action_space)
        self.bomb_list = self.get_bomb()
        self.pokes = self._deal()
        self.cards = ['3d', '3c', '3h', '3s', '4d', '4h', '4s', '4c', '5s', '5h', 
        '5d', '5c', '6c', '6h', '6d', '6s', '7s', '7d', '7c', '7h', '8d', '8c',
         '8s', '8h', '9s', '9d', '9h', '9c', '10s', '10d', '10h', '10c', 'Js', 
         'Jc', 'Jd', 'Jh', 'Qh', 'Qc', 'Qs', 'Qd', 'Kd', 'Ks', 'Kc', 'Kh', 'As',
          'Ah', 'Ac', 'Ad', '2d', '2h', '2s', '2c', 'BJ', 'CJ']
        # self.state = np.zeros(len(self.cards)) 
        # self.leftcards_onehot = self.left_cards()
        self.n_features = 21+18+18+2

    #出牌动作列表
    def space(self):
        action_space = []
        for i in range(len(card_type)):
                actions_list = card_type[i]['func']() #actions_list=[('BJ-CJ', 0)]
                for j in actions_list:    #j=('BJ-CJ', 0)
                    action = [a for a in j[0].split('-')]  #action=['BJ', 'CJ']
                    action_space.append(action)
        action_space.append('pass')
        return action_space
    #发牌，返回玩家手牌
    def _deal(self):
        #初始玩家手牌列表
        player1_real_pokes = []
        player2_real_pokes = []
        player3_real_pokes = []

        #开始游戏，生产牌列表
        cards_groups = new_game()
        #玩家手牌赋值
        player1_pokes = cards_groups[0]
        # print(player1_pokes)
        player2_pokes = cards_groups[1]
        player3_pokes = cards_groups[2]
        #地主牌
        last_poke = cards_groups[3]
        #将地主牌给玩家1
        for i in last_poke:
            player1_pokes.append(i)

        #将数值手牌转变为带花色的牌
        for i in player1_pokes:
            player1_real_pokes.append(Card.int_to_str(i))
        for i in player2_pokes:
            player2_real_pokes.append(Card.int_to_str(i))
        for i in player3_pokes:
            player3_real_pokes.append(Card.int_to_str(i))
        #将有大小王玩家的空格去除
        for i in range(len(player1_real_pokes)):
            player1_real_pokes[i]=player1_real_pokes[i].replace(' ','')
        for i in range(len(player2_real_pokes)):
            player2_real_pokes[i]=player2_real_pokes[i].replace(' ','')
        for i in range(len(player3_real_pokes)):
            player3_real_pokes[i]=player3_real_pokes[i].replace(' ','')
        
        return player1_pokes,player2_pokes,player3_pokes,player1_real_pokes,player2_real_pokes,player3_real_pokes

    #将牌转变为无花色牌
    def no_suit(self,pokes):
        no_suit_cards = Card.cards_without_suit(pokes)
        cards_str = [card_str for card_str in no_suit_cards.split('-')]
        return cards_str

    #打印候选列表
    def PrettyPrint(self,cards_gt):
        for card_type, cards_list in cards_gt.items():
            print('card type: {},'.format(card_type),'cards_candidate numbers:',len(cards_list))
            for card_int in cards_list:
                print(cards_list.index(card_int),end='')
                Card.print_pretty_cards(list(card_int))
        return cards_list

    def list_candidate(self,cards_str):  #候选牌列表
        for i in range(len(card_type)):
            actions_list = card_type[i]['func']() #actions_list=[('BJ-CJ', 0)]
            for j in actions_list:    #j=('BJ-CJ', 0)
                action = [a for a in j[0].split('-')]  #action=['BJ', 'CJ']
                flag = True
                for k in action:
                    if  cards_str.count(k) >= action.count(k):
                        pass
                    else:
                        flag = False
                        break
                if flag == True:
                    self.cards_candidate.append(action)
        actions =self.cards_candidate
        return actions

    def get_bomb(self):                 #获取炸弹列表（包括双王）
        bomb_list = []
        for i in card_type[-2:]:
            for j in i['func']():
                bomb_list.append(j[0])
        for i in range(len(bomb_list)):
            bomb_list[i]=bomb_list[i].split('-')
        return bomb_list
    #获取无花色牌动作列表
    def get_actions(self,act_mark,player,pokes):
        #主动出牌时动作列表
        if act_mark[0] == player:
            actions = self.list_candidate(self.no_suit(pokes))
        #被动出牌时动作列表
        else:
            rival_cards = Card.card_ints_from_string(cards2str(act_mark[1]))
            actions = list_greater_cards(rival_cards,pokes)
            if len(actions) == 0:
                actions = ['pass']
            else:
                actions = [(k,v) for k,v in actions.items()]
                actions = actions[0][1]
                for i in range(len(actions)):
                    actions[i] = self.no_suit(actions[i])
                actions.append('pass')
        return actions

                
    #将无花色动作变为有花色动作
    def get_real_action(self,nosuit_action,player_real_pokes,player_pokes):
        real_action = []
        for i in range(len(nosuit_action)):   #['10']
            if Card.is_joker(nosuit_action[i]):
                player_real_pokes.remove(nosuit_action[i])
                real_action.append(nosuit_action[i])
            else:
                real_cards = Card.card_rank_to_real_card(nosuit_action[i])
                for j in real_cards:
                    if j in player_real_pokes:
                        player_real_pokes.remove(j)
                        real_action.append(j)
                        break
        for i in real_action:
            player_pokes.remove(Card.new(i))                                  
        return real_action,player_real_pokes,player_pokes

    def get_playerinfo(self,round):
        if round%3 == 1:
            return players[0],self.pokes[0],self.pokes[3]
        elif round%3 ==2:
            return players[1],self.pokes[1],self.pokes[4]
        else:
            return players[2],self.pokes[2],self.pokes[5]

    def get_reward(self,act_cards):
        if act_cards == 'pass' or len(act_cards) == 1:
            reward = 0
        elif act_cards in self.bomb_list:
            reward = 2*len(act_cards)
        else:
            reward = len(act_cards)
        return reward

    def get_observation(self,record):
        player1_record = []
        player2_record = []
        player3_record = []
        observation = np.zeros(21+18+18+2)
        if record != []:
            records =[x for x in record if x[1]!='pass']
            act_mark = records[-1] 
            for i in records:
                if i[0] == players[0]:
                    for j in i[1]:
                        player1_record.append(CHAR_RANK_TO_INT_RANK[j])
                if i[0] == players[1]:
                    for j in i[1]:
                        player2_record.append(CHAR_RANK_TO_INT_RANK[j])
                if i[0] == players[2]:
                    for j in i[1]:
                        player3_record.append(CHAR_RANK_TO_INT_RANK[j])
            for i in range (len(player1_record)):
                observation[i]=player1_record[i]
            observation[20] = 20-len(player1_record)
            for i in range (len(player2_record)):
                observation[i+21] = player2_record[i]
            observation[38] = 17-len(player2_record)
            for i in range (len(player3_record)):
                observation[i+39] = player3_record[i]
            observation[56] = 17-len(player3_record)
            observation[57] = players.index(act_mark[0])+1
            observation[58] = self.action_space.index(act_mark[1])
        return observation

    





              





