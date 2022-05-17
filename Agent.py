import numpy as np
from Policy import *


class Agent():
    def __init__(self):
        self.target_list=None
        self.policy = None
        
    def set_target_list(self, local_target):
        self.target_list = local_target
        print("Agent's local target = {}".format(local_target))
    
    def reset(self):
        pass
    
    def update_target(self):
        pass
    
    # 현재 state를 받고 policy의 action하기
    # action_list = ['up','down','left','right'] = 0,1,2,3
    def select_action(self, state):
        # 진열대 array rack(좌 상 우, 세 종류) 만들어서 isin으로 체크하고 위치에 따라 행동 return 만들기
        left_rack = [5,0], [4,0], [3,0], [2,0]
        top_rack = [0,0], [0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7], [0,8]
        right_rack = [2,8], [3,8], [4,8], [5,8]
        if state == [9,4]:
            return 0 # up
        elif state in top_rack:
            return 1 # down
        elif state in right_rack:
            return 2 # left
        elif state in left_rack:
            return 3 # right
        else:
            action = self.policy.select_action(state)
            return action
    
    def set_policy(self, mode):
        if mode=="random":
            self.policy = RandomPolicy()
        elif mode=="q_learning":
            self.policy = QLearning()
        else:
            print("no")