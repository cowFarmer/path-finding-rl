import random
import numpy as np
from Sim import Simulator

class Policy():
    def __init__(self):
        pass
    
    def select_action(self):
        pass
    
#     def predict_pi(self):
#         pass
    
#     def predict_value(self):
#         pass
    
#     def train_net(self):
#         pass

class RandomPolicy(Policy):
    def __init__(self):
        pass
    
    def select_action(self):
        coin = random.randint(0,3)
        return coin

# off-policy q-learning
class QLearning(Policy):
    def __init__(self):
#         self.q_table = np.zeros((sim.height, sim.width, 4), dtype="float16")
        self.q_table = np.zeros((10, 9, 4), dtype="float16")
        self.eps = 0.9
        self.alpha = 0.01
    
    def select_action(self, state):
        x, y = state
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0,3)
        else:
            action_val = self.q_table[x,y,:]
            action = np.argmax(action_val)
            print(f"policy action = {action}")
        return action
    
    def update_table(self, history):
        s,a,r,s_prime = transition
        x, y = s
        next_x, next_y = s_prime
        # Q-learning update
        self.q_table[x,y,a] = self.q_table[x,y,a] + 0.1 * (r + np.amax(self.q_table[next_x,next_y,:]) - self.q_table[x,y,a])
        
    def anneal_eps(self):
        self.eps -= 0.01 # epsilon
        self.eps = max(self.eps, 0.2)
        
    def show_table(self):
        q_lst = self.q_table.tolist()
        data = np.zeors((10,9,4))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)
        