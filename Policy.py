import random

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