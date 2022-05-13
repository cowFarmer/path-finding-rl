from Policy import RandomPolicy

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
    
    def select_action(self):
        a = self.policy.select_action()
        return a
    
    def set_policy(self, mode):
        if mode=='random':
            self.policy = RandomPolicy()
            print(r"Agent's Policy = random")