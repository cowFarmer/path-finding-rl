import pandas as pd
from Sim import *
from Agent import Agent


if __name__ == "__main__":
    files = pd.read_csv("./data/factory_order_train.csv")
    
    sim = Simulator() # env, visualization
    agent = Agent()
    mode = "q_learning" # random,q_learning,dqn,... ∈ set_policy
    
    # 평가를 위한 변수
    episode = 0
    timestep = 0
    average_reward = sim.goal_reward * 10
    
    agent.set_policy(mode)
    print(f"mode = {agent.set_policy(mode)}")
   
    for epi in range(1): # len(files)):
        episode += 1
        
        # env 초기화
        obs = sim.reset(epi) # obs = sim.reset(epi)
        
        print(f"{epi}번째 에피소드 시작")
        print(f"obs = \n{obs}")
        print(f"order box = {sim.local_target}")
        
        # agent에게 local_target 주기
        items = list(files.iloc[epi])[0]
        agent.set_target_list(items)
        print(items)
        done = False
        
        print(f"시작 state = {sim.curloc}") # [9,4]
        
        while done == False:
            timestep += 1
            
            # policy에 맞춰서 agent가 action하기
            action_list = ['up','down','left','right']
            action = agent.select_action(sim.curloc)
            print(f"action = {action_list[action]}")
            print(f"현재 state = {sim.curloc}")
            next_obs, reward, cumul ,done, goal_reward = sim.step(action)
            
            
            obs = next_obs # 다음 episode
            
            
#     print(f"Episode : {episode}, Timestep : {timestep} \nAverage Reward : {}, Finish Rate : {}")