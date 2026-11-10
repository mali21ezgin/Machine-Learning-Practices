import gymnasium as gym
import random 
import numpy as np 
import random
from tqdm import tqdm
import matplotlib.pyplot as plt






env = gym.make("CliffWalking-v1", render_mode="ansi")


action_space = env.action_space.n
state_space = env.observation_space.n

q_table = np.zeros((state_space, action_space))

alpha = 0.1
gamma = 0.6
e = 0.1

for i in tqdm(range(1,10001)):
    state, _ = env.reset()
    
    done = False
    
    while not done:
        
        if random.uniform(0,1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        next_state,reward,terminated, truncated,info = env.step(action)
        
        done = terminated or truncated
        
        q_table[state,action] = q_table[state,action] + alpha * (reward + gamma *np.max(q_table[next_state]) - q_table[state,action])
        state = next_state


total_epoch, total_penalties = 0,0 
eposides = 100

       
for i in tqdm(range(eposides)):
    state, _ = env.reset()
    epochs, penalties, reward = 0,0,0
    done = False
    while not done:

    


        action = np.argmax(q_table[state])       
        next_state,reward,terminated, truncated,info = env.step(action)
        done = terminated or truncated  
        state = next_state
        
        if reward ==-100:
            penalties +=1
        epochs +=1
            
    total_epoch +=epochs
    total_penalties += penalties
    
print(f"Test bitti! Ortalama Adım: {total_epoch / eposides}, Toplam Ceza: {total_penalties}")