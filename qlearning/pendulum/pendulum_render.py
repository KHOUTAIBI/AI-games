import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pickle
import sys

file = 0
if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    print("No file given")

assert isinstance(file,str)

env = gym.make('Pendulum-v1',g=9.81,render_mode="human")
env.reset()

pos_x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
pos_y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)
ang_vel = np.linspace(env.observation_space.low[2], env.observation_space.high[2], 20)
torque = np.linspace(env.action_space.low[0],env.action_space.high[0],50)

episode = 10
affichage = 1

f = open(file, 'rb')
q = pickle.load(f)
f.close()

rewards_per_episode = []
for i in range(episode):
    
    state = env.reset()[0]
    state_x = np.digitize(state[0], pos_x)
    state_y = np.digitize(state[1], pos_y)
    state_v = np.digitize(state[2], ang_vel)

    terminated = False
    rewards=0
    
    while(not terminated):
        
        #choose action : the best possible
        action_index = np.argmax(q[state_x, state_y, state_v, :])
        action = torque[action_index]

        #define new states with the action choosen
        state,reward,terminated,_,_ = env.step([action])
        state_x = np.digitize(state[0], pos_x)
        state_y = np.digitize(state[1], pos_y)
        state_v = np.digitize(state[2], ang_vel)

        rewards+=reward
    
    rewards_per_episode.append(rewards)
    mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

    if i%affichage==0:
        print(f'Episode: {i}, Rewards: {rewards}, Mean Rewards: {mean_rewards:0.1f}')

env.close()