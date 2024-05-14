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

env = gym.make('CartPole-v1',render_mode="human")
env.reset()

pos_space = np.linspace(-2.4, 2.4, 10)
vel_space = np.linspace(-4, 4, 10)
ang_space = np.linspace(-.2095, .2095, 10)
ang_vel_space = np.linspace(-4, 4, 10)

episode = 10
affichage = 1

f = open(file, 'rb')
q = pickle.load(f)
f.close()

rewards_per_episode = []
for i in range(episode):
    
    state = env.reset()[0] 
    state_p = np.digitize(state[0], pos_space)
    state_v = np.digitize(state[1], vel_space)
    state_a = np.digitize(state[2], ang_space)
    state_av = np.digitize(state[3], ang_vel_space)

    terminated = False
    tronc = False
    rewards=0
    
    while(not terminated):
        
        #choose action : the best possible
        action = np.argmax(q[state_p, state_v, state_a, state_av, :])

        #define new states with the action choosen
        state,reward,terminated,tronc,_ = env.step(action)
        state_v = np.digitize(state[0], pos_space)
        new_state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av= np.digitize(state[3], ang_vel_space)

        rewards+=reward
    
    rewards_per_episode.append(rewards)
    mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

    if i%affichage==0:
        print(f'Episode: {i}, Rewards: {rewards}, Mean Rewards: {mean_rewards:0.1f}')

env.close()