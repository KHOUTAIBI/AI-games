import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pickle

#definition de l'env
env = gym.make('Pendulum-v1',g=9.81,render_mode=None)
env.reset()

#discretisation de l'env
pos_x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
pos_y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)
ang_vel = np.linspace(env.observation_space.low[2], env.observation_space.high[2], 20)
torque = np.linspace(env.action_space.low[0],env.action_space.high[0],50)

#hyperparameters
episode = 20000
learning_rate_a = 0.1 # alpha or learning rate
discount_factor_g = 0.9 # beta or discount factor.
epsilon = 1     # 1 = 100% random actions
epsilon_decay_rate = 1/episode # epsilon decay rate
rng = np.random.default_rng()   # random number generator
affichage = 1000

#definition qmatrix
q = np.zeros((len(pos_x)+1, len(pos_y)+1, len(ang_vel)+1, len(torque)))

rewards_per_episode = []
#training
for i in range(episode):
    
    state = env.reset()[0]
    state_x = np.digitize(state[0], pos_x)
    state_y = np.digitize(state[1], pos_y)
    state_v = np.digitize(state[2], ang_vel)

    terminated = False
    tronc = False
    rewards=0
    
    while(not terminated and not tronc):
        
        #choose action : random or the best possible
        if rng.random() < epsilon:
            action_index = np.random.randint(len(torque))
        else:
            action_index = np.argmax(q[state_x, state_y, state_v, :])

        #define new states with the action choosen
        action = torque[action_index]
        new_state,reward,terminated,tronc,_ = env.step([action])
        new_state_x = np.digitize(new_state[0], pos_x)
        new_state_y = np.digitize(new_state[1], pos_y)
        new_state_v = np.digitize(new_state[2], ang_vel)

        #belmann equation : update qmatrix
        q[state_x, state_y, state_v, action_index] = q[state_x, state_y, state_v, action_index] + learning_rate_a * (
                reward + discount_factor_g*np.max(q[new_state_x, new_state_y, new_state_v,:]) - q[state_x, state_y, state_v, action_index]
            )

        #update states
        state = new_state
        state_x = new_state_x
        state_y = new_state_y
        state_v = new_state_v

        rewards+=reward
    
    rewards_per_episode.append(rewards)
    mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

    if i%affichage==0:
        print(f'Episode: {i}, Rewards: {rewards},  Epsilon: {epsilon:0.2f}, Mean Rewards: {mean_rewards:0.1f}')

    epsilon = max(epsilon - epsilon_decay_rate, 0)
    if mean_rewards > 300:
        break

env.close()

mean_rewards = []
for t in range(i):
    mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
plt.plot(mean_rewards)

f = open('pendulum.pkl','wb')
pickle.dump(q, f)
f.close()