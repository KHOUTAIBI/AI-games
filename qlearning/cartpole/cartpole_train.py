import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pickle

#definition de l'env
env = gym.make('CartPole-v1',render_mode=None)
env.reset()

#discretisation de l'env
pos_space = np.linspace(-2.4, 2.4, 10)
vel_space = np.linspace(-4, 4, 10)
ang_space = np.linspace(-.2095, .2095, 10)
ang_vel_space = np.linspace(-4, 4, 10)

#hyperparameters
episode = 10000
learning_rate_a = 0.1 # alpha or learning rate
discount_factor_g = 0.99 # beta or discount factor.
epsilon = 1     # 1 = 100% random actions
epsilon_decay_rate = 1/episode # epsilon decay rate
rng = np.random.default_rng()   # random number generator
affichage = 1000

#definition qmatrix
q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))

rewards_per_episode = []
#training
for i in range(episode):
    
    state = env.reset()[0] 
    state_p = np.digitize(state[0], pos_space)
    state_v = np.digitize(state[1], vel_space)
    state_a = np.digitize(state[2], ang_space)
    state_av = np.digitize(state[3], ang_vel_space)

    terminated = False
    tronc = False
    rewards=0
    
    while(not terminated and not tronc):
        
        #choose action : random or the best possible
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state_p, state_v, state_a, state_av, :])

        #define new states with the action choosen
        new_state,reward,terminated,tronc,_ = env.step(action)
        new_state_p = np.digitize(new_state[0], pos_space)
        new_state_v = np.digitize(new_state[1], vel_space)
        new_state_a = np.digitize(new_state[2], ang_space)
        new_state_av= np.digitize(new_state[3], ang_vel_space)

        #belmann equation : update qmatrix
        q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                reward + discount_factor_g*np.max(q[new_state_p, new_state_v, new_state_a, new_state_av,:]) - q[state_p, state_v, state_a, state_av, action]
        )

        # update states
        state = new_state
        state_p = new_state_p
        state_v = new_state_v
        state_a = new_state_a
        state_av= new_state_av

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

f = open('cartpole.pkl','wb')
pickle.dump(q, f)
f.close()