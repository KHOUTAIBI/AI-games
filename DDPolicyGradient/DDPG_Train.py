from DDPG import *
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
env = gym.make("Pendulum-v1", render_mode = 'human')
training = False
if training:
    agent = Agent(alpha = 0.000025, beta = 0.000025, input_dims = [3], tau = 0.001, env = env, batch_size=64, layer1_size= 400, layer2_size= 300, n_actions=1)
else:
    f=open("StateSaves.pkl",'rb')
    agent =pk.load(f)
    f.close()

np.random.seed(0)
score_history = []
n_games = 500
for i in range(n_games):
    terminated , truncated= False , False
    score =0 
    observation = env.reset()[0]
    while not (terminated or truncated):
        action = agent.choose_action(observation)
        new_state, reward, terminated, truncated, info = env.step(action)
        if training:
            agent.replay(observation,action, reward, new_state, int(terminated or truncated))
            agent.learn()
        score+= reward
        observation = new_state
    score_history.append(score)
    j=0
    meany = np.mean(score_history[-100:])
    print("episode = ", i, "means = ", meany)

if training:
    f=open("StateSaves.pkl",'wb')
    pk.dump(agent,f)
    f.close()
plt.plot(score_history)
plt.savefig("MeansofFly.png")