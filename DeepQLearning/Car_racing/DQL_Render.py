from DQL_Image_2_res import *

env = gym.make('CarRacing-v2', continuous=False, render_mode= "human")

n_episodes = 10
agent = Agents(gamma= 0.99, lr = 0.001, batch_size=64, n_actions=env.action_space.n, input_dims=env.observation_space.shape, epsilon=0, eps_dec=1/n_episodes,max_mem_size=10)
agent.Q_eval.load_state_dict(T.load('save_w_car41.pth', map_location=T.device('cpu')))

rewards_per_episode = []

for i in range(n_episodes):
    rewards = 0
    terminated = False
    observation = env.reset()[0]
    steps = 0
    
    while not terminated:
        action = agent.choose_action(observation)
        observation, reward, terminated, _, _ =  env.step(action)
        rewards += reward
        steps += 1
    
    rewards_per_episode.append(rewards)
    mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])
    
    print(f'Episode: {i}, Rewards: {rewards},  Epsilon: {agent.epsilon:0.2f}, Mean Rewards: {mean_rewards:0.1f}')

env.close()
