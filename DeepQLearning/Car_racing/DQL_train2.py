from DQL_Image import *

env = gym.make('CarRacing-v2', continuous=False, render_mode= None)

n_episodes = 400

agent = Agents(gamma= 0.99, lr = 0.001, batch_size=64, n_actions=env.action_space.n, input_dims=env.observation_space.shape, epsilon=1, eps_dec=0.05)
if T.cuda.is_available():
    print("Running on GPU :", T.cuda.get_device_name(0))
else:
    print("Running on CPU")

SKIP_FRAMES = 2


rewards_per_episode = []
print("=====Starting=====")
try:
    for i in range(n_episodes+1):
        rewards = 0
        time = 0
        terminated = False
        tronc = False
        observation = env.reset()[0]
        steps = 0
        negative_reward_counter = 0

        while True:
            action = agent.choose_action(observation)
            reward = 0
            for _ in range(SKIP_FRAMES+1):
                observation_neu, r, done, info, _ = env.step(action)
                reward += r
                if done:
                    break

            negative_reward_counter = negative_reward_counter + 1 if steps > 100 and reward < 0 else 0

            if action == 3:
                reward *= 1.5

            agent.memory.store_transition(observation,action,reward,observation_neu,terminated)
            agent.learn()


            observation = observation_neu
            rewards += reward

            if done or negative_reward_counter >= 25 or rewards < 0:
                break


        #agent.epsilon = max(agent.epsilon - agent.eps_dec, agent.eps_min)
        agent.epsilon = np.interp(i, [0, n_episodes-50], [1.0, 0.1]) # split the interval and stay in the eps range for epsilon greedy


        #agent.epsilon = max(agent.epsilon - agent.eps_dec, agent.eps_min)
        #agent.epsilon = np.interp(i, [0, n_episodes], [1.0, 0.02]) # split the interval and stay in the eps range for epsilon greedy
        #agent.epsilon = max(agent.eps_min, np.exp(-agent.eps_dec * i))
        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

        if i%int(n_episodes/100)==0:
            text = "save_w_checkpoint_"+str(i)+".pth"
            #T.save(agent.Q_eval.state_dict(),"./saves/"+text)
            print(f'Episode: {i}, Rewards: {rewards},  Epsilon: {agent.epsilon:0.2f}, Mean Rewards: {mean_rewards:0.1f}')

    env.close()


    T.save(agent.Q_eval.state_dict(), "save_w_car11.pth")

    #plt.plot(rewards_per_episode)
    #plt.savefig("Means5.png")
except KeyboardInterrupt:
    env.close()
    #f = open("Saves2.pkl",'wb')
    #pk.dump(agent,f)
    #f.close()
    T.save(agent.Q_eval.state_dict(), "save_w_interrupt.pth")
    plt.plot(rewards_per_episode)
    #plt.savefig("Means5.png")
    sys.exit()
