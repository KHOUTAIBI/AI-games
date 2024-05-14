from DQL_Image_2_res import *

env = gym.make('CarRacing-v2', continuous=False, render_mode= None)

n_episodes = 1000
MAX_PENALTY = -5

agent = Agents(gamma= 0.99, lr = 0.001, batch_size=65, n_actions=env.action_space.n, input_dims=env.observation_space.shape, epsilon=1, eps_dec=0.9999)
target_res = DeepQNetwork(env.observation_space.shape, env.action_space.n).to(T.device('cuda:1' if T.cuda.is_available() else 'cpu'))

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
        latest_rewards = []
        reward_terminate = False # garde historique des dernières récompenses, les 500 dernières doivent avoir une somme positive

        while not terminated and rewards > MAX_PENALTY and not reward_terminate:
            action = agent.choose_action(observation)
            reward = 0
            for _ in range(SKIP_FRAMES+1):
                observation_neu, r, done, info, _ = env.step(action)
                latest_rewards.append(r)
                reward += r
                if done:
                    break

                if len(latest_rewards) >500:
                    latest_rewards.pop(0)
                    if sum(latest_rewards) <0:
                        reward_terminate = True

            negative_reward_counter = negative_reward_counter + 1 if steps > 100 and reward < 0 else 0
            if negative_reward_counter > 5:
                break

            agent.memory.store_transition(observation,action,reward,observation_neu,terminated)
            agent.learn(target_res)


            # clip reward to 1. Really fast driving would otherwise receive higher reward which
            # is not good for tight turns. Additionally, skipping frames might produce really big
            # rewards as well (x1 tile is ~ 3 pnts so clip 6 is max 2 tiles reward per step).
            reward = np.clip( reward, a_max=1, a_min=-10 )

            observation = observation_neu
            rewards += reward
            steps += 1

        #agent.epsilon = np.interp(i, [0, n_episodes-50], [1.0, 0.1]) # split the interval and stay in the eps range for epsilon greedy


        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])


        # Update target weights
        if i % 5 == 0:
            target_res.load_state_dict(agent.Q_eval.state_dict())


        if i%int(n_episodes/100)==0:
            text = "save_w_checkpoint_"+str(i)+".pth"
            T.save(agent.Q_eval.state_dict(),"./saves/"+text)
            print(f'Episode: {i}, Rewards: {rewards},  Epsilon: {agent.epsilon:0.2f}, Mean Rewards: {mean_rewards:0.1f}')

    env.close()
    T.save(agent.Q_eval.state_dict(), "save_w_car40.pth")

except KeyboardInterrupt:
    env.close()
    T.save(agent.Q_eval.state_dict(), "save_w_interrupt.pth")
    sys.exit()
