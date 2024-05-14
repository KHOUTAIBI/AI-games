from DQL_Image_2_res import *

env = gym.make('CarRacing-v2', continuous=False, render_mode= None)
env = SkipFrame(env, skip=4)
env = wrap.ResizeObservation(env,84)
env = wrap.GrayScaleObservation(env)
env = wrap.FrameStack(env, 4)
n_episodes = 10000
MAX_PENALTY = -5
load = True
agent = Agents(gamma= 0.95, lr = 0.001, batch_size=65, n_actions=env.action_space.n, input_dims=env.observation_space.shape, epsilon=0.2, eps_dec=0.9999925,eps_end=0.01)
target_res = DeepQNetwork(env.observation_space.shape, env.action_space.n).to(T.device('cuda:2' if T.cuda.is_available() else 'cpu'))
if load:
    agent.Q_eval.load_state_dict(T.load('./saves/save_w_checkpoint_car_0.pth', map_location=T.device('cuda:2')))
if T.cuda.is_available():
    print("Running on GPU :", T.cuda.get_device_name(0))
else:
    print("Running on CPU")

SKIP_FRAMES = 2


rewards_per_episode = []
print("=====Starting=====")
try:
   # for j in range(0,agent.min_mem +1):
   #     observation = env.reset()[0]
   #     action = np.random.choice(agent.action_space)
   #     observation_neu, reward, terminated, _, info = env.step(action)
   #     agent.memory.store_transition(observation,action,reward,observation_neu,terminated)
   #     observation = observation_neu
   #     if j%100 == 0:
   #         print("Completed training of iteration number :", j)
   # agent.mem_cntr = 40001
    for i in range(n_episodes+1):
        rewards = 0
        time = 0
        terminated = False
        truncated = False
        observation = env.reset()[0]
        steps = 0
        negative_reward_counter = 0
        latest_rewards = []
        reward_terminate = False # garde historique des dernières récompenses, les 500 dernières doivent avoir une somme positive
        while not (terminated or truncated):
            action = agent.choose_action(observation)
            reward = 0
#                if len(latest_rewards) >500:
#                    latest_rewards.pop(0)
#                    if sum(latest_rewards) <0:
#                        reward_terminate = True
            observation_neu, reward, done, truncated, info = env.step(action)
            #if action == 3:
            #    reward *= 1.5
            #negative_reward_counter = negative_reward_counter + 1 if steps > 100 and reward < 0 else 0
            #if negative_reward_counter > 5:
            #    break
            
            #reward = np.clip( reward, a_max=1, a_min=-10 )
            agent.memory.store_transition(observation,action,reward,observation_neu,terminated)
            agent.learn(target_res,terminated)


            # clip reward to 1. Really fast driving would otherwise receive higher reward which
            # is not good for tight turns. Additionally, skipping frames might produce really big
            # rewards as well (x1 tile is ~ 3 pnts so clip 6 is max 2 tiles reward per step).
            observation = observation_neu
            rewards += reward
            #steps += 1

        #agent.epsilon = np.interp(i, [0, n_episodes-50], [1.0, 0.1]) # split the interval and stay in the eps range for epsilon greedy


        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-30:])


        # Update target weights
        if i % 10 == 0:
            target_res.load_state_dict(agent.Q_eval.state_dict())


        if i%30==0:
            text = "save_w_checkpoint_car_"+str(0)+".pth"
            T.save(agent.Q_eval.state_dict(),"./saves/"+text)
            print(f'Episode: {i}, Rewards: {rewards},  Epsilon: {agent.epsilon:0.2f}, Mean Rewards: {mean_rewards:0.1f}')

    env.close()
    T.save(agent.Q_eval.state_dict(), "./saves/save_w_car_0.pth")
    plt.plot(rewards_per_episode)
    plt.xlabel("epochs")
    plt.ylabel("rewards")
    plt.savefig("./plots/rewardplot.png")

except KeyboardInterrupt:
    env.close()
    T.save(agent.Q_eval.state_dict(), "./saves/save_w_interrupt.pth")
    plt.plot(rewards_per_episode)
    plt.xlabel("epochs")
    plt.ylabel("rewards")
    plt.savefig("./plots/rewardplot.png")
    sys.exit()
