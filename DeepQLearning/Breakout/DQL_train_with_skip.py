from DQL_image_with_skip import *

env = gym.make("BreakoutNoFrameskip-v4", render_mode=None)
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
recorder = False
if recorder:
    env = RecordVideo(env, video_folder="./videos", name_prefix="Breakout",
                      episode_trigger=lambda x: x %100 == 0)
n_episodes = 10000

SKIP_FRAMES = 0

agent = Agents(gamma= 0.99, lr = 0.000025, batch_size=32, n_actions=env.action_space.n, input_dims=(4,84,84), epsilon=0.5, eps_dec=0.9999)
target_res = DeepQNetwork(env.observation_space.shape, env.action_space.n).to(T.device('cuda:1' if T.cuda.is_available() else 'cpu'))

if T.cuda.is_available():
    print("Running on GPU :", T.cuda.get_device_name(0))
else:
    print("Running on CPU")

rewards_per_episode = []

print("=====Starting=====")

try:
    observation = env.reset()[0]
    terminated = False
    for j in range(0,agent.min_mem +1):
        action = np.random.choice(agent.action_space)
        observation_neu, reward, terminated, _, info = env.step(action)
        agent.memory.store_transition(observation,action,reward,observation_neu,terminated)
        observation = observation_neu
        if j%100 == 0:
            print("Completed training of iteration number :", j)
    agent.mem_cntr = 40001
    for i in range(n_episodes+1):
        rewards = 0
        time = 0
        terminated = False
        tronc = False
        observation = env.reset()[0]
        observation = env.step(1)[0]
        steps = 0
        nbr_lives = 5
        while not terminated:
            action = agent.choose_action(observation)
            reward = 0
            observation_neu, reward, terminated, _, info = env.step(action)
            if nbr_lives > int(info["lives"]):
                nbr_lives = int(info["lives"])
                reward -= 1
            reward = np.clip( reward, a_min=-1, a_max=1 )
            agent.memory.store_transition(observation,action,reward,observation_neu,terminated)
            agent.learn(target_res,terminated)
            observation  = observation_neu
            rewards += reward
            steps += 1
            agent.epsilon = max(agent.epsilon*agent.eps_dec, agent.eps_min)

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])


        # Update target weights
        if i % 10 == 0:
            target_res.load_state_dict(agent.Q_eval.state_dict())
        if i%100==0:
            text = "save_w_checkpoint_"+str(0)+".pth"
            T.save(agent.Q_eval.state_dict(),"./"+text)
            print(f'Episode: {i}, Rewards: {rewards},  Epsilon: {agent.epsilon:0.2f}, Mean Rewards: {mean_rewards:0.1f}')
    env.close()
    T.save(agent.Q_eval.state_dict(), "save_w_break.pth")
    plt.plot(rewards_per_episode)
    plt.xlabel("epochs")
    plt.ylabel("rewards")
    plt.savefig("./plots/rewardplot.png")
except KeyboardInterrupt:
    env.close()
    T.save(agent.Q_eval.state_dict(), "save_w_interrupt.pth")
    plt.plot(rewards_per_episode)
    plt.xlabel("epochs")
    plt.ylabel("rewards")
    plt.savefig("./plots/rewardplot.png")
    sys.exit()
