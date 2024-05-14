from DQL_image_with_skip import *

env = gym.make("PongnoFrameskip-v4", render_mode="human")
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
recorder = False
if recorder:
    env = RecordVideo(env, video_folder="./", name_prefix="eval",
                      episode_trigger=lambda x: x %100 == 0)
n_episodes = 100
agent = Agents(gamma= 0.99, lr = 0.000025, batch_size=32, n_actions=env.action_space.n, input_dims=env.observation_space.shape, epsilon=0.01, eps_dec=1/n_episodes,max_mem_size=10)

agent.Q_eval.load_state_dict(T.load('save_w_checkpoint_0.pth', map_location=T.device('cpu')))

rewards_per_episode = []
for i in range(n_episodes):
    rewards = 0
    terminated = False
    observation,_ = env.reset()
    observation = env.step(1)[0]
    steps = 0
    nbr_lives = 5
    while not terminated:
        action = agent.choose_action(observation)
        new_observation, reward, terminated, _, info =  env.step(action)
        rewards += reward
        observation = new_observation
        if nbr_lives > int(info["lives"]):
            nbr_lives = int(info["lives"])
            observation = env.step(1)[0]
        steps += 1

    rewards_per_episode.append(rewards)
    mean_rewards = np.mean(rewards_per_episode[-100:])
    
    print(f'Episode: {i}, Rewards: {rewards},  Epsilon: {agent.epsilon:0.2f}, Mean Rewards: {mean_rewards:0.1f}')

env.close()
