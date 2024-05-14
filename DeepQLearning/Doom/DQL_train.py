from DQL_image_with_skip import *

env = gym.make("BreakoutNoFrameskip-v4", render_mode=None)
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
n_episodes = 8000

SKIP_FRAMES = 0

agent = Agents(gamma= 0.99, lr = 0.001, batch_size=65, n_actions=env.action_space.n, input_dims=(4,84,84), epsilon=1, eps_dec=0.999)
target_res = DeepQNetwork(env.observation_space.shape, env.action_space.n).to(T.device('cuda:1' if T.cuda.is_available() else 'cpu'))

if T.cuda.is_available():
    print("Running on GPU :", T.cuda.get_device_name(0))
else:
    print("Running on CPU")

rewards_per_episode = []


frame_repeat = 12 # Repeat action for nb of frames
config_file_path = os.path.join(vzd.scenarios_path, "simpler_basic.cfg")


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game


game = create_simple_game()
n = game.get_available_buttons_size()



print("=====Starting=====")

try:
    for i in range(n_episodes+1):
        rewards = 0
        time = 0
        terminated = False
        tronc = False
        game.new_episode()
        steps = 0
        nbr_lives = 5
        while not terminated:
            action = agent.choose_action(observation)
            reward = 0
            reward = game.make_action(action, frame_repeat)
            done = game.is_episode_finished()
            if not done:
                observation_neu = preprocess(game.get_state().screen_buffer)
            else:
                observation_neu = np.zeros((1, 30, 45)).astype(np.float32)
            if nbr_lives > int(info["lives"]):
                nbr_lives = int(info["lives"])
                reward -= 1
                if nbr_lives < 4:
                    terminated = True
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

except KeyboardInterrupt:
    env.close()
    T.save(agent.Q_eval.state_dict(), "save_w_interrupt.pth")
    sys.exit()