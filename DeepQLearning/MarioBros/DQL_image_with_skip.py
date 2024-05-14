import torch as T
import torch.nn as nn
import numpy as np
import gym_super_mario_bros
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optimum
import  gym.wrappers as wrap
#import gymnasium as gym
from nes_py.wrappers import JoypadSpace
import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import sys
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.spaces import Box
class DeepQNetwork(nn.Module):
    def __init__(self, input_dims,n_actions):
        super(DeepQNetwork,self).__init__()

        self.input_dims = input_dims
        self.conv1 = nn.Conv2d(4,32,8,stride=4)
        self.conv2 = nn.Conv2d(32,64,4,stride=2)
        self.conv3 = nn.Conv2d(64,64,3,stride=1)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, n_actions)
        self.relu = nn.ReLU()


    def forward(self, x):

        # Convert the image to grayscale
        x = T.as_tensor(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x).flatten(start_dim=1))
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class Memory():
    def __init__(self,input_dims,max_mem_size = 20000):
        self.mem_size = max_mem_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.new_state_mem = np.zeros((self.mem_size,*input_dims),dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size,dtype= np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype= np.float32)
        self.terminal_memory = np.zeros(self.mem_size,dtype = bool)

    def store_transition(self, state, action, reward, state_neu, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_mem[index] = state_neu
        self.reward_memory [index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr +=1


class Agents():
    def __init__(self, gamma, lr, batch_size, n_actions, input_dims, epsilon, eps_end=0.1, eps_dec = 0.9999, max_mem_size = 20000, min_mem = 4000):
        self.device = T.device('cuda:1' if T.cuda.is_available() else 'cpu')
        self.gamma= gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.min_mem = min_mem
        self.action_space  = [i for i in range(n_actions)]
        self.batch_size = batch_size

        self.memory = Memory(input_dims,max_mem_size)

        self.Q_eval = DeepQNetwork(input_dims, n_actions).to(self.device)
        self.optimizer = optimum.Adam(self.Q_eval.parameters(),lr=self.lr,eps=1e-7)
        self.loss_fn = nn.SmoothL1Loss()
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = np.array([observation],dtype=np.float32)
            state = T.tensor(state).to(self.device)
            actions = self.Q_eval(state)
            return T.argmax(actions).item()
        else:
            return np.random.choice(self.action_space)

    def learn(self, target_res , terminated):
        if self.memory.mem_cntr < self.min_mem:
            return

        max_mem = min(self.memory.mem_cntr, self.memory.mem_size)
        batch = np.random.choice(max_mem,self.batch_size, replace= False)
        batch_index = np.arange(self.batch_size,dtype = np.int32)

        state_batch =  T.tensor(self.memory.state_memory[batch]).to(self.device)
        new_state_batch = T.tensor(self.memory.new_state_mem[batch]).to(self.device)
        reward_batch = T.tensor(self.memory.reward_memory[batch]).to(self.device)
        terminal_batch = T.tensor(self.memory.terminal_memory[batch]).to(self.device)

        action_batch = self.memory.action_memory[batch]

        q_eval = self.Q_eval(state_batch)[batch_index,action_batch]
        q_next = target_res(new_state_batch)
        #q_next[terminal_batch] = 0.0

        q_target = reward_batch + (1 - terminated)*self.gamma * T.max(q_next, dim=1)[0]

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_target, q_eval).to(self.device)
        loss.backward()
        self.optimizer.step()

        #self.epsilon = max(self.epsilon*self.eps_dec, self.eps_min)


def update_stack_frame(buffer_obs, new_state):
    #return np.array([observation[1],observation[2],observation[3],new_state])
    cropped_buffer = buffer_obs[1:,:,:]
    return np.concatenate([new_state,cropped_buffer], axis=0)


def process_image(x):
    x = T.as_tensor(x)
    x = x.permute(2, 0, 1) # mettre la couleur au début
    x = transforms.functional.crop(x, top=34,left=0,height=160,width=160)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Converts RGB to grayscale
        transforms.Resize([84, 84], interpolation=transforms.InterpolationMode.NEAREST),  # Resizes the image
    ])
    x = transform(x)

    return x / 255.

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = T.tensor(observation.copy(), dtype=T.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = transforms.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transform = transforms.Compose(
            [transforms.Resize(self.shape, antialias=True), transforms.Normalize(0, 255)]
        )
        observation = transform(observation).squeeze(0)
        return observation
