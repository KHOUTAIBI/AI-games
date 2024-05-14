import torch as T
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optimum

import gymnasium as gym
#import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import sys

class DeepQNetwork(nn.Module):
    def __init__(self, input_dims,n_actions):
        super(DeepQNetwork,self).__init__()

        self.input_dims = input_dims
        self.conv1 = nn.Conv2d(4,32,3,stride=1)
        self.conv2 = nn.Conv2d(32,32,3,stride=1)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, n_actions)
        self.tanh = nn.Tanh() # activation function on input (transforms into range of -1 to 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        transform1 = transforms.CenterCrop(40)
        transform2 = transforms.Grayscale()

        # Convert the image to grayscale
        x = x.permute(0, 3, 1, 2)
        x = transform2(x)
        x = transform1(x)


        x = self.conv1(x)
        x = nn.functional.max_pool2d(x,kernel_size=2)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x,kernel_size=2).flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Memory():
    def __init__(self,input_dims,max_mem_size = 100000):
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
    def __init__(self, gamma, lr, batch_size, n_actions, input_dims, epsilon, eps_end=0.01, eps_dec = 5e-4, max_mem_size = 100000):
        self.device = T.device('cuda:1' if T.cuda.is_available() else 'cpu')
        self.gamma= gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space  = [i for i in range(n_actions)]
        self.batch_size = batch_size

        self.memory = Memory(input_dims,max_mem_size)

        self.Q_eval = DeepQNetwork(input_dims, n_actions).to(self.device)
        self.optimizer = optimum.Adam(self.Q_eval.parameters(),lr=self.lr)
        self.loss_fn = nn.MSELoss()
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = np.array([observation],dtype=np.float32)
            state = T.tensor(state).to(self.device)
            actions = self.Q_eval(state)
            return T.argmax(actions).item()
        else:
            return np.random.choice(self.action_space)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
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
        q_next = self.Q_eval(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_target, q_eval).to(self.device)
        loss.backward()
        self.optimizer.step()

        #self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)
