import os
import sys

_path = os.path.abspath(os.path.pardir)
if not _path in sys.path:
    sys.path = [_path] + sys.path

import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from torchcule.atari import Env as AtariEnv

from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

from allutils import make_atari, wrap_deepmind, wrap_pytorch

USE_OPENAI = False
env_id = "Pong" + "NoFrameskip-v4"

if USE_OPENAI:
    env    = make_atari(env_id)
    env    = wrap_deepmind(env)
    env    = wrap_pytorch(env)
else:
    num_ales = 256
    torch.cuda.set_device(0)
    env_device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    reward_clip = True
    env = AtariEnv(env_id, num_ales, color_mode = 'gray',
                         device = env_device, rescale = True,
                         clip_rewards = reward_clip,
                         episodic_life = True, repeat_prob = 0.0)
    env.train()
    observation = env.reset(initial_steps = 100, verbose = True).clone().squeeze(-1)
    print(observation.dtype)
    try:
      print(observation.shape)
    except:
      pass

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action

model = CnnDQN(env.observation_space.shape, env.action_space.n)

if USE_CUDA:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.00001)

def plot(frame_idx, rewards, losses):
    print('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

num_frames = 1400000
batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()
for frame_idx in tqdm(range(1, num_frames + 1)):
    epsilon = epsilon_by_frame(frame_idx)
    if USE_OPENAI:
       action = model.act(state, epsilon)
       next_state, reward, done, _ = env.step(action)
       replay_buffer.push(state, action, reward, next_state, done)
       state = next_state
    else:
       observation, reward, done, info = env.step(torch.rand(256, device=env_device, dtype=torch.float32))

    episode_reward += reward

    if done == True and USE_OPENAI == True:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if frame_idx % 10000 == 0 and USE_OPENAI:
        plot(frame_idx, all_rewards, losses)
