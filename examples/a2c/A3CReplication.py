# coding: utf-8

# In[1]:
import os
print(os.getcwd())
import sys
sys.path.append("..") # worked from /cule/examples/a2c
#sys.path.append("/cule/")

import math
import time
import torch

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

print(sys.version)

from utils.initializers import args_initialize, env_initialize, log_initialize, model_initialize
from helper import callback, format_time, gen_data
from model import ActorCritic

#sys.path.append(".")
#from test import test

try:
    from apex import amp
except ImportError:
    raise ImportError('Please install apex from https://www.github.com/nvidia/apex to run this example.')

from torchcule.atari import Env as AtariEnv
import subprocess


# In[13]:


seed = 1337
use_cuda_env = True

num_steps = 10 # n in n-step returns
num_ales = 100
total_steps = 1000000

env_name = 'PongNoFrameskip-v4'
episodic_life = False # check those defaults
clip_rewards = True

verbose = True
ale_start_steps = 100

normalize = True
num_stack = 4
lr = 1.0e-4

gamma = 0.99
entropy_coef = 0.0001
max_grad_norm = 10.0


# In[3]:


np.random.seed(seed)
torch.manual_seed(np.random.randint(1, 10000))
if use_cuda_env:
    torch.cuda.manual_seed(np.random.randint(1, 10000))

gpu = 0
train_device = env_device = device = torch.device('cuda', gpu)


# In[4]:


train_env = AtariEnv(env_name, num_ales, color_mode='gray', repeat_prob=0.0,
                             device = device, rescale=True, episodic_life = episodic_life,
                             clip_rewards= clip_rewards, frameskip=4)

train_env.train()
observation = train_env.reset(initial_steps=ale_start_steps, verbose=verbose).squeeze(-1)


# In[5]:


print(*train_env.observation_space.shape[-2:])
shape = (num_steps + 1, num_ales, num_stack, *train_env.observation_space.shape[-2:])
print(shape)

states = torch.zeros(shape, device=train_device, dtype=torch.float32)
states[0, :, -1] = observation.to(device=train_device, dtype=torch.float32)

shape = (num_steps + 1, num_ales)
values  = torch.zeros(shape, device=train_device, dtype=torch.float32)
returns = torch.zeros(shape, device=train_device, dtype=torch.float32)

shape = (num_steps, num_ales)
rewards = torch.zeros(shape, device=train_device, dtype=torch.float32)
masks = torch.zeros(shape, device=train_device, dtype=torch.float32)
actions = torch.zeros(shape, device=train_device, dtype=torch.long)

# These variables are used to compute average rewards for all processes.
episode_rewards = torch.zeros(num_ales, device=train_device, dtype=torch.float32)
final_rewards = torch.zeros(num_ales, device=train_device, dtype=torch.float32)
episode_lengths = torch.zeros(num_ales, device=train_device, dtype=torch.float32)
final_lengths = torch.zeros(num_ales, device=train_device, dtype=torch.float32)


import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


# In[7]:


get_gpu_memory_map()
# in PyTorch 1.4, looks like we have print(torch.cuda.memory_summary(device)) and torch.cuda.memory_stats() amongst other functions


# In[8]:


print(torch.cuda.get_device_properties(device).total_memory/(1024.0*1024.0))


# In[9]:


torch.cuda.synchronize()

model = ActorCritic(num_stack, train_env.action_space, normalize=normalize, name=env_name)
model = model.to(device).train()
optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad= False) # savage, but AMSGrad was enabled by default !


opt_level = 0
loss_scale = 0.5

from apex.amp import __version__
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier

if device.type == 'cuda':
    model, optimizer = amp.initialize(model, optimizer,
                                          opt_level= opt_level,
                                          loss_scale=loss_scale)



for i in tqdm(range(total_steps)):
    
    with torch.no_grad():
        for step in range(num_steps):
            value, logit = model(states[step])
            # store values
            values[step] = value.squeeze(-1)

            # convert actions to numpy and perform next step
            probs_action = F.softmax(logit, dim=1).multinomial(1).to(env_device)
            observation, reward, done, info = train_env.step(probs_action)
            observation = observation.squeeze(-1).unsqueeze(1)

            # move back to training memory
            observation = observation.to(device=train_device)
            reward = reward.to(device=train_device, dtype=torch.float32)
            done = done.to(device=train_device, dtype=torch.bool)
            probs_action = probs_action.to(device=train_device, dtype=torch.long)

            not_done = 1.0 - done.float()

            # update rewards and actions
            actions[step].copy_(probs_action.view(-1))
            masks[step].copy_(not_done)
            rewards[step].copy_(reward.sign())

            # update next observations
            states[step + 1, :, :-1].copy_(states[step, :, 1:].clone())
            states[step + 1] *= not_done.view(-1, *[1] * (observation.dim() - 1))
            states[step + 1, :, -1].copy_(observation.view(-1, *states.size()[-2:]))

            # update episodic reward counters
            episode_rewards += reward
            final_rewards[done] = episode_rewards[done]
            episode_rewards *= not_done

            episode_lengths += not_done
            final_lengths[done] = episode_lengths[done]
            episode_lengths *= not_done

            returns[-1] = values[-1] = model(states[-1])[0].data.squeeze(-1)

        for step in reversed(range(num_steps)):
            returns[step] = rewards[step] + (gamma * returns[step + 1] * masks[step])

    value, logit = model(states[:-1].view(-1, *states.size()[-3:]))

    log_probs = F.log_softmax(logit, dim=1)
    probs = F.softmax(logit, dim=1)

    action_log_probs = log_probs.gather(1, actions.view(-1).unsqueeze(-1))
    dist_entropy = -(log_probs * probs).sum(-1).mean()

    advantages = returns[:-1].view(-1).unsqueeze(-1) - value

    value_loss = advantages.pow(2).mean()
    policy_loss = -(advantages.clone().detach() * action_log_probs).mean()

    loss = 0.5 * value_loss + policy_loss - dist_entropy * entropy_coef
    optimizer.zero_grad()
    
    # assume no AMP-style loss scaling in the below two lines.
    #loss.backward()
    #master_params = model.parameters()
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
        master_params = amp.master_params(optimizer)
    
    torch.nn.utils.clip_grad_norm_(master_params, max_grad_norm)
    optimizer.step()

    states[0].copy_(states[-1])

    torch.cuda.synchronize()



