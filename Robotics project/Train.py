import gym
import numpy as np
import Agents
from collections import deque
import sys


env = gym.make('CartPole-v1')

Agent = Agents.spiked_model(env.observation_space.shape[0], env.action_space.n, 10000, 16)
epsilon_initial = 1.0
epsilon_decay = 0.99
epsilon_minium = 0.1
epsilon = epsilon_initial
reward_window = deque(maxlen=(50))

for i in range(1, 1001):
    
    done = False
    score = 0
    state = env.reset()

    
    while not done:
        epsilon = max(epsilon * epsilon_decay, epsilon_minium)
        action = Agent.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        Agent.push_mem(state, action, reward, next_state, done)
        state = next_state
        score += reward
    
    Agent.is_update()
    reward_window.append(score)
    if i % 10 == 0:
        Agent.update_target()
        print('Episode:{}/{}|Mean score:{}| '.format(i, 1_000, np.mean(reward_window)))


