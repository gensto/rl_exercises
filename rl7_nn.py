from tile_coding import IHT, tiles
import time 
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# env = gym.make('Acrobot-v1')
np.random.seed(123)
env = gym.make("CartPole-v0")
env.action_space.seed(42)
print(env.action_space.n)

def relu(x):
    if x <= 0:
        return 0
    else:
        return x

def get_state_action_values(state):
    state_action_values = np.array([])
    for i in range(2):
        state_action_vector[i*4:(i*4)+4] = state
        state_value = 0
        for i, weights in enumerate(w1):
            state_action_vector = np.zeros(8)
            state_value += relu((np.dot(state_action_vector, weights))) * w2[i]
        state_action_values.append(state_value)

def e_greedy_policy(state):
    if np.random.rand() < 0.9:
        return np.argmax(get_state_action_values(state))
    else:
        return np.random.randint(2)

if __name__ == "__main__":
    w1 = np.ones((10, 8))
    w2 = np.ones((10))
    
    num_of_episodes = 100
    
    for n in range(num_of_episodes):
        last_state = env.reset(seed=42)[0]
        last_action = e_greedy_policy(last_state)
        done = False
        
        while not done:
            new_state, reward, done, truncated, info = env.step(last_action)
            new_action = e_greedy_policy(new_state)