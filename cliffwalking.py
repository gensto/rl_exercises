from tile_coding import IHT, tiles
import time 
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

env = gym.make('CliffWalking-v0')

if __name__ == "__main__":
    last_state = env.reset(seed=42)[0]
    new_state = env.step(0)
