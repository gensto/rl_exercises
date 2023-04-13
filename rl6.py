from tile_coding import IHT, tiles
import time 
import gym
import numpy as np
import matplotlib.pyplot as plt

gym.make('MountainCar-v0')

if __name__ == "__main__":
    position_bins = 0
    velocity_bins = 0
    q_table = np.ones((30, 30, 3))