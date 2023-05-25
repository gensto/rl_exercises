import numpy as np
import gym
env = gym.make('CliffWalking-v0')

if __name__ == "__main__":
    weights = [
        {
            "W": np.ones((48, 30))
        },
        {
            "W": np.ones((30, 4))
        }
    ]
    num_of_episodes = 100
    done = False
    R = 0.1
    states_action_rewards = []
    
    for ep in num_of_episodes(100):
        state = env.reset(seed=42)[0]
        while not done:
            