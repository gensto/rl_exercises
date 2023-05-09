import numpy as np

calculate_g()

if __name__ == "__main__":
    state_space_size = 10
    state_and_rewards_received = []
    
    num_of_episodes = 50
    gamma = 0.5
    
    weights = [
        {
            "W": np.ones((state_space_size, 40))
        },
        {
            "W": np.ones((40,1))        
        }
    ]
    
    for n in range(num_of_episodes):
        T = float('inf')
        t = 0
        n = 3
        current_state = int(state_space_size / 2)
        
        while t < T:
            action = np.random.choice([-1, 1])
            next_state = current_state + action
            reward = 0
            if next_state == state_space_size - 1:
                T = t + 1
                reward = 10
            elif next_state == 0:
                T = t + 1
                reward = -10
            state_and_rewards_received.append((reward, next_state))
            tau = t - n + 1
            if tau >= 0:
                G = 0
                i  = tau + 1
                while i < range(min(tau + n, T)):
                    G += pow(gamma, i - tau - 1) * 