import numpy as np

calculate_g()

if __name__ == "__main__":
    state_space_size = 10
    state_and_rewards_received = []
    
    num_of_episodes = 50
    
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
            if next_state == state_space_size - 1:
                T = t + 1
                reward = 10
            elif next_state == 0:
                T = t + 1
                reward = -10
            tau = t - n + 1
        if tau >= 0:
            G = 
    