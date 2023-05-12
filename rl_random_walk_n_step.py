import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt

def relu(inputs):
    inputs[0][inputs[0] < 0] = 0
    
    return inputs            

def forward_prop(state):
    feature_vector = np.zeros((state_space_size, 1))
    feature_vector[state, 0] = 1
    
    return np.matmul(relu(np.matmul(np.transpose(feature_vector), weights[0]["W"])), weights[1]["W"]) 

def get_state_values():
    for i in range(state_space_size - 1):
        if i > 0:
            print(f"State {i} value: {forward_prop(i)[0,0]}")

def get_grads(state, td_target, pred):
    feature_vector = np.zeros((state_space_size, 1))
    feature_vector[state, 0] = 1
    w1_grads = np.transpose(np.matmul(weights[1]["W"], np.transpose(feature_vector)))
    w2_grads = np.transpose(relu(np.matmul(np.transpose(feature_vector), weights[0]["W"])))
    
    return np.array([w1_grads, w2_grads])

if __name__ == "__main__":
    state_space_size = 10
    num_of_episodes = 4000
    gamma = 0.7
    step_size = 0.001
    
    weights = [
        {
            "W": np.ones((state_space_size, 30))
        },
        {
            "W": np.ones((30,1))        
        }
    ]
    
    state_values = []
    
    for n in range(num_of_episodes):
        T = float('inf')
        t = 0
        n = 10
        tau = 0
        rewards_and_next_states = []
        current_state = int(state_space_size / 2)
        
        while tau < T - 1:
            if t < T:
                action = np.random.choice([-1, 1])
                next_state = current_state + action
                next_state_value = forward_prop(next_state)[0,0]
                reward = 0
        
                if next_state == state_space_size - 1:
                    T = t + 1
                    reward = 10
                    next_state_value = 0
                elif next_state == 0:
                    T = t + 1
                    reward = -10
                    next_state_value = 0
                rewards_and_next_states.append((current_state, reward, next_state))
                
            tau = t - n + 1
            if tau >= 0:
                G = 0
                i  = tau + 1
                while i < min(tau + n, T):
                    G += pow(gamma, i - tau - 1) * rewards_and_next_states[i][1]
                    i += 1
                if tau + n < T:
                    state_tau_plus_n_value = forward_prop(rewards_and_next_states[tau + n - 1][0])[0,0]
                    G += pow(gamma, n) * state_tau_plus_n_value
                    
                grads = get_grads(current_state, G, next_state_value)
                state_tau_value = forward_prop(rewards_and_next_states[tau][0])[0,0]
                weights[0]["W"] += step_size * (G - state_tau_value) * grads[0]
                weights[1]["W"] += step_size * (G - state_tau_value) * grads[1]
            current_state = next_state
            t += 1
        state_values.append(forward_prop(6)[0,0])
    get_state_values()
    
    # plt.plot(state_values)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("Rewards per Episode")
    # # xtick_positions = np.arange(0, 5000 / 50)
    # # xtick_labels = xtick_positions * 50
    # # plt.xticks(xtick_positions, xtick_labels)
    # plt.show()          