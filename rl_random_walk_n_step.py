import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt

def relu(inputs):
    # inputs[0][inputs[0] < 0] = 0
    for i, input in enumerate(inputs[0]):
        if input < 0:
            inputs[0][i] = input * 0.01
    
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
    testing = relu(np.matmul(np.transpose(feature_vector), weights[0]["W"]))
    testing1 = np.matmul(np.transpose(feature_vector), weights[0]["W"])
    w2_grads = np.transpose(relu(np.matmul(np.transpose(feature_vector), weights[0]["W"])))
    
    return [w1_grads, w2_grads]

if __name__ == "__main__":
    state_space_size = 10
    num_of_episodes = 7000
    gamma = 0.99
    step_size = 0.001
    
    weights = [
        {
            "W": np.ones((state_space_size, 10))
        },
        {
            "W": np.ones((10,1))        
        }
    ]
    
    state_values = []
    
    for ep in range(num_of_episodes):
        T = float('inf')
        t = 0
        n = 8
        tau = 0
        rewards_and_next_states = []
        current_state = int(state_space_size / 2)
        rewards_and_next_states.append((0, 0))
        
        while True:
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
                rewards_and_next_states.append((reward, next_state))
                
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n + 1, T + 1)):
                    G += np.power(gamma, i - tau - 1) * rewards_and_next_states[i][0]
                if tau + n < T:
                    state_tau_plus_n_value = forward_prop(rewards_and_next_states[tau + n][1])[0,0]
                    G += np.power(gamma, n) * state_tau_plus_n_value
                
                state_tau_value = forward_prop(rewards_and_next_states[tau][1])[0,0]
                grads = get_grads(rewards_and_next_states[tau][1], G, next_state_value)
                weights[0]["W"] += step_size * (G - state_tau_value) * grads[0]
                weights[1]["W"] += step_size * (G - state_tau_value) * grads[1]

            if tau == T - 1:
                break
            current_state = next_state
            t += 1
        state_values.append(forward_prop(8)[0,0])
    get_state_values()
    
    plt.plot(state_values)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Rewards per Episode")
    # xtick_positions = np.arange(0, 5000 / 50)
    # xtick_labels = xtick_positions * 50
    # plt.xticks(xtick_positions, xtick_labels)
    plt.show()          