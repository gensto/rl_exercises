import numpy as np

def relu(inputs):
    inputs[0][inputs[0] < 0] = 0
    
    return inputs            

def forward_prop(state):
    feature_vector = np.zeros((state_space_size, 1))
    feature_vector[state, 0] = 1
    
    return np.matmul(relu(np.matmul(np.transpose(feature_vector), weights[0]["W"])), weights[1]["W"]) 

def get_grads(state, td_target, pred):
    feature_vector = np.zeros((state_space_size, 1))
    feature_vector[state, 0] = 1
    w1_grads = np.matmul(weights[1]["W"], feature_vector)
    w2_grads = relu(np.matmul(np.transpose(feature_vector), weights[0]["W"]))

if __name__ == "__main__":
    state_space_size = 10
    
    num_of_episodes = 50
    gamma = 0.5
    step_size = 0.01
    
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
        rewards_and_next_states = []
        
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
            rewards_and_next_states.append((reward, next_state))
            tau = t - n + 1
            if tau >= 0:
                G = 0
                i  = tau + 1
                while i < range(min(tau + n, T)):
                    G += pow(gamma, i - tau - 1) * rewards_and_next_states[i[0]]
                if tau + n < T:
                    G += pow(gamma, n) * state_tau_plus_n_value
                weights[0]["W"] += step_size * (G - state_tau_value)
                        