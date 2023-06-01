import numpy as np
import gym
env = gym.make('CartPole-v1', render_mode='human')

def relu(inputs):
    for i, x in enumerate(inputs[0]):
        if x < 0:
            inputs[0][i] = x
    
    return inputs

def forward_prop(state):
    state_reshaped = np.transpose(state.reshape(4,1))
    return np.matmul(relu(np.matmul(state_reshaped, weights[0]["W"])), weights[1]["W"])

def choose_action(state):
    state_qa_values = forward_prop(state)
    
    if np.random.rand() < 0.9:
        print(state_qa_values)
        return np.argmax(state_qa_values[0])
    else:
        return np.random.choice(np.arange(2))
    
def normalize_state(state):
    state[0] = (state[0] - (-2.4)) / (2.4 - (-2.4))
    state[1] = (state[1] - (-4)) / (4 - (-4))
    state[2] = (state[2] - (-0.2095)) / (0.2095 - (-0.2095))
    state[3] = (state[3] - (-5)) / (5 - (-5))
    
    return state

def get_grads(state, action):
    state_reshaped = np.transpose(state.reshape(4,1))
    return [
        np.transpose(np.matmul(weights[1]["W"][:, action].reshape(30, 1), state_reshaped)),
        np.squeeze(np.transpose(relu(np.matmul(state_reshaped, weights[0]["W"]))))
    ]

if __name__ == "__main__":
    weights = [
        {
            "W": np.ones((4, 30))
        },
        {
            "W": np.ones((30, 2))
        }
    ]
    states = []
    actions = []
    rewards = []
    gamma = 0.5
    n = 8
    step_size = 0.001
    num_of_episodes = 1000
    
    for ep in range(num_of_episodes):
        state = normalize_state(env.reset(seed=42)[0])
        action = choose_action(state)
        states = [state]
        actions = [action]
        T = float('inf')
        t = 0
        total_steps = 0
        while True:
            if t < T:
                next_state, reward, done, truncated, info = env.step(action)
                states.append((normalize_state(next_state)))
                rewards.append((reward))
                if done:
                    T = t + 1
                else:
                    next_action = choose_action(next_state)
                    actions.append(next_action)
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T)):
                    G += np.power(gamma, i - tau - 1) * rewards[i]
                if tau + n < T:
                    tau_plus_n_qa = forward_prop(states[tau + n])[0][actions[tau + n]]
                    G += np.power(gamma, n) * tau_plus_n_qa
                grads = get_grads(states[tau], action)
                tau_qa = forward_prop(states[tau])[0][actions[tau]]
                weights[0]["W"] += step_size * (G - tau_qa) * grads[0]
                weights[1]["W"][:, action] += step_size * (G - tau_qa) * grads[1]
            if tau == T - 1:
                break
            state = next_state
            action = next_action
            total_steps += 1
            t += 1
        # print("done")
        print(total_steps)
            