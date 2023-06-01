import numpy as np
import gym
env = gym.make('MountainCar-v0', render_mode='human')

def relu(inputs):
    for i, x in enumerate(inputs[0]):
        if x < 0:
            inputs[0][i] = 0
    
    return inputs

def forward_prop(state):
    state_reshaped = np.transpose(state.reshape(2,1))
    return np.matmul(relu(np.matmul(state_reshaped, weights[0]["W"])), weights[1]["W"])

def choose_action(state):
    state_qa_values = forward_prop(state)
    
    if np.random.rand() < 0.9:
        return np.argmax(state_qa_values[0])
    else:
        return np.random.choice(np.arange(3))

def get_grads(state, action):
    state_reshaped = np.transpose(state.reshape(2,1))
    return [
        np.transpose(np.matmul(weights[1]["W"][:, action].reshape(30, 1), state_reshaped)),
        np.squeeze(np.transpose(relu(np.matmul(state_reshaped, weights[0]["W"]))))
    ]

if __name__ == "__main__":
    weights = [
        {
            "W": np.ones((2, 30))
        },
        {
            "W": np.ones((30, 3))
        }
    ]
    states = []
    actions = []
    rewards = []
    gamma = 0.5
    n = 8
    step_size = 0.3 / 8
    num_of_episodes = 1000
    
    for ep in range(num_of_episodes):
        state = env.reset(seed=42)[0]
        action = choose_action(state)
        states = [state]
        actions = [action]
        T = float('inf')
        t = 0
        total_steps = 0
        while True:
            if t < T:
                next_state, reward, done, truncated, info = env.step(action)
                states.append((next_state))
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
        print("done")
        print(total_steps)
            