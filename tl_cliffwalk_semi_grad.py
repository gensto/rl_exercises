import numpy as np
import gym
env = gym.make('CliffWalking-v0')

def relu(inputs):
    for i, input_value in enumerate(inputs):
        if input_value < 0:
            inputs[i] = input_value * 0.001
    
    return inputs

def get_state_action_values(state):
    feature_vector = np.zeros((1, 48))
    feature_vector[0][state] = 1

    return np.matmul(np.transpose(relu(np.transpose(np.matmul(feature_vector, weights[0]["W"]))), weights[1]["W"]))

def get_grads(state, action):
    feature_vector = np.zeros((4, 48))
    feature_vector[action][state] = 1

    return [
        np.transpose(np.matmul(weights[1]["W"], np.transpose(feature_vector))),
        np.transpose(np.matmul(feature_vector.reshape(1, 48), weights))
    ]

def choose_action(state):
    state_action_values = get_state_action_values(state)[0]

    if np.random.rand() < 0.9:
        return np.argmax(state_action_values)
    else:
        return np.random.choice(np.arange(4))


if __name__ == "__main__":
    weights = [
        {
            "W": np.zeros((48, 30))
        },
        {
            "W": np.zeros((30, 4))
        }
    ]
    num_of_episodes = 1000

    for ep in range(num_of_episodes):
        state = env.reset(seed=42)[0]
        action = choose_action(state)
        done = False

        while not done:
            state_action_value = get_state_action_values(state)[0][action]
            new_state, reward, done, truncated, info = env.step(action)
            if done:
                grads

