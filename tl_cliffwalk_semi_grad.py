import numpy as np
import gym
env = gym.make('CliffWalking-v0')

def relu(inputs):
    for i, input_value in enumerate(inputs[0]):
        if input_value < 0:
            inputs[0][i] = input_value * 0.001
    
    return inputs

def get_state_action_values(state):
    feature_vector = np.zeros((1, 48))
    feature_vector[0][state] = 1

    return np.matmul(relu(np.matmul(feature_vector, weights[0]["W"])), weights[1]["W"])

def get_grads(state, td_target, action):
    feature_vector = np.zeros((1, 48))
    feature_vector[0][state] = 1
    ohe = np.zeros((1, 4)) 
    ohe[0][action] = 1
    y_pred = get_state_action_values(state)
    return [
        np.transpose(np.matmul(np.transpose(np.matmul(ohe * (td_target - y_pred), np.transpose(weights[1]["W"]))), feature_vector)),
        np.transpose(np.matmul(np.transpose(ohe * (td_target - y_pred)), relu(np.matmul(feature_vector, weights[0]["W"]))))
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
    step_size = 0.001
    gamma = 0.5

    for ep in range(num_of_episodes):
        state = env.reset(seed=42)[0]
        action = choose_action(state)
        done = False
        total_steps = 0

        while not done:
            state_action_value = get_state_action_values(state)[0][action]
            next_state, reward, done, truncated, info = env.step(action)
            if done:
                grads = get_grads(state, reward, action)
                weights[0]["W"] += step_size * grads[0]
                weights[1]["W"] += step_size * grads[1]
                break
            next_action = choose_action(next_state)
            td_target = reward + gamma * get_state_action_values(next_state)
            grads = get_grads(state, td_target, next_action)
            weights[0]["W"] += step_size * grads[0]
            weights[1]["W"] += step_size * grads[1]
            state = next_state
            action = next_action
            
            total_steps += 1
        print(f"total steps: {total_steps}")
        print("done")
            

