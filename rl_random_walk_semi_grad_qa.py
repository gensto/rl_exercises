import numpy as np

def relu(inputs):
    for i, x in enumerate(inputs[0]):
        if x < 0:
            inputs[0][i] = input * 0.001
    
    return inputs

def get_grads(state, action):
    feature_vector = np.zeros((1, state_space_size))
    feature_vector[0][state] = 1
    w1_grads = weights[1]["W"][:, action] * feature_vector
    w2_grads = np.transpose(relu(feature_vector))
    return np.array([
        w1_grads,
        w2_grads
    ])

def get_qa_values(state):
    feature_vector = np.zeros((1, state_space_size))
    feature_vector[0][state] = 1
    return np.matmul(relu(np.matmul(feature_vector, weights[0]["W"])), weights[1]["W"])

def choose_action(state):
    qa_values = get_qa_values(state)
    
    if np.random.randint() < 0.9:
        return np.argmax(qa_values)
    else:
        return np.random.choice([0, 1])
    

if __name__ == "__main__":
    num_of_episodes = 1000
    state_space_size = 10
    num_of_actions = 2
    
    weights = [
        {
            "W": np.ones(state_space_size, 20)
        },
        {
            "W": np.ones(20, num_of_actions)
        }
    ]
    
    for n in range(num_of_episodes):
        done = False
        state = int(state_space_size / 2)
        action = choose_action(state)
        state_action_value = get_qa_values(state)[action]
        step_size = 0.001
        gamma = 0.5
        
        while not done:
            # take action, observe reward
            if action == 0:
                new_state = state - 1
            else:
                new_state = state + action
            reward = 0
            if new_state == state_space_size - 1:
                reward = 10
                done = True
            elif new_state == 0:
                reward = -10
                done = True
            
            if done:
                td_error = reward + gamma * state_action_value
                grads = get_grads(state, action)
                weights[0]["W"] += step_size * td_error * grads[0]
                weights[1]["W"][:, action] += step_size * td_error * grads[1]
                break
            
            new_action = choose_action(new_state)
            new_state_action_value = get_qa_values(new_state)[new_action]
            td_error = reward + gamma * new_state_action_value - state_action_value
            grads = get_grads(state, action)
            weights[0]["W"] += step_size * td_error * grads[0]
            weights[1]["W"][:, action] += step_size * td_error * grads[1]
            
            state = new_state
            action = new_action
            state_action_value = new_state_action_value