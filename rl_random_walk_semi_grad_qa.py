import numpy as np

def relu(inputs):
    for i, x in enumerate(inputs[0]):
        if x < 0:
            inputs[0][i] = input * 0.001
    
    return inputs

def get_grads(state, action, td_target):
    feature_vector = np.zeros((1, state_space_size))
    feature_vector[0][state] = 1
    ohe = np.zeros((1,2))
    ohe[0][action] = 1
    y_pred = get_qa_values(state)
    w1_grads = np.matmul(np.transpose(np.matmul((ohe * (td_target - y_pred)), np.transpose(weights[1]["W"]))), feature_vector)
    w2_grads = np.matmul(np.transpose(ohe * (td_target - y_pred)), relu(np.matmul(feature_vector, weights[0]["W"])))
    return np.array([
        np.transpose(w1_grads),
        np.transpose(w2_grads)
    ])

def get_qa_values(state):
    feature_vector = np.zeros((1, state_space_size))
    feature_vector[0][state] = 1
    return np.matmul(relu(np.matmul(feature_vector, weights[0]["W"])), weights[1]["W"])

def choose_action(state):
    qa_values = get_qa_values(state)
    
    if np.random.rand() < 0.9:
        return np.argmax(qa_values[0])
    else:
        return np.random.choice([0, 1])

if __name__ == "__main__":
    num_of_episodes = 100
    state_space_size = 25
    num_of_actions = 2
    
    weights = [
        {
            "W": np.ones((state_space_size, 20))
        },
        {
            "W": np.ones((20, num_of_actions))
        }
    ]
    
    for n in range(num_of_episodes):
        done = False
        state = int(state_space_size / 2)
        action = choose_action(state)
        step_size = 0.001
        gamma = 0.5
        num_steps = 0
        
        while not done:
            state_action_value = get_qa_values(state)
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
                grads = get_grads(state, action, reward)
                weights[0]["W"] += step_size * grads[0]
                weights[1]["W"] += step_size * grads[1]
                break
            
            new_action = choose_action(new_state)
            new_state_action_value = get_qa_values(new_state)
            td_target = reward + gamma * new_state_action_value
            grads = get_grads(state, action, td_target)
            weights[0]["W"] += step_size * grads[0]
            weights[1]["W"] += step_size * grads[1]
            
            state = new_state
            action = new_action
            num_steps += 1
        
        print(num_steps)