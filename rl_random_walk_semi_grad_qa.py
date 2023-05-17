import numpy as np

def relu(inputs):
    for i, x in enumerate(inputs[0]):
        if x < 0:
            inputs[0][i] = input * 0.001
    
    return inputs

def get_qa_values(state):
    feature_vector = np.zeros((1, state_space_size))
    feature_vector[0][state] = 1
    return np.matmul(relu(np.matmul(feature_vector, weights[0]["W"])), weights[1]["W"])

def choose_action(state):
    qa_values = get_qa_values(state)
    

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
        state = int(state_space_size)
        action = 
        
        while not done:
            