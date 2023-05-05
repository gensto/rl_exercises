import numpy as np

def relu(inputs):
    inputs[0][inputs[0] < 0] = 0
    
    return inputs

def forward_prop(state):
    feature_vector = np.zeros((50, 1))
    feature_vector[state][0] = 1

    return np.dot(relu(np.dot(np.transpose(feature_vector), weights[0]["W"])), weights[1]["W"])

def get_grads(state):
    feature_vector = np.zeros((50, 1))
    feature_vector[state][0] = 1
    state_value = forward_prop(state)[0, 0]
    
    return np.array([
        np.dot(feature_vector, np.transpose(weights[1]["W"])),
        np.dot(relu(np.transpose(weights[0]["W"])), feature_vector)
    ])

def get_state_values():
    for i in range(50):
        print(f"State {i} value: {forward_prop(i)[0,0]}")
    

if __name__ == "__main__":
    weights = [
        {
            "W": np.ones((50,40))
        },
        {
            "W": np.ones((40, 1))
        }
    ]
    
    step_size = 0.01
    gamma = 0.7
    num_of_episodes = 1000
    
    for n in range(num_of_episodes):
        reward = 0
        done = False
        current_state = 25
        
        while not done:
            action = np.random.choice([-1, 1])
            new_state = current_state + action
            reward = 0
            if new_state == 0:
                done = True
                reward = -10
            elif new_state == 49:
                done = True
                reward = 10
            td_error = reward + gamma * forward_prop(new_state)[0, 0] - forward_prop(current_state)[0, 0]
            grads = get_grads(current_state)
            weights[0]["W"] += step_size * td_error * grads[0]
            weights[1]["W"] += step_size * td_error * grads[1]
            
            current_state = new_state
    
    get_state_values()           

# def state_info(state):
#     num_of_rows = 4
#     num_of_columns = 12
    
#     state_row_index = state // num_of_columns
#     state_column_index = state % num_of_columns
    
#     is_border_state = False
#     is_in_first_row = False
#     is_in_last_row = False
#     is_in_corner = False
#     is_in_first_column = False
#     is_in_last_column = False
    
    
#     if state_row_index == 0 or state_row_index == num_of_rows - 1 or state_column_index == 0 or state_column_index == num_of_columns - 1:
#         is_border_state = True
#     if state_row_index == 0:
#         is_in_first_row = True
#     if state_row_index == num_of_rows - 1:
#         is_in_last_row = True
#     if (state_row_index == 0 and state_column_index == 0) or (state_row_index == 0 and state_column_index == num_of_columns - 1) \
#         (state_row_index == num_of_rows - 1 and state_column_index == 0) or (state_row_index == num_of_rows - 1 and state_column_index == num_of_columns - 1):
#         is_in_corner = True
#     if state_column_index == 0:
#         is_in_first_column = True
#     if state_column_index == num_of_columns - 1:
#         is_in_last_column = True
    
#     return {
#         state: state,
#         is_border_state: is_border_state,
#         is_in_first_row: is_in_first_row,
#         is_in_last_row: is_in_last_row,
#         is_in_first_column: is_in_first_column,
#         is_in_last_column: is_in_last_column
#     }

# def get_surrounding_states(state_info):
#     if state_info.is_border:
#         if state_info.is_in_first_row:
#             if state_info.state == 0 or state_info.state == 11:
#                 return {
#                     "left": state_info.state - 1,
#                     "right": state_info.state + 1,
#                     "up": state_info.state - 12,
#                     "down": state_info.state + 12
#                 }   
#         elif state_info.is_in_last_row:
#             if state_info.state == 36 or state_info.state == 47:
#                 return {
#                     "left": state_info.state - 1,
#                     "right": state_info.state + 1,
#                     "up": state_info.state - 12,
#                     "down": state_info.state + 12
#                 }   
#         elif state_info.is_in_first_column:
#             return {
#                 "left": state_info.state,
#                 "right": state_info.state + 1,
#                 "up": state_info.state - 12,
#                 "down": state_info.state + 12
#             }
#         else:
#             return {
#                 "left": state_info.state - 1,
#                 "right": state_info.state,
#                 "up": state_info.state - 12,
#                 "down": state_info.state + 12
#             }
#     else:
#         return {
#             "left": state_info.state - 1,
#             "right": state_info.state + 1,
#             "up": state_info.state - 12,
#             "down": state_info.state + 12
#         }

# def get_values(surrounding_states):
    

# def select_action(state):
#     values_of_states = get_values(get_surrounding_states(state_info(state)))
#     if is_border_state(state):