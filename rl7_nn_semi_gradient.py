import numpy as np

def relu(inputs):
    

def forward_prop(state):
    feature_vector = np.zeros((500, 1))
    feature_vector[state][0] = 1
    
    return relu(np.transpose(feature_vector) * weights[0]["W"]) * weights[1]["W"]

if __name__ == "__main__":
    weights = [
        {
            "W": np.zeros((500,400))
        },
        {
            "W": np.zeros((400, 1))
        }
    ]


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