from tile_coding import IHT, tiles
import time 
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# env = gym.make('Acrobot-v1')
np.random.seed(123)
env = gym.make("CartPole-v1")
# env = gym.make("CartPole-v1", render_mode='human')
env.action_space.seed(42)
print(env.action_space.n)

# def relu(x):
#     if x <= 0:
#         return 0
#     else:
#         return x

def relu(inputs):
    return np.maximum(inputs, inputs)

def get_state_action_values(state):
    state_action_values = np.zeros((2))
    for i in range(2):
        state_action_vector = np.zeros((1, 8))
        state_action_vector[0][i*4:(i*4)+4] = state
        # relu_inputs = np.dot(np.transpose(w1), state_action_vector)
        relu_inputs = np.dot(state_action_vector, w1)
        relu_outputs = relu(relu_inputs)
        state_action_values[i] = np.dot(relu_outputs, w2)
    # state_action_values = np.zeros((2))
    # for i in range(2):
    #     final_value = 0
    #     relu_inputs = np.zeros((10))
    #     state_action_vector = np.zeros(8)
    #     state_action_vector[i*4:(i*4)+4] = state
    #     for j, weights in enumerate(w1):
    #         relu_inputs += state_action_vector[j] * weights
    #     for j, relu_input in enumerate(relu_inputs):
    #         final_value += relu(relu_input) * w2[j]
    #     state_action_values[i] = final_value
    return state_action_values

def get_gradients_w1(td_target, state, action, state_action_value):
    hidden_layer_inputs = np.zeros((10))
    state_action_vector = np.zeros((1, 8))
    state_action_vector[0][action*4:(action*4)+4] = state
    # hidden_layer_inputs = np.dot(np.transpose(w1), state_action_vector)
    # # for i, weights in enumerate(w1):
    # #     hidden_layer_inputs += state_action_vector[i] * weights
    # for i, hidden_layer_input in enumerate(hidden_layer_inputs):
    #     # if hidden_layer_input > 0:
    #     #     hidden_layer_inputs[i] = 1.0
    #     # else:
    #     #     hidden_layer_inputs[i] = 0.0
    #     if hidden_layer_input <= 0:
    #         hidden_layer_inputs[i] = 0
    # print(f"State action vector: {state_action_vector}")
    # print(f"W2: {w2}")
    # print(f"Mult: {(w2 * state_action_vector).shape}")
    # print(f"W1 grad: {2 * (td_target - state_action_value) * np.transpose(state_action_vector) * np.transpose(w2)}")
    return (td_target - state_action_value) * np.transpose(state_action_vector) * np.transpose(w2)

def get_gradients_w2(td_target, state, action, state_action_value):
    hidden_layer_inputs = np.zeros((10))
    state_action_vector = np.zeros((1, 8))
    state_action_vector[0][action*4:(action*4)+4] = state
    relu_inputs = np.dot(state_action_vector, w1)
    relu_outputs = np.transpose(relu(relu_inputs))

    # hidden_layer_inputs = np.dot(np.transpose(w1), state_action_vector)
    # # for i, weights in enumerate(w1):
    # #     hidden_layer_inputs += state_action_vector[i] * weights
    # for i, hidden_layer_input in enumerate(hidden_layer_inputs):
    #     if hidden_layer_input < 0:
    #         hidden_layer_inputs[i] = 0

    # print(f"State vector: {state_action_vector}")
    # print(f"hidden layer inputs: {hidden_layer_inputs}")
    # print(2 * (td_target - state_action_value) * relu_outputs)
    return (td_target - state_action_value) * relu_outputs

def e_greedy_policy(state):
    if np.random.rand() < 0.9:
        return np.argmax(get_state_action_values(state))
    else:
        return np.random.randint(2)
    
def get_max_state_action_value(state):
    return np.max(get_state_action_values(state))

def normalize_state_vector(state):
    for i, value in enumerate(state):
        old_max = ranges[i][0]
        old_min = ranges[i][1]
        new_max = 1
        new_min = 0
        old_value = value
        old_range = (old_max - old_min)  
        new_range = (new_max - new_min)  
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min
        state[i] = new_value
    
    return state

def clip_by_norm(grad):
    clip_norm = 1
    grad_norm = np.linalg.norm(grad)
    if grad_norm > clip_norm:
        grad = grad * (clip_norm / grad_norm)
    return grad


if __name__ == "__main__":
    # w1 = np.ones((8, 10))
    # w2 = np.ones((10, 1))
    w1 = np.random.randn(8, 10) * 0.01
    w2 = np.random.randn(10, 1) * 0.01

    ranges = [[-4.8, 4,8], [-4, 4], [-0.418, 0.418], [-5, 5]]
    gamma = 0.5
    step_size = 0.1
    num_of_episodes = 1000
    rewards_per_episode = []
    total_rewards = 0
    
    for n in range(num_of_episodes):
        last_state = normalize_state_vector(env.reset(seed=42)[0])
        last_action = e_greedy_policy(last_state)
        done = False
        
        while not done:
            last_state_action_value = get_state_action_values(last_state)[last_action]
            new_state, reward, done, truncated, info = env.step(last_action)
            new_state = normalize_state_vector(new_state)
            if done:
                max_state_action_value = get_max_state_action_value(new_state)
                td_target = -100
                w1_gradients = clip_by_norm(get_gradients_w1(td_target, last_state, last_action, last_state_action_value))
                w2_gradients = clip_by_norm(get_gradients_w2(td_target, last_state, last_action, last_state_action_value))
                w1 -= step_size * w1_gradients
                w2 -= step_size * w2_gradients
            else:
                new_action = e_greedy_policy(new_state)
                # new_state_action_value = get_state_action_values(new_state)[new_action]
                max_state_action_value = get_max_state_action_value(new_state)
                td_target = reward + gamma * max_state_action_value
                w1_gradients = clip_by_norm(get_gradients_w1(td_target, last_state, last_action, last_state_action_value))
                w2_gradients = clip_by_norm(get_gradients_w2(td_target, last_state, last_action, last_state_action_value))
                # print(w1_gradients)
                w1 -= step_size * w1_gradients
                w2 -= step_size * w2_gradients
                    
                last_state = new_state
                last_action = new_action
            total_rewards += reward
        if n % 50 == 0:
            print(total_rewards)
            rewards_per_episode.append(total_rewards / 50)
            total_rewards = 0
    
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Rewards per Episode")
    # xtick_positions = np.arange(0, 5000 / 50)
    # xtick_labels = xtick_positions * 50
    # plt.xticks(xtick_positions, xtick_labels)
    plt.show()           
    env.close()