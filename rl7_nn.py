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
    for i, x_input in enumerate(inputs):
        if x_input <= 0:
            inputs[i] = 0
    return inputs

def get_state_action_values(state):
    state_action_values = np.zeros((2))
    for i in range(2):
        state_action_vector = np.zeros(8)
        state_action_vector[i*4:(i*4)+4] = state
        relu_inputs = np.dot(np.transpose(w1), state_action_vector)
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
    print(f"State action vector: {state_action_vector}")
    print(f"W2: {w2}")
    print(f"Mult: {(w2 * state_action_vector).shape}")
    return 2 * (td_target - state_action_value) * w2 * state_action_vector

def get_gradients_w2(td_target, state, action, state_action_value):
    hidden_layer_inputs = np.zeros((10))
    state_action_vector = np.zeros(8)
    state_action_vector[action*4:(action*4)+4] = state
    hidden_layer_inputs = np.dot(np.transpose(w1), state_action_vector)
    # for i, weights in enumerate(w1):
    #     hidden_layer_inputs += state_action_vector[i] * weights
    for i, hidden_layer_input in enumerate(hidden_layer_inputs):
        if hidden_layer_input < 0:
            hidden_layer_inputs[i] = 0

    print(f"State vector: {state_action_vector}")
    print(f"hidden layer inputs: {hidden_layer_inputs}")
    return 2 * (td_target - state_action_value) * hidden_layer_inputs

def e_greedy_policy(state):
    if np.random.rand() < 0.9:
        return np.argmax(get_state_action_values(state))
    else:
        return np.random.randint(2)

if __name__ == "__main__":
    w1 = np.ones((8, 10))
    w2 = np.ones((10, 1))
    
    gamma = 0.5
    step_size = 0.1
    num_of_episodes = 1000
    rewards_per_episode = []
    
    for n in range(num_of_episodes):
        last_state = env.reset(seed=42)[0]
        last_action = e_greedy_policy(last_state)
        done = False
        total_rewards = 0
        
        while not done:
            last_state_action_value = get_state_action_values(last_state)[last_action]
            # print(last_state_action_value)
            new_state, reward, done, truncated, info = env.step(last_action)
            if done:
                td_target = -20 + gamma * new_state_action_value
                w1_gradients = get_gradients_w1(td_target, last_state, last_action, last_state_action_value)
                w2_gradients = get_gradients_w2(td_target, last_state, last_action, last_state_action_value)    
            else:
                new_action = e_greedy_policy(new_state)
                new_state_action_value = get_state_action_values(new_state)[new_action]
                td_target = reward + gamma * new_state_action_value
                w1_gradients = get_gradients_w1(td_target, last_state, last_action, last_state_action_value)
                w2_gradients = get_gradients_w2(td_target, last_state, last_action, last_state_action_value)
                for weights in w1:
                    weights += step_size * w1_gradients
                for weights in w2:
                    weights += step_size * w2_gradients
                    
                last_state = new_state
                last_action = new_action
            total_rewards += reward
        rewards_per_episode.append(total_rewards)
    
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Rewards per Episode")
    # xtick_positions = np.arange(0, 5000 / 50)
    # xtick_labels = xtick_positions * 50
    # plt.xticks(xtick_positions, xtick_labels)
    plt.show()           
    env.close()