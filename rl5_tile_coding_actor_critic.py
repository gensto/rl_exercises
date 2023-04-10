from tile_coding import IHT, tiles
import time 
import gym
import numpy as np


env = gym.make('Pendulum-v1', g=9.81, render_mode='human')

def scale_state(state):
    #scale values to [0,1] then multiply by partition width
    scaled_state = []
    for i, val in enumerate(state):
        scaled_state.append(np.interp(val, (state_ranges[i][0], state_ranges[i][1]), (0, 1)) * partition_width)

    return scaled_state

def e_greedy_policy(state_tiles):
    if np.random.rand() < epsilon:
        action_vals = np.sum(weights[:, tuple(state_tiles)], axis=1)
        # print(action_vals)
        chosen_action = np.argmax(action_vals)
        return chosen_action
    else:
        return np.random.randint(0, num_of_actions)

def softmax_policy(state_tiles):
    action_preferences = []
    
    for i, action in enumerate(actions):
        numerator = np.exp(np.sum(actor_weights[i][state_tiles]))
        denominator = np.sum(np.exp(np.sum(actor_weights[:,state_tiles], axis=1)))
        action_preference = numerator / denominator
        action_preferences.append(action_preference)

    chosen_action = np.random.choice(len(action_preferences), p=action_preferences)
    return chosen_action

def gradient_of_log_of_softmax(state_tiles, action):
    feature_vector = np.zeros(iht_size)
    feature_vector[state_tiles] = 1
    softmax_distribution_sum = 0
    
    for i, _ in enumerate(actions):
        feature_vector_b = np.zeros(iht_size)
        feature_vector_b[state_tiles] = 1
        softmax_distribution_sum += 

if __name__ == "__main__":
    iht_size = 8192
    iht = IHT(iht_size)

    state_ranges = [[-1, 1], [-1, 1], [-8, 8]]
    # num_of_actions = 30
    # action_range = [-2, 2]
    # actions = np.linspace(action_range[0], action_range[1], num_of_actions)
    num_of_actions = 3
    actions = [-1, 0, 1]
    epsilon = 0.9
    step_size = 0.1
    reward_step_size = 0.1
    critic_step_size = 0.1
    actor_step_size = 0.1
    gamma = 0.9

    critic_weights = np.ones((iht_size))
    actor_weights = np.ones((num_of_actions, iht_size))
    weights = np.ones((num_of_actions, iht_size))
    num_of_tilings = 32
    partition_width = 8

    num_of_episodes = 1000

    for i in range(num_of_episodes):
        last_state = scale_state(env.reset()[0])
        last_state_tiles = tiles(iht, num_of_tilings, last_state)
        # last_action = e_greedy_policy(last_state_tiles)
        last_action = softmax_policy(last_state_tiles)
        total_reward = 0
        total_step_count = 0
        step_count = 0
        avg_reward = 0
        done = False

        while not done:
            # last_state_action_value = np.sum(weights[last_action][last_state_tiles])
            last_state_action_value = np.sum(actor_weights[last_action][last_state_tiles])
            last_state_value = np.sum(critic_weights[last_state_tiles])
            new_state, reward, done, truncated, info = env.step([actions[last_action]])
            new_state_tiles = tiles(iht, num_of_tilings, scale_state(new_state))
            # new_action = e_greedy_policy(new_state_tiles)
            new_action = softmax_policy(new_state_tiles)
            new_state_action_value = np.sum(weights[new_action][new_state_tiles])
            
            td_error = reward - avg_reward + new_state_action_value - last_state_action_value
            avg_reward += reward_step_size * td_error

            weights[last_action, last_state_tiles] += step_size * (reward + gamma * new_state_action_value - last_state_action_value)
            last_state = new_state
            last_state_tiles = new_state_tiles
            last_action = new_action

            total_reward += reward

            step_count += 1

            # if step_count % 500 == 0:
            #     total_step_count += 500
            #     print(f"Avg reward: {total_reward / 500}, step_count: {total_step_count}")
            #     step_count = 0
            #     total_reward = 0
            #     done = True