from tile_coding import IHT, tiles
import time 
import gym
import numpy as np

np.random.seed(123)
env = gym.make('Pendulum-v1', g=9.81, render_mode='human')
env.action_space.seed(42)

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
    
    # for i, action in enumerate(actions):
    #     numerator = np.exp(np.sum(actor_weights[i][state_tiles]))
    #     denominator = np.sum(np.exp(np.sum(actor_weights[:,state_tiles], axis=1)))
    #     action_preference = numerator / denominator
    #     action_preferences.append(action_preference)
    for i, action in enumerate(actions):
        action_prob = softmax_probability(state_tiles, i)
        print(f"Action i: {i}, prob: {action_prob}, preference val: {softmax_numerical_preference(state_tiles, i)}")
        action_preferences.append(action_prob)

    chosen_action = np.random.choice(len(action_preferences), p=action_preferences)
    # print(action_preferences)
    return chosen_action

def softmax_numerical_preference(state_tiles, action):
    return np.sum(actor_weights[action][state_tiles])

def softmax_probability(state_tiles, action):
    max_action_val = np.max(np.sum(actor_weights[:, state_tiles], axis=1))
    numerator = np.exp(np.sum(actor_weights[action][state_tiles]) - max_action_val)
    denominator = np.sum(np.exp(np.sum(actor_weights[:,state_tiles], axis=1) - max_action_val))
    
    return numerator / denominator

def gradient_of_log_of_softmax(state_tiles, action):
    feature_vector = np.zeros(iht_size)
    feature_vector[state_tiles] = 1
    softmax_distribution_sum = 0
    
    for i, _ in enumerate(actions):
        feature_vector_b = np.zeros(iht_size)
        feature_vector_b[state_tiles] = 1
        softmax_distribution_sum += softmax_probability(state_tiles, i) * feature_vector_b
    
    return feature_vector - softmax_distribution_sum

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
    num_of_tilings = 32
    step_size = 0.01
    reward_step_size = 0.1 /num_of_tilings
    critic_step_size = 0.1 / num_of_tilings
    actor_step_size = 0.1
    gamma = 0.9

    critic_weights = np.ones((iht_size))
    actor_weights = np.ones((num_of_actions, iht_size))
    weights = np.ones((num_of_actions, iht_size))
    partition_width = 8

    num_of_episodes = 100
    terminate = False
    for i in range(num_of_episodes):
        last_state = scale_state(env.reset(seed=42)[0])
        last_state_tiles = tiles(iht, num_of_tilings, last_state)
        # last_action = e_greedy_policy(last_state_tiles)
        last_action = softmax_policy(last_state_tiles) # 0 pushes it left, 1, does nothing, 2 pushes it right
        total_reward = 0
        total_step_count = 0
        step_count = 0
        avg_reward = 0
        truncated = False

        while not truncated:
            # last_state_action_value = np.sum(weights[last_action][last_state_tiles])  
            last_state_action_value = np.sum(actor_weights[last_action][last_state_tiles])
            last_state_value = np.sum(critic_weights[last_state_tiles])
            new_state, reward, done, truncated, info = env.step([actions[last_action]])
            print(f"Truncated: {truncated}")
            print(f"New state: {new_state}")
            new_state_tiles = tiles(iht, num_of_tilings, scale_state(new_state))
            # new_action = e_greedy_policy(new_state_tiles)
            new_action = softmax_policy(new_state_tiles)
            new_state_action_value = np.sum(actor_weights[new_action][new_state_tiles])
            new_state_value = np.sum(critic_weights[new_state_tiles])
            
            td_error = reward - avg_reward + new_state_value - last_state_value
            avg_reward += reward_step_size * td_error
            critic_weights[last_state_tiles] += critic_step_size * td_error
            
            print(f"New state action value: {new_state_action_value}")
            print(f"Last state action value: {last_state_action_value}")
            print(f"reward: {reward}")
            print(f"td_error: {td_error}")
            print(f"Last action: {last_action}")
            # print(f"softmax_prob: {softmax_probability(last_state_tiles, last_action)}")
            # print(f"Before updating actor weights: {actor_weights[last_action, last_state_tiles]}")
            print(f"Actor weights before updating: {actor_weights[:, last_state_tiles]}")
            for i, a in enumerate(actions):
                if i == last_action:
                    actor_weights[i, last_state_tiles] += actor_step_size * td_error * (1 - softmax_probability(last_state_tiles, i))
                else:
                    actor_weights[i, last_state_tiles] += actor_step_size * td_error * (0 - softmax_probability(last_state_tiles, i))
            print(f"Actor weights after updating: {actor_weights[:, last_state_tiles]}")
            # print(f"After updating actor weights: {actor_weights[i, last_state_tiles]}")

            # weights[last_action, last_state_tiles] += step_size * (reward + gamma * new_state_action_value - last_state_action_value)
            last_state = new_state
            last_state_tiles = new_state_tiles
            last_action = new_action

            total_reward += reward

            step_count += 1
            time.sleep(1)
            
            # time.sleep(1)
            # if step_count % 500 == 0:
            #     total_step_count += 500
            #     print(f"Avg reward: {total_reward / 500}, step_count: {total_step_count}")
            #     step_count = 0
            #     total_reward = 0
            #     done = True