import time 
import gym
import numpy as np

env = gym.make("CartPole-v1", render_mode='human')
# env = gym.make("CartPole-v1")
print(env.action_space.n)
print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)

def create_bin_edges(ranges, num_of_bins):
    return np.array([np.linspace(ranges[0][0], ranges[0][1], num_of_bins),
                         np.linspace(ranges[1][0], ranges[1][1], num_of_bins),
                         np.linspace(ranges[2][0], ranges[2][1], num_of_bins),
                         np.linspace(ranges[3][0], ranges[3][1], num_of_bins)])

def discretize_state2(state):
    return [
        np.digitize(state[0], bin_edges[0]) - 1,
        np.digitize(state[1], bin_edges[1]) - 1,
        np.digitize(state[2], bin_edges[2]) - 1,
        np.digitize(state[3], bin_edges[3]) - 1
    ]

def discretize_state(state, ranges, num_of_intervals):
    # ranges will be formatted like: [[-4.8, 4.8], [-5, 5], [-0.21, 0.21], [-5, 5]]
    cart_position_index, cart_velocity_index, pole_angle_index, pole_angular_velocity_index = 0,0,0,0
    for i, range in enumerate(ranges):
        if i == 0: # cart_position_index
            difference = range[1] - range[0]
            interval_distance = difference / num_of_intervals
            cart_position_index = int(np.floor((state[i] - range[0]) / interval_distance))
        elif i == 1:
            difference = range[1] - range[0]
            interval_distance = difference / num_of_intervals
            cart_velocity_index = int(np.floor((state[i] - range[0]) / interval_distance))
        elif i == 2:
            difference = range[1] - range[0]
            interval_distance = difference / num_of_intervals
            pole_angle_index = int(np.floor((state[i] - range[0]) / interval_distance))
        elif i == 3:
            difference = range[1] - range[0]
            interval_distance = difference / num_of_intervals
            pole_angular_velocity_index = int(np.floor((state[i] - range[0]) / interval_distance))
    
    return [cart_position_index, cart_velocity_index, pole_angle_index, pole_angular_velocity_index]

def e_greedy_policy(state, epsilon):
    chosen_action = None
    max_action = None
    max_val = float('-inf')
    
    for action, q_val in enumerate(q_table[tuple(state)]):
        if q_val > max_val:
            max_action = action
            max_val = q_val

    if np.random.rand() < epsilon:
        chosen_action = max_action
    else:
        # choose non-greedy action, since there's only one other action
        for i in range(2):
            if i != max_action:
                chosen_action = i
    
    return chosen_action

def max_action(state):
    max_action = None
    max_val = float('-inf')
    for action, q_val in enumerate(q_table[tuple(state)]):
        if q_val > max_val:
            max_val = q_val
            max_action = action
    return max_action

if __name__ == "__main__":
    num_of_bins = 20
    q_table = np.random.uniform(low=-2, high=0, size=([num_of_bins, num_of_bins, num_of_bins, num_of_bins] + [env.action_space.n]))
    # q_table = np.zeros([40, 40, 40, 40, 2])
    # ranges = [[-2.4, 2.4], [-5, 5], [-0.21, 0.21], [-5, 5]]
    ranges = [[-4.8, 4,8], [-4, 4], [-0.418, 0.418], [-5, 5]]
    num_of_intervals = 40
    epsilon = 0.99
    gamma = 0.95
    bin_edges = create_bin_edges(ranges, num_of_bins)
    print(bin_edges[0])
    step_size = 0.1
    average_reward = 0
    total_reward = 0
    episode_count = 0
    for i in range(10000):
        if i % 50 == 0:
            average_reward = total_reward / 50
            print(f"Average reward for past 50 episodes: {average_reward}, episode count: {episode_count}")
            total_reward = 0
        obs = env.reset()
        done = False
        # last_state = discretize_state(obs[0], ranges, num_of_intervals)
        last_state = discretize_state2(obs[0])
        while not done:
            # action = env.action_space.sample()
            last_action = e_greedy_policy(last_state, epsilon)
            obs, reward, done, truncated, info = env.step(last_action)
            if not done:
                # current_state = discretize_state(obs, ranges, num_of_intervals)
                current_state = discretize_state2(obs)
                last_state_action_value = q_table[tuple(last_state) + (last_action,)]
                max_next_state_action_value = np.max(q_table[tuple(current_state)])
                # q_table[last_state, last_action] += step_size * (reward + gamma * max_next_state_action_value - last_state_action_value)
                # q_table[tuple(last_state) + (last_action,)] = (1 - step_size) * q_table[tuple(last_state) + (last_action,)] + step_size * (reward + gamma * max_next_state_action_value)
                q_table[tuple(last_state) + (last_action,)] += step_size * (reward + gamma * max_next_state_action_value - last_state_action_value)
                # print(f"Discretized state: {discretize_state(obs, ranges, num_of_intervals)}")
                # print(reward)
                last_state = current_state
                total_reward += 1
                # env.render()
                # time.sleep(0.01)
            else:
                # q_table[last_state, last_action] += step_size * (-10 + gamma * max_next_state_action_value - last_state_action_value)
                # q_table[tuple(last_state) + (last_action,)] = (1 - step_size) * q_table[tuple(last_state) + (last_action,)] + step_size * (-100)
                q_table[tuple(last_state) + (last_action,)] += step_size * (-100)
                episode_count += 1
                print(f"Episode count: {episode_count}")
                
    env.close()