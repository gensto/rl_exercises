import time 
import gym
import numpy as np

# env = gym.make("CartPole-v1", render_mode='human')
env = gym.make("CartPole-v0")
print(env.action_space.n)
print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)

def create_bin_edges(ranges, num_of_bins):
    return np.array([np.linspace(ranges[0][0], ranges[0][1], num_of_bins),
                         np.linspace(ranges[1][0], ranges[1][1], num_of_bins),
                         np.linspace(ranges[2][0], ranges[2][1], num_of_bins),
                         np.linspace(ranges[3][0], ranges[3][1], num_of_bins)])

def discretize_state2(state):
    return (
        np.digitize(state[0], bin_edges[0]) - 1,
        np.digitize(state[1], bin_edges[1]) - 1,
        np.digitize(state[2], bin_edges[2]) - 1,
        np.digitize(state[3], bin_edges[3]) - 1
    )

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
    
    if np.random.rand() < epsilon:
        chosen_action = np.argmax(q_table[tuple(state)])
    else:
        chosen_action = np.random.randint(0, env.action_space.n)
    
    return chosen_action

if __name__ == "__main__":
    q_table = np.random.uniform(low=-2, high=0, size=([20, 20, 20, 20] + [env.action_space.n]))
    # q_table = np.zeros([40, 40, 40, 40, 2])
    ranges = [[-4.8, 4,8], [-4, 4], [-0.418, 0.418], [-5, 5]]
    # ranges = [[-2.4, 2.4], [-5, 5], [-0.21, 0.21], [-5, 5]]
    epsilon = 0.90
    gamma = 0.95
    num_of_bins = 20
    bin_edges = create_bin_edges(ranges, num_of_bins)
    print(bin_edges[0])
    step_size = 0.1
    average_reward = 0
    total_reward = 0
    episode_count = 0
    for i in range(10000):
        if i % 1000 == 0:
            episode_count += 1000
            average_reward = total_reward / 1000
            print(f"Average reward for past 50 episodes: {average_reward}, total eps: {episode_count}")
            total_reward = 0
        done = False
        last_state = discretize_state2(env.reset()[0])
        
        while not done:
            # action = env.action_space.sample()
            last_action = e_greedy_policy(last_state, epsilon)
            obs, reward, done, truncated, info = env.step(last_action)
            if not done:
                # current_state = discretize_state(obs, ranges, num_of_intervals)
                current_state = discretize_state2(obs)
                last_state_action_value = q_table[last_state + (last_action,)]
                max_next_state_action_value = np.max(q_table[current_state])
                # q_table[last_state, last_action] += step_size * (reward + gamma * max_next_state_action_value - last_state_action_value)
                q_table[last_state + (last_action,)] = (1 - step_size) * q_table[last_state + (last_action,)] + step_size * (reward + gamma * max_next_state_action_value)
                # print(f"Discretized state: {discretize_state(obs, ranges, num_of_intervals)}")
                # print(reward)
                last_state = current_state
                total_reward += 1
                if episode_count > 5000:
                    env.render()
                    time.sleep(0.01)
            else:
                # q_table[last_state, last_action] += step_size * (-10 + gamma * max_next_state_action_value - last_state_action_value)
                q_table[last_state + (last_action,)] = (1 - step_size) * q_table[last_state + (last_action,)] + step_size * (-100)
        # epsilon *= 0.99
                
    env.close()