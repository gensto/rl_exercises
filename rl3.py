import time 
import gym
import numpy as np

env = gym.make("CartPole-v1", render_mode='human')

print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)


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
    q_table = np.zeros([10, 10, 10, 10, 2])
    ranges = [[-4.8, 4.8], [-5, 5], [-0.21, 0.21], [-5, 5]]
    num_of_intervals = 10
    epsilon = 0.9
    gamma = 0.5
    step_size = 0.5
    obs = env.reset()
    last_state = discretize_state(obs[0], ranges, num_of_intervals)
    done = False
    for i in range(10):
        while not done:
            # action = env.action_space.sample()
            last_action = e_greedy_policy(last_state, epsilon)
            print(f"last_state: {last_state}")
            print(f"obs: {obs}")
            obs, reward, done, truncated, info = env.step(last_action)
            print(f"done: {done}, reward: {reward}")
            if not done:
                current_state = discretize_state(obs, ranges, num_of_intervals)
                last_state_action_value = q_table[tuple(last_state), last_action]
                max_next_state_action_value = q_table[tuple(current_state), max_action(current_state)]
                print(f"reward: {reward}")
                q_table[last_state, last_action] += step_size * (reward + gamma * max_next_state_action_value - last_state_action_value)
                # print(f"Discretized state: {discretize_state(obs, ranges, num_of_intervals)}")
                # print(reward)
                last_state = current_state
                env.render()
                time.sleep(0.01)
    env.close()