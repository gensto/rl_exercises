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

def e_greedy_policy(obs, epsilon):
    discretized_state = discretize_state(obs)
    chosen_action = None
    max_action = None
    max_val = float('-inf')

    for action, q_val in q_table[tuple(discretized_state)]:
        if q_val > max_val:
            chosen_action = action
            max_val = q_val

    if np.random.rand() < epsilon:
        chosen_action = max_action
    else:
        # choose non-greedy action, since there's only one other action
        for i in range(2):
            if i != max_action:
                chosen_action = i
    
    return chosen_action

if __name__ == "__main__":
    q_table = np.zeros([10, 10, 10, 10, 2])

    for i in range(10):
        obs = env.reset()
        last_state = None
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(done)
            ranges = [[-4.8, 4.8], [-5, 5], [-0.21, 0.21], [-5, 5]]
            print(f"Discretized state: {discretize_state(obs, ranges, 10)}")
            print(reward)
            env.render()
            time.sleep(0.01)
    env.close()