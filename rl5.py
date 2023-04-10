import time 
import gym
import numpy as np

env = gym.make('Pendulum-v1', g=9.81, render_mode='human')

def get_discrete_state(state):
    discrete_state = []
    for i, bin in enumerate(bins):
        discrete_state.append(np.digitize(state[i], bin) - 1)
    
    return tuple(discrete_state)

def create_bins(size_of_bins, ranges):
    new_bins = []
    for i, range in enumerate(ranges):
        new_bins.append(np.linspace(range[0], range[1], size_of_bins[i]))
    
    return new_bins

def e_greedy_policy(state):
    if np.random.rand() < epsilon:
        return [actions[np.argmax(q_table[state])]]
    else:
        return [actions[np.random.randint(num_of_actions)]]

def softmax_policy(state):
    action_preferences = np.array([])
    max_action_value = np.max(q_table[state])
    for action in actions:
        numerator = np.exp(q_table[state + (np.digitize(action, actions) - 1,)] - max_action_value)
        denominator = 0
        for action_b in actions:
            denominator += np.exp(q_table[state + (np.digitize(action_b, actions) - 1,)] - max_action_value)
        
        print(f"State action val: {np.exp(q_table[state + (np.digitize(action, actions) - 1,)])}")
        print(f"Num/den: {(numerator / denominator)}, num: {numerator}, den: {denominator}")
        action_preferences = np.append(action_preferences, (numerator / denominator))

    print(f"action pref Shape: {action_preferences.shape}")
    print(f"actions shape: {actions.shape}")
    chosen_action = np.random.choice(actions, p=action_preferences)
    
    return [np.digitize(chosen_action, actions)]

if __name__ == "__main__":
    obs = env.reset()
    step_size = 0.1
    epsilon = 0.9
    gamma = 0.5
    size_of_bins = [21, 21, 65]
    num_of_actions = 21
    actions = np.linspace(-2, 2, num_of_actions) # discretize action space
    q_table = np.random.uniform(low=-2, high=0, size=(tuple(size_of_bins) + (num_of_actions, )))
    ranges = [[-1, 1], [-1, 1], [-8, 8]]
    bins = create_bins(size_of_bins, ranges)
    done = False
    num_episodes = 1000
    
    for i in range(num_episodes): 
        obs = env.reset()
        last_state = get_discrete_state(obs[0])
        total_reward = 0
        step_count = 0
        total_step_count = 0
        
        while not done:
            if step_count % 500 == 0:
                total_step_count += 500
                print(f"Avg reward: {total_reward / 500}, step_count: {total_step_count}")
                step_count = 0
            action = e_greedy_policy(last_state)
            # action = softmax_policy(last_state)
            new_state, reward, done, truncated, info = env.step(action)
            new_state = get_discrete_state(new_state)
            total_reward += reward
            if not done:
                last_state_action_value = q_table[last_state + (np.digitize(action[0], actions) - 1, )]
                max_new_state_action_value = np.max(q_table[new_state])
                q_table[last_state + (np.digitize(action[0], actions) - 1, )] += step_size * (reward + gamma * max_new_state_action_value - last_state_action_value)
            else:
                q_table[last_state + (np.digitize(action[0], actions) - 1, )] += step_size * (-100)
            step_count += 1
            # print(step_count)
            time.sleep(0.01)
        print('done')
    # env.close()
        