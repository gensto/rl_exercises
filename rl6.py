from tile_coding import IHT, tiles
import time 
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(123)
env = gym.make('MountainCar-v0')
# env = gym.make('MountainCar-v0', render_mode='human')
env._max_episode_steps = 1000
env.action_space.seed(42)

def discretize_state(state):
    discretized_position = np.digitize(state[0], position_bins) - 1
    discretized_velocity = np.digitize(state[1], velocity_bins) - 1
    return tuple([discretized_position, discretized_velocity])

def epsilon_greedy_policy(state):
    if np.random.rand() < epsilon:
        return np.argmax(q_table[state])
    else:
        return np.random.choice(np.arange(3))

def softmax_policy(state):
    action_preferences = np.array([])
    max_action_value = np.max(q_table[state])
    for i, action in enumerate(actions):
        numerator = np.exp(q_table[state + (i, )] - max_action_value)
        denominator = 0
        for j, action_b in enumerate(actions):
            denominator += np.exp(q_table[state + (j, )] - max_action_value)
        
        # print(f"State action val: {np.exp(q_table[state + (np.digitize(action, actions) - 1,)])}")
        # print(f"Num/den: {(numerator / denominator)}, num: {numerator}, den: {denominator}")
        action_preferences = np.append(action_preferences, (numerator / denominator))

    # print(f"action pref Shape: {action_preferences.shape}")
    # print(f"actions shape: {actions.shape}")
    chosen_action = np.random.choice(actions, p=action_preferences)
    
    return np.digitize(chosen_action, actions) - 1

if __name__ == "__main__":
    actions = np.arange(3)
    num_of_bins = 10
    position_bins = np.linspace(-1.2, 0.6, num_of_bins)
    velocity_bins = np.linspace(-0.7, 0.7, num_of_bins)
    # q_table = np.random.uniform(low=-2, high=0, size=((num_of_bins, num_of_bins, 3)))
    q_table = np.zeros((num_of_bins, num_of_bins, 3))
    
    num_of_episodes = 3000
    epsilon = 1
    gamma = 0.99
    step_size = 0.1
    
    rewards_per_episode = []

    for i in range(num_of_episodes):
        # if i == 1000:
        #     env = gym.make('MountainCar-v0', render_mode='human')
        #     env.action_space.seed(42)
        last_state = discretize_state(env.reset(seed=42)[0])
        truncated = False
        total_reward = 0
        step_count = 0
        
        while not truncated:
            # action = epsilon_greedy_policy(last_state)
            action = softmax_policy(last_state)
            last_state_action_value = q_table[last_state + (action, )]
            new_state, reward, done, truncated, info = env.step(action)
            new_state = discretize_state(new_state)
            max_next_action = np.argmax(q_table[new_state])
            new_state_action_value = q_table[new_state + (max_next_action, )]
            td_update = reward + step_size * (gamma * new_state_action_value - last_state_action_value)
            q_table[last_state + (action, )] += td_update
            
            last_state = new_state
            total_reward += reward
            if truncated:
                # print(total_reward)
                rewards_per_episode.append(total_reward)

        epsilon = epsilon - 2/i if epsilon < 0.01 else 0.01
                
    # plt.plot(rewards_per_episode)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("Rewards per Episode")
    # # xtick_positions = np.arange(0, num_of_episodes / 50)
    # # xtick_labels = xtick_positions * 50
    # # plt.xticks(xtick_positions, xtick_labels)
    # plt.show()
    
X = position_bins
Y = velocity_bins
X, Y = np.meshgrid(X, Y)
Z = np.max(q_table, axis=2)

# Set up the figure and axis for the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='coolwarm')

# Set axis labels
ax.set_xlabel('X-axis (e.g., position)')
ax.set_ylabel('Y-axis (e.g., velocity)')
ax.set_zlabel('Z-axis (max state action values)')

# Show the plot
plt.show()