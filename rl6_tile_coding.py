from tile_coding import IHT, tiles
import time 
import gym
import gym.vector
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

def softmax_policy(state_tiles):
    preference_values = np.sum(weights[:,state_tiles], axis=1)
    action_preferences = np.exp(preference_values - logsumexp(preference_values))
    chosen_action = np.random.choice(len(action_preferences), p=action_preferences)
    return chosen_action

# def e_greedy_policy(state_tiles):
#     if np.random.rand() < epsilon:
#         action_vals = np.sum(weights[:, tuple(state_tiles)], axis=1)
#         # print(action_vals)
#         chosen_action = np.argmax(action_vals)
#         return chosen_action
#     else:
#         return np.random.randint(0, 3)
def e_greedy_policy(state_action_tiles_list):
    max_state_action_val = float('-inf')
    max_action = 0
    for i in range(len(state_action_tiles_list)):
        if np.sum(weights[i, state_action_tiles_list[i]]) > max_state_action_val:
            max_state_action_val = np.sum(weights[i, state_action_tiles_list[i]])
            max_action = i
    
    return max_state_action_val, max_action

def scale_state(state):
    #scale values to [0,1] then multiply by partition width
    scaled_state = []
    for i, val in enumerate(state):
        scaled_state.append(np.interp(val, (state_ranges[i][0], state_ranges[i][1]), (0, 1)) * partition_width)

    return scaled_state

if __name__ == "__main__":
    np.random.seed(123)
    num_envs = 4
    env = gym.make('MountainCar-v0') 
    # env = gym.vector.make("MountainCar-v0", num_envs=num_envs)
    env._max_episode_steps = 1000
    env.action_space.seed(42)

    partition_width = 10
    num_of_tilings = 8
    state_ranges = [[-1.2, 0.6], [-0.7, 0.7]]
    iht_size = 4096
    iht = IHT(iht_size)

    weights = np.zeros((3, iht_size))

    num_of_episodes = 1000
    rewards_per_episode = []
    step_size = 0.5 / num_of_tilings
    gamma = 0.5
    epsilon = 0.9

    for i in range(num_of_episodes):
        last_state = scale_state(env.reset(seed=42)[0])
        # last_state_tiles = tiles(iht, num_of_tilings, last_state, last_action)
        # last_action = e_greedy_policy(last_state_tiles)
        last_state_action_tiles_list = [tiles(iht, num_of_tilings, last_state, [i]) for i in range(3)]
        last_state_action_val, last_action = e_greedy_policy(last_state_action_tiles_list)
        # last_action = softmax_policy(last_state_tiles)
        truncated = False
        done = False
        total_rewards = 0

        while not truncated and not done:
            # last_state_action_val = np.sum(weights[last_action][last_state_tiles])
            new_state, reward, done, truncated, info = env.step(last_action)
            new_state = scale_state(new_state)
            new_state_action_tiles_list = [tiles(iht, num_of_tilings, new_state, [i]) for i in range(3)]
            # print(new_state_action_tiles_list)
            new_state_action_val, new_action = e_greedy_policy(new_state_action_tiles_list)
            # new_action = softmax _policy(new_state_tiles)
            # new_state_tiles = tiles(iht, num_of_tilings, new_state, new_action)
            # new_state_action_val = np.sum(weights[new_action][new_state_tiles])

            td_error = step_size * (reward + gamma * new_state_action_val - last_state_action_val)
            # print(np.sum(weights[:,last_state_tiles], axis=1))
            weights[last_action][last_state_action_tiles_list[last_action]] += td_error

            total_rewards += reward
            # last_state = new_state
            # last_state_tiles = new_state_tiles
            last_state_action_tiles = new_state_action_tiles_list[new_action]
            last_state_action_val = new_state_action_val
            last_action = new_action

            if truncated or done:
                rewards_per_episode.append(total_rewards)
                # print(total_rewards)

    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Rewards per Episode")
    # xtick_positions = np.arange(0, num_of_episodes / 50)
    # xtick_labels = xtick_positions * 50
    # plt.xticks(xtick_positions, xtick_labels)
    plt.show()
    # num_of_plots_per_axis = 50
    # X = np.linspace(-1.2, 0.6, num_of_plots_per_axis)
    # Y = np.linspace(-0.7, 0.7, num_of_plots_per_axis)
    # Z = np.zeros((num_of_plots_per_axis, num_of_plots_per_axis))
    # for i, x in enumerate(X):
    #     for j, y in enumerate(Y):
    #         # scaled_state = scale_state([x, y])
    #         # state_tiles = tiles(iht, num_of_tilings, scaled_state)
    #         # # Z = np.append(Z, np.max(np.sum(weights[:, state_tiles])))
    #         # Z[i][j] = np.max(np.sum(weights[:, state_tiles]))

    #         scaled_state = scale_state([x, y])
    #         # state_action_tiles = [tiles(iht, num_of_tilings, scaled_state) for i in range(3)]
    #         state_action_tile_values = [np.sum(weights[i][tiles(iht, num_of_tilings, scaled_state, [i])]) for i in range(3)]
    #         Z[i][j] = np.max(state_action_tile_values)

    # print(f"Z shape: {Z.shape}")
    # X, Y = np.meshgrid(X, Y)

    # # Set up the figure and axis for the 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the surface
    # ax.plot_surface(X, Y, Z, cmap='coolwarm')

    # # Set axis labels
    # ax.set_xlabel('X-axis (e.g., position)')
    # ax.set_ylabel('Y-axis (e.g., velocity)')
    # ax.set_zlabel('Z-axis (max state action values)')

    # # Show the plot
    # plt.show()
