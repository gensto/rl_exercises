import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym

def create_neural_network(input_size, hidden_layer_size, output_size):
    model = tf.keras.Sequential([layers.Dense(units=hidden_layer_size, activation='relu', input_shape=(input_size,)),
                                 layers.Dense(output_size)])
    model.compile(optimizer='adam', loss='mse')
    
    return model

def e_greedy(qa_values):
    if np.random.rand() < 0.9:
        return np.argmax(qa_values)
    else:
        return np.random.randint(4)
    
def normalize_state_vector(state):
    for i, value in enumerate(state):
        old_max = ranges[i][0]
        old_min = ranges[i][1]
        new_max = 1
        new_min = 0
        old_value = value
        old_range = (old_max - old_min)  
        new_range = (new_max - new_min)  
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min
        state[i] = new_value
    
    return np.expand_dims(state, axis=0)

if __name__ == "__main__":
    np.random.seed(123)
    # env = gym.make('CliffWalking-v0', render_mode='human')
    env = gym.make("CartPole-v0")
    env.action_space.seed(42)
    
    model = create_neural_network(4, 20, 2)
    print("sdffd")
    
    num_of_episodes = 1000
    ranges = [[-4.8, 4,8], [-4, 4], [-0.418, 0.418], [-5, 5]]
    
    for n in range(num_of_episodes):
        last_state = normalize_state_vector(env.reset(seed=42)[0])
        last_action = e_greedy(last_state)
        done = False
        while not done:
            print(f"Last state shape: {last_state.shape}")
            last_state_qa_value = model.predict(last_state)
            print(f"Last qa value: {last_state_qa_value}")
            # new_state, reward, done, truncated, info = env.step(last_action)
            # new_state = normalize_state_vector(new_state)
            # next_state_qa_values = model.predict(new_state)
            # print(next_state_qa_values)