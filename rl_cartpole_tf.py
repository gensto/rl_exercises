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
        return np.random.randint(2)
    
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
    
    normalized_feature_vectors = []
    for i in range(2):    
        normalized_state = np.zeros(8)
        normalized_state[i * 4:(i * 4) + 4] = state
        normalized_feature_vectors.append(np.expand_dims(normalized_state, axis=0))
        
    return np.array(normalized_feature_vectors)

if __name__ == "__main__":
    np.random.seed(123)
    # env = gym.make('CliffWalking-v0', render_mode='human')
    # env = gym.make("CartPole-v0")
    env = gym.make("CartPole-v1", render_mode='human')
    env.action_space.seed(42)
    
    model = create_neural_network(8, 20, 1)
    
    num_of_episodes = 100
    gamma = 0.5
    ranges = [[-4.8, 4,8], [-4, 4], [-0.418, 0.418], [-5, 5]]
    
    for n in range(num_of_episodes):
        last_state_action_vectors = normalize_state_vector(env.reset(seed=42)[0])
        print(last_state_action_vectors[0].shape)
        last_state_action_values = [model.predict(last_state_action_vector) for last_state_action_vector in last_state_action_vectors]
        last_action = e_greedy(last_state_action_values)
        done = False
        total_reward = 0
        
        while not done:
            new_state, reward, done, truncated, info = env.step(last_action)
            new_state_action_vectors = normalize_state_vector(new_state)
            new_state_action_values = [model.predict(new_state_action_vector) for new_state_action_vector in new_state_action_vectors]
            new_action = e_greedy(new_state_action_values)
            
            td_target = reward + gamma * np.max(new_state_action_values)
            td_target_vector = np.array([[td_target]])
            print(td_target)
            model.fit(last_state_action_vectors[last_action], td_target_vector, epochs=1, verbose=0)
            
            last_state_action_vectors = new_state_action_vectors
            last_state_action_values = new_state_action_values
            last_action = new_action
            total_reward += reward
            
            if done:
                print(total_reward)

