import gym
import numpy as np

class NeuralNetworkAgent:
    def __init__(self, n_states, n_actions, n_hidden=32, lr=0.01, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.lr = lr
        self.gamma = gamma

        self.W1 = np.random.randn(n_states, n_hidden)
        self.W2 = np.random.randn(n_hidden, n_actions)

    def forward(self, state):
        state = state.reshape(-1, self.n_states)
        hidden = np.maximum(0, np.dot(state, self.W1))
        output = np.dot(hidden, self.W2)
        return output

    def choose_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            output = self.forward(state)
            return np.argmax(output)

    def update(self, state, action, reward, next_state, done):
        output_current = self.forward(state)
        v_current = output_current[0, action]

        if done:
            target = reward
        else:
            output_next = self.forward(next_state)
            target = reward + self.gamma * np.max(output_next)

        td_error = target - v_current

        # Backpropagation
        hidden = np.maximum(0, np.dot(state.reshape(-1, self.n_states), self.W1))
        d_output = np.zeros((1, self.n_actions))
        d_output[0, action] = td_error

        d_hidden = np.dot(d_output, self.W2.T)
        d_hidden[hidden <= 0] = 0

        dW1 = np.dot(state.reshape(self.n_states, -1), d_hidden)
        dW2 = np.dot(hidden.T, d_output)

        self.W1 += self.lr * dW1
        self.W2 += self.lr * dW2

# env = gym.make('CartPole-v0')
env = gym.make("CartPole-v1", render_mode='human')
n_episodes = 1000

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

agent = NeuralNetworkAgent(n_states, n_actions)

for episode in range(n_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    print(f'Episode {episode + 1}, Total Reward: {total_reward}')

env.close()