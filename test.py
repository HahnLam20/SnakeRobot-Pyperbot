import numpy as np
import tensorflow as tf
from collections import deque

# Define the Deep Q-Network
class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(24, input_dim=state_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_size, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

# Define the Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model(np.array([state]))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = np.array(random.sample(self.memory, batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = self.target_model(np.array([state])).numpy()[0]
            if done:
                target[action] = reward
            else:
                Q_future = max(self.target_model(np.array([next_state])).numpy()[0])
                target[action] = reward + self.gamma * Q_future

            with tf.GradientTape() as tape:
                predictions = self.model(np.array([state]), training=True)
                loss = tf.reduce_mean(tf.square(target - predictions[0]))

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

# Usage example:
state_size = 4  # Example state size
action_size = 2  # Example action size
agent = DQNAgent(state_size, action_size)

# Training loop
for episode in range(num_episodes):
    # Run the snake in the environment, collect experiences, and update the agent
    # Update the target network occasionally with the main network's weights
    agent.target_train()
