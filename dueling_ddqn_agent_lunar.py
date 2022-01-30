from importlib import import_module
import tensorflow as tf
from dueling_ddqn_lunar import DuelingDeepQNetwork
from dueling_ddqn_replay_buffer_lunar import ReplayBuffer
from tensorflow.keras.optimizers import Adam
import numpy as np


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=1e-3, eps_end=0.01, 
                 mem_size=100000, fc1_dims=128,
                 fc2_dims=128, replace=100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)

        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
                            
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states)
        q_next = self.q_next(states_)
        # changing q_pred doesn't matter because we are passing states to the train function anyway
        # also, no obvious way to copy tensors in tf2?
        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        
        # improve on my solution!
        for idx, terminal in enumerate(dones):
            #if terminal:
                #q_next[idx] = 0.0
            q_target[idx, actions[idx]] = rewards[idx] + \
                    self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min

        self.learn_step_counter += 1
