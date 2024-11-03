import os
from datetime import datetime

import tensorflow as tf

from environment import TwoPlayerGameEnvironment


class DQN(tf.keras.Model):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', name='dense1')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu', name='dense2')
        self.dense3 = tf.keras.layers.Dense(action_size, name='dense3')

    def call(self, state: tf.Tensor) -> tf.Tensor:
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x)

    def get_config(self):
        return {
            'state_size': self.state_size,
            'action_size': self.action_size
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


import tensorflow as tf
import numpy as np
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_size: int, max_actions: int):
        self.state_size = state_size
        self.max_actions = max_actions  # Maximum possible actions

        # Hyperparameters
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.invalid_action_penalty = -5.0  # Significant penalty for invalid actions

        # Networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

    def _build_model(self):
        input_state = tf.keras.layers.Input(shape=(self.state_size,))
        input_mask = tf.keras.layers.Input(shape=(self.max_actions,))

        x = tf.keras.layers.Dense(256, activation='relu')(input_state)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        q_values = tf.keras.layers.Dense(self.max_actions)(x)

        # Simple masking using multiplication
        masked_q_values = tf.keras.layers.Multiply()([q_values, input_mask])

        # Use a very large negative constant for invalid actions
        invalid_penalty = -1000.0
        invalid_mask = tf.keras.layers.Lambda(lambda x: (1 - x) * invalid_penalty)(input_mask)
        final_q_values = tf.keras.layers.Add()([masked_q_values, invalid_mask])

        model = tf.keras.Model(inputs=[input_state, input_mask], outputs=final_q_values)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def _get_action_mask(self, num_available_actions: int) -> np.ndarray:
        """Create a mask for valid actions"""
        mask = np.zeros(self.max_actions)
        mask[:num_available_actions] = 1.0
        return mask

    def update_target_network(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool, num_actions: int, invalid_action: bool):
        """Store experience in memory with invalid action flag"""
        # Apply penalty for invalid actions
        if invalid_action:
            reward += self.invalid_action_penalty
        self.memory.append((state, action, reward, next_state, done, num_actions))

    def act(self, state: np.ndarray, num_available_actions: int) -> int:
        """Return action using epsilon-greedy policy with action masking"""
        if random.random() < self.epsilon:
            return random.randrange(num_available_actions)

        state = np.array(state).reshape((1, self.state_size))
        action_mask = self._get_action_mask(num_available_actions)
        action_mask = np.reshape(action_mask, (1, self.max_actions))

        act_values = self.model.predict([state, action_mask], verbose=0)
        return np.argmax(act_values[0][:num_available_actions])

    def replay(self):
        """Train on a batch of experiences from memory"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        num_actions_list = np.array([exp[5] for exp in minibatch])

        # Create action masks for current and next states
        current_masks = np.array([self._get_action_mask(n) for n in num_actions_list])
        next_masks = current_masks.copy()

        # Predict Q-values for current states
        current_q_values = self.model.predict([states, current_masks], verbose=0)
        next_q_values = self.target_model.predict([next_states, next_masks], verbose=0)

        targets = current_q_values.copy()

        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                # Only consider valid actions for next state
                valid_next_q_values = next_q_values[i][:num_actions_list[i]]
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(valid_next_q_values)

        # Train the model
        self.model.fit([states, current_masks], targets, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: str):
        """Save the agent's models and parameters"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save main network with .keras extension
        model_path = filepath + '_model.keras'
        self.model.save(model_path)

        # Save parameters
        params = {
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'state_size': self.state_size,
        }
        params_path = filepath + '_params.npy'
        np.save(params_path, params)

    @classmethod
    def load(cls, filepath: str):
        """Load a saved agent"""
        # Load parameters
        params_path = filepath + '_params.npy'
        params = np.load(params_path, allow_pickle=True).item()

        # Create new agent with saved parameters
        agent = cls(params['state_size'], params['action_size'])

        # Load model
        model_path = filepath + '_model.keras'
        agent.model = tf.keras.models.load_model(model_path)
        agent.target_model = tf.keras.models.load_model(model_path)

        # Set other parameters
        agent.epsilon = params['epsilon']
        agent.gamma = params['gamma']
        agent.learning_rate = params['learning_rate']

        return agent


class TwoPlayerTrainer:
    def __init__(self):
        self.env = TwoPlayerGameEnvironment()
        self.state_size = 16
        self.max_actions = 8

        # Set up directories
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('models', f'training_{self.timestamp}')
        self.log_dir = os.path.join('logs', self.timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Create agents
        self.agent1 = DQNAgent(self.state_size, self.max_actions)
        self.agent2 = DQNAgent(self.state_size, self.max_actions)

        # Training metrics
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.history = {
            'agent1_rewards': [],
            'agent2_rewards': [],
            'agent1_wins': 0,
            'agent2_wins': 0,
            'draws': 0,
            'episodes': 0,
            'invalid_actions': 0
        }

    def train(self, episodes: int, save_freq: int = 100):
        progbar = tf.keras.utils.Progbar(
            episodes,
            stateful_metrics=[
                'Player', 'A1_WR', 'A2_WR', 'A1_eps', 'A2_eps',
                'Turns', 'Rounds', 'Invalid', 'Avg_Reward'
            ]
        )

        for episode in range(episodes):
            states = self.env.reset()
            total_rewards = [0, 0]
            done = False
            turn_count = 0
            rounds_played = 1
            invalid_actions = 0
            current_player_idx = self.env.get_current_player()

            while not done:
                current_agent = self.agent1 if current_player_idx == 0 else self.agent2
                current_state = states[current_player_idx]

                # Calculate available actions
                current_player = self.env.game.players[current_player_idx]
                num_items = len(current_player.items)
                num_available_actions = num_items * 2 + 2

                # Get action
                action = current_agent.act(current_state, num_available_actions)

                try:
                    next_states, rewards, done, info = self.env.step(current_player_idx, action)

                    is_invalid = info.get("invalid_action", False)
                    if is_invalid:
                        invalid_actions += 1

                    # Store experience with invalid action flag
                    current_agent.remember(
                        current_state,
                        action,
                        rewards[current_player_idx],
                        next_states[current_player_idx],
                        done,
                        num_available_actions,
                        is_invalid
                    )

                    current_agent.replay()

                    # Update states and rewards
                    states = next_states
                    total_rewards[0] += rewards[0]
                    total_rewards[1] += rewards[1]
                    turn_count += 1

                    if info.get("round_ended", False):
                        rounds_played += 1

                    # Update current player
                    if not done:
                        current_player_idx = self.env.get_current_player()

                except ValueError as e:
                    print(f"Error during step: {e}")
                    break

            # Update history
            self.history['episodes'] += 1
            self.history['agent1_rewards'].append(total_rewards[0])
            self.history['agent2_rewards'].append(total_rewards[1])
            self.history['invalid_actions'] += invalid_actions

            if total_rewards[0] > total_rewards[1]:
                self.history['agent1_wins'] += 1
            elif total_rewards[1] > total_rewards[0]:
                self.history['agent2_wins'] += 1
            else:
                self.history['draws'] += 1

            # Calculate average reward for this episode
            avg_reward = (total_rewards[0] + total_rewards[1]) / 2

            # Log to TensorBoard
            with self.writer.as_default():
                tf.summary.scalar('reward/agent1', total_rewards[0], step=episode)
                tf.summary.scalar('reward/agent2', total_rewards[1], step=episode)
                tf.summary.scalar('reward/average', avg_reward, step=episode)
                tf.summary.scalar('training/agent1_epsilon', self.agent1.epsilon, step=episode)
                tf.summary.scalar('training/agent2_epsilon', self.agent2.epsilon, step=episode)
                tf.summary.scalar('game/turns', turn_count, step=episode)
                tf.summary.scalar('game/rounds', rounds_played, step=episode)
                tf.summary.scalar('game/invalid_actions', invalid_actions, step=episode)

            # Update progress bar
            win_rate_1 = self.history['agent1_wins'] / (episode + 1)
            win_rate_2 = self.history['agent2_wins'] / (episode + 1)

            metrics = [
                ('Player', current_player_idx),
                ('A1_WR', win_rate_1),
                ('A2_WR', win_rate_2),
                ('A1_eps', self.agent1.epsilon),
                ('A2_eps', self.agent2.epsilon),
                ('Turns', turn_count),
                ('Rounds', rounds_played),
                ('Invalid', invalid_actions),
                ('Avg_Reward', avg_reward)
            ]

            progbar.update(episode + 1, metrics)

            if episode % save_freq == 0:
                self._save_checkpoint(episode)

        self._save_checkpoint(episodes, is_final=True)

    def _save_checkpoint(self, episode: int, is_final: bool = False):
        """Save model checkpoints and training history"""
        prefix = 'final' if is_final else f'checkpoint_ep_{episode}'

        # Save models
        model_path = os.path.join(self.save_dir, prefix)
        self.agent1.save(f"{model_path}_agent1")
        self.agent2.save(f"{model_path}_agent2")

        # Save history
        np.save(f"{model_path}_history.npy", self.history)


trainer = TwoPlayerTrainer()
trainer.train(episodes=1000, save_freq=100)
