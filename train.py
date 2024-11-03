import os
import random
from collections import deque
from datetime import datetime

import numpy as np
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


class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        # Networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

    def _build_model(self):
        """Neural Net for Deep-Q learning Model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_dim=self.state_size),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def update_target_network(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """Return action for given state using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = np.array(state).reshape((1, self.state_size))
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        """Train on a batch of experiences from memory"""
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])

        # Get Q values for current states
        targets = self.model.predict(states, verbose=0)

        # Get Q values for next states using target network
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Update Q values
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)

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
            'action_size': self.action_size
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
        self.action_size = 8

        # Set up directories
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('models', f'training_{self.timestamp}')
        self.log_dir = os.path.join('logs', self.timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Create agents
        self.agent1 = DQNAgent(self.state_size, self.action_size)
        self.agent2 = DQNAgent(self.state_size, self.action_size)

        # Training metrics
        self.writer = tf.summary.create_file_writer(self.log_dir)

        self.history = {
            'agent1_rewards': [],
            'agent2_rewards': [],
            'agent1_wins': 0,
            'agent2_wins': 0,
            'draws': 0,
            'episodes': 0
        }

    def train(self, episodes: int, save_freq: int = 100):
        progbar = tf.keras.utils.Progbar(
            episodes,
            stateful_metrics=[
                'Player', 'A1_WR', 'A2_WR', 'A1_eps', 'A2_eps',
                'Turns', 'Rounds'
            ]
        )

        for episode in range(episodes):
            states = self.env.reset()
            total_rewards = [0, 0]
            done = False
            turn_count = 0
            rounds_played = 1

            while not done:
                # Get current player
                current_player_idx = self.env.get_current_player()
                current_agent = self.agent1 if current_player_idx == 0 else self.agent2
                current_state = states[current_player_idx]

                # Get action from current agent
                action = current_agent.act(current_state)

                # Execute action
                try:
                    next_states, rewards, done, info = self.env.step(current_player_idx, action)

                    # Store experience in current agent's memory
                    current_agent.remember(
                        current_state,
                        action,
                        rewards[current_player_idx],
                        next_states[current_player_idx],
                        done
                    )

                    # Train current agent
                    current_agent.replay()

                    # Update total rewards
                    total_rewards[0] += rewards[0]
                    total_rewards[1] += rewards[1]

                    # Update states
                    states = next_states
                    turn_count += 1

                    # Track round changes
                    if info.get("round_ended", False):
                        rounds_played += 1

                except ValueError as e:
                    print(f"Error during step: {e}")
                    break

            # Update history
            self.history['episodes'] += 1
            self.history['agent1_rewards'].append(total_rewards[0])
            self.history['agent2_rewards'].append(total_rewards[1])

            if total_rewards[0] > total_rewards[1]:
                self.history['agent1_wins'] += 1
            elif total_rewards[1] > total_rewards[0]:
                self.history['agent2_wins'] += 1
            else:
                self.history['draws'] += 1

            # Log to TensorBoard
            with self.writer.as_default():
                tf.summary.scalar('reward/agent1', total_rewards[0], step=episode)
                tf.summary.scalar('reward/agent2', total_rewards[1], step=episode)
                tf.summary.scalar('training/agent1_epsilon', self.agent1.epsilon, step=episode)
                tf.summary.scalar('training/agent2_epsilon', self.agent2.epsilon, step=episode)
                tf.summary.scalar('game/turns', turn_count, step=episode)
                tf.summary.scalar('game/rounds', rounds_played, step=episode)
                tf.summary.scalar('game/current_player', current_player_idx, step=episode)

            # Save checkpoints
            if episode % save_freq == 0:
                self._save_checkpoint(episode)

            # Update progress bar with metrics
            win_rate_1 = self.history['agent1_wins'] / (episode + 1)
            win_rate_2 = self.history['agent2_wins'] / (episode + 1)

            metrics = [
                ('Player', current_player_idx),
                ('A1_WR', win_rate_1),
                ('A2_WR', win_rate_2),
                ('A1_eps', self.agent1.epsilon),
                ('A2_eps', self.agent2.epsilon),
                ('Turns', turn_count),
                ('Rounds', rounds_played)
            ]

            progbar.update(episode + 1, metrics)

        # Save final models
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
