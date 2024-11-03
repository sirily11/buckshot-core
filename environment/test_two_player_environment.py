import unittest
import numpy as np
from core.constants import BulletType
from environment.two_player_environment import TwoPlayerGameEnvironment


class TestTwoPlayerGameEnvironment(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test case"""
        self.env = TwoPlayerGameEnvironment()

    def test_initialization(self):
        """Test if environment initializes correctly"""
        state_player1, state_player2 = self.env.reset()

        # Check state dimensions
        self.assertEqual(len(state_player1), 16)  # 8 base features + 8 item types
        self.assertEqual(len(state_player2), 16)

        # Check if states are different for players
        self.assertFalse(np.array_equal(state_player1, state_player2))

        # Check if game initialized with correct number of players
        self.assertEqual(len(self.env.game.players), 2)

    def test_state_representation(self):
        """Test if state representation is correctly formatted"""
        state = self.env._get_state_representation(0)

        # Check state components
        self.assertGreaterEqual(state[0], 0)  # Current player HP ratio
        self.assertLessEqual(state[0], 1)
        self.assertGreaterEqual(state[1], 0)  # Opponent HP ratio
        self.assertLessEqual(state[1], 1)
        self.assertGreaterEqual(state[2], 2)  # Total bullets
        self.assertLessEqual(state[2], 6)
        self.assertGreaterEqual(state[3], 0)  # Damage bullets ratio
        self.assertLessEqual(state[3], 1)

    def test_invalid_actions(self):
        """Test handling of invalid actions"""
        initial_states = self.env._get_state_representation(0), self.env._get_state_representation(1)
        current_player = self.env.get_current_player()

        # Test invalid action index
        next_states, rewards, done, info = self.env.step(current_player, 999)
        self.assertTrue(info.get("invalid_action"))
        self.assertEqual(rewards, (-1, 0))
        self.assertFalse(done)

        # Test action by wrong player
        wrong_player = 1 if current_player == 0 else 0
        with self.assertRaises(ValueError):
            self.env.step(wrong_player, 0)

    def test_shooting_mechanics(self):
        """Test shooting actions and their outcomes"""
        current_player = self.env.get_current_player()
        initial_hp = self.env.game.players[1].current_hp

        # Get number of items to calculate shoot action index
        num_items = len(self.env.game.players[current_player].items)
        shoot_opponent_action = num_items * 2 + 1  # Last action is shoot opponent

        # Perform shoot action
        next_states, rewards, done, info = self.env.step(current_player, shoot_opponent_action)

        # Verify turn changed
        self.assertNotEqual(self.env.get_current_player(), current_player)

        # Check if damage was applied (if it was a damage bullet)
        if self.env.game.players[1].current_hp < initial_hp:
            self.assertEqual(initial_hp - self.env.game.players[1].current_hp, 1)

    def test_item_usage(self):
        """Test item usage mechanics"""
        current_player = self.env.get_current_player()
        player = self.env.game.players[current_player]

        # If player has items, test using first item
        if len(player.items) > 0:
            initial_item_count = len(player.items)
            next_states, rewards, done, info = self.env.step(current_player, 0)  # Use first item on self

            # Verify item was consumed
            self.assertEqual(len(player.items), initial_item_count - 1)

    def test_round_ending(self):
        """Test round ending conditions"""
        # Force a player's HP to 1
        current_player = self.env.get_current_player()
        target_player = 1 if current_player == 0 else 0
        self.env.game.players[target_player].current_hp = 1

        # Force a damage bullet
        self.env.game.round_info.bullets = [BulletType.DAMAGE]
        self.env.game.round_info.current_bullet = 0

        # Shoot the low HP player
        num_items = len(self.env.game.players[current_player].items)
        shoot_opponent_action = num_items * 2 + 1

        next_states, rewards, done, info = self.env.step(current_player, shoot_opponent_action)

        # Verify round ended
        self.assertTrue(info.get("round_ended"))
        self.assertEqual(rewards[current_player], 10)  # Winner reward
        self.assertEqual(rewards[target_player], -11)  # Loser penalty

    def test_game_ending(self):
        """Test game ending after max rounds"""
        self.env.game.current_round = 2  # Set to last round
        self.env.game.max_rounds = 2
        current_player = self.env.get_current_player()

        # Force round end
        self.env.game.players[1].current_hp = 1
        self.env.game.round_info.bullets = [BulletType.DAMAGE]
        self.env.game.round_info.current_bullet = 0

        # Execute final action
        num_items = len(self.env.game.players[current_player].items)
        shoot_opponent_action = num_items * 2 + 1

        next_states, rewards, done, info = self.env.step(current_player, shoot_opponent_action)

        # Verify game ended
        self.assertTrue(done)
        self.assertTrue(info.get("game_over"))