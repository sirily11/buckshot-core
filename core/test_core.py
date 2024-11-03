import random
import unittest

from core import Stopper
from core.constants import BulletType, StatusEffectType, GameConstants
from core.core import GameState, GameCore
from core.items import StatusEffect, Magnifier


class TestGameCore(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.game = GameCore(num_players=2)
        random.seed(42)  # Set seed for reproducible tests

    def test_game_initialization(self):
        """Test game initialization"""
        self.game.initialize_game()

        # Check basic game state
        self.assertEqual(self.game.num_players, 2)
        self.assertEqual(self.game.current_round, 1)
        self.assertEqual(len(self.game.players), 2)
        self.assertEqual(self.game.game_state, GameState.PLAYING)

        # Check players initialization
        for player in self.game.players:
            self.assertTrue(GameConstants.MIN_START_HP <= player.current_hp <= GameConstants.MAX_START_HP)
            self.assertTrue(player.is_alive)
            self.assertTrue(2 <= len(player.items) <= 3)

    def test_invalid_player_count(self):
        """Test initialization with invalid player count"""
        with self.assertRaises(ValueError):
            GameCore(1)
        with self.assertRaises(ValueError):
            GameCore(5)

    def test_round_setup(self):
        """Test round setup and bullet distribution"""
        self.game.initialize_game()
        round_info = self.game.round_info

        # Check bullet distribution
        self.assertTrue(2 <= round_info.total_bullets <= 10)
        self.assertEqual(len(round_info.bullets), round_info.total_bullets)
        self.assertEqual(round_info.damage_bullets + round_info.no_damage_bullets,
                         round_info.total_bullets)

    def test_shooting_mechanics(self):
        """Test shooting mechanics and damage calculation"""
        self.game.initialize_game()
        initial_hp = self.game.players[1].current_hp

        # Force a damage bullet
        self.game.round_info.bullets[0] = BulletType.DAMAGE
        success, message = self.game.shoot(target_player_idx=1)

        self.assertTrue(success)
        self.assertEqual(self.game.players[1].current_hp, initial_hp - 1)

    def test_player_death(self):
        """Test player death and round end condition"""
        self.game.initialize_game()
        target_player = self.game.players[1]
        target_player.current_hp = 1

        # Force a damage bullet
        self.game.round_info.bullets[0] = BulletType.DAMAGE
        success, message = self.game.shoot(target_player_idx=1)

        self.assertTrue(success)
        self.assertFalse(target_player.is_alive)
        self.assertEqual(self.game.game_state, GameState.ROUND_END)

    def test_turn_progression(self):
        """Test turn progression and skip effects"""
        self.game.initialize_game()
        initial_player = self.game.current_player_idx

        # Normal turn progression
        self.game.shoot(target_player_idx=1)
        self.assertNotEqual(self.game.current_player_idx, initial_player)

        # Skip turn effect
        current_player = self.game.get_current_player()
        current_player.add_status_effect(
            StatusEffect("skip_turn", 1, StatusEffectType.SKIP_TURN)
        )
        result = self.game.shoot(target_player_idx=1)
        self.assertFalse(result[0])

    def test_magnifier(self):
        """Test Magnifier bullet revelation"""
        self.game.initialize_game()
        self.game.round_info.bullets[0] = BulletType.DAMAGE
        # Setup Magnifier item
        self.game.players[0].items = [Magnifier()]
        info = self.game.get_game_state(player_index=0)
        self.assertEqual(info["players"][0]["items"], ["Magnifier"])
        info = self.game.get_game_state(player_index=1)
        self.assertEqual(info["players"][0]["items"], ["Magnifier"])
        result = self.game.use_item(item_idx=0, target_player_idx=0)
        self.assertTrue(result.success)
        self.assertEqual(result.message, BulletType.DAMAGE)
        info = self.game.get_game_state(player_index=0)
        self.assertEqual(info["players"][0]["items"], [])
        self.assertTrue("known_current_bullet" in info["players"][0])
        self.assertTrue("revealed_future_bullet" in info["players"][0])
        info = self.game.get_game_state(player_index=1)
        # other player should not know the bullet
        self.assertEqual(info["players"][0]["items"], [])
        self.assertFalse("known_current_bullet" in info["players"][0])
        self.assertFalse("revealed_future_bullet" in info["players"][0])

    def test_skip_effect(self):
        """Test skip turn effect"""
        self.game.initialize_game()
        player_1 = self.game.players[0]
        player_2 = self.game.players[1]

        player_1.add_status_effect(StatusEffect("skip_turn", 1, StatusEffectType.SKIP_TURN))
        player_2.add_status_effect(StatusEffect("skip_turn", 1, StatusEffectType.SKIP_TURN))
        self.game.shoot(target_player_idx=0)
        self.assertEqual(self.game.current_player_idx, 0)

    def test_should_not_apply_skip_effect_twice(self):
        self.game.initialize_game()
        player_1 = self.game.players[0]
        player_2 = self.game.players[1]

        player_1.items = [Stopper()]
        player_2.items = [Stopper()]

        # player 1 uses stopper on player 2
        self.game.use_item(0, 1)
        self.assertTrue(player_2.get_status_effect(StatusEffectType.SKIP_TURN))

        # player 2 should not be able to use stopper on player 1
        self.game.current_player_idx = 1
        result = self.game.shoot(0)
        self.assertFalse(result[0])
        self.assertEqual(result[1], "Invalid action")

        result = self.game.use_item(0, 0)
        self.assertFalse(result.success)


