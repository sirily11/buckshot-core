import unittest

from core.core import GameCore
from core.constants import StatusEffectType, BulletType, GameConstants
from core.items import StatusEffect, Cigarette, ShortKnife, Switcher, Magnifier, Bear, Stealer


class TestItems(unittest.TestCase):
    def setUp(self):
        self.game = GameCore(3)
        self.game.initialize_game()

    def test_cigarette(self):
        """Test Cigarette healing"""
        player = self.game.players[0]
        initial_hp = player.current_hp
        player.take_damage(1)

        cigarette = Cigarette()
        cigarette.use(player, player, self.game.round_info)

        self.assertEqual(player.current_hp, min(initial_hp, player.current_hp + GameConstants.CIGARETTE_HEAL_AMOUNT))

    def test_short_knife(self):
        """Test Short Knife damage multiplier"""
        target = self.game.players[1]
        initial_hp = target.current_hp

        # Apply Short Knife effect
        knife = ShortKnife()
        knife.use(self.game.players[0], target, self.game.round_info)

        # Force damage bullet and shoot
        self.game.round_info.bullets[0] = BulletType.DAMAGE
        self.game.shoot(target_player_idx=1)

        self.assertEqual(target.current_hp, initial_hp - 2)

    def test_switcher(self):
        """Test Switcher bullet reversal"""
        target = self.game.players[1]
        initial_hp = target.current_hp

        # Apply Switcher effect
        switcher = Switcher()
        switcher.use(self.game.players[0], target, self.game.round_info)

        # Force damage bullet and shoot
        self.game.round_info.bullets[0] = BulletType.DAMAGE
        self.game.shoot(target_player_idx=1)

        # Damage should be reversed
        self.assertEqual(target.current_hp, initial_hp)

    def test_stealer(self):
        """Test Stealer item theft"""
        player1 = self.game.players[0]
        player2 = self.game.players[1]

        # Give player2 a specific item
        test_item = Cigarette()
        player2.items = [test_item]

        stealer = Stealer()
        initial_items_count = len(player1.items)

        success = stealer.use(player1, player2, self.game.round_info)

        self.assertTrue(success)
        self.assertEqual(len(player2.items), 0)
        self.assertEqual(len(player1.items), initial_items_count + 1)

    def test_magnifier(self):
        """Test Magnifier bullet revelation"""
        self.game.round_info.bullets[0] = BulletType.DAMAGE

        magnifier = Magnifier()
        result = magnifier.use(self.game.players[0], self.game.players[0], self.game.round_info)

        self.assertEqual(result, BulletType.DAMAGE)

    def test_bear(self):
        """Test Bear bullet removal"""
        # Setup bullets with known sequence
        self.game.round_info.bullets = [BulletType.DAMAGE, BulletType.NO_DAMAGE, BulletType.NO_DAMAGE]
        initial_bullets = len(self.game.round_info.bullets)

        bear = Bear()
        result = bear.use(self.game.players[0], self.game.players[0], self.game.round_info)

        self.assertEqual(len(self.game.round_info.bullets), initial_bullets - 1)
        self.assertTrue(result)


class TestStatusEffects(unittest.TestCase):
    def setUp(self):
        self.game = GameCore(3)
        self.game.initialize_game()

    def test_status_effect_duration(self):
        """Test status effect duration and removal"""
        player = self.game.players[0]
        effect = StatusEffect("test", 2, StatusEffectType.DAMAGE_MULTIPLIER)
        player.add_status_effect(effect)

        self.assertEqual(len(player.status_effects), 1)

        # Update twice to expire effect
        player.update_status_effects()
        self.assertEqual(len(player.status_effects), 1)
        player.update_status_effects()
        self.assertEqual(len(player.status_effects), 0)

    def test_status_effect_override(self):
        """Test that adding same type effect overrides previous"""
        player = self.game.players[0]
        effect1 = StatusEffect("test1", 2, StatusEffectType.DAMAGE_MULTIPLIER, 2.0)
        effect2 = StatusEffect("test2", 1, StatusEffectType.DAMAGE_MULTIPLIER, 3.0)

        player.add_status_effect(effect1)
        player.add_status_effect(effect2)

        self.assertEqual(len(player.status_effects), 1)
        self.assertEqual(player.status_effects[0].value, 3.0)