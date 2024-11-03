import random
from collections import deque
from typing import List, Optional, Dict, Tuple

from core.constants import BulletType, GameConstants, StatusEffectType, RoundInfo, GameState, UseItemResult, \
    RevealedFutureBullet
from core.items import Cigarette, ShortKnife, Switcher, Stealer, Magnifier, Stopper, PainKiller, Bear, Cellphone
from core.player import Player


class GameCore:
    def __init__(self, num_players: int):
        if not GameConstants.MIN_PLAYERS <= num_players <= GameConstants.MAX_PLAYERS:
            raise ValueError(
                f"Number of players must be between {GameConstants.MIN_PLAYERS} and {GameConstants.MAX_PLAYERS}")

        self.num_players = num_players
        self.current_round = 0
        self.max_rounds = 3
        self.players: List[Player] = []
        self.round_info: Optional[RoundInfo] = None
        self.current_player_idx = 0
        self.game_state = GameState.SETUP
        self.turn_queue = deque()

    def initialize_game(self):
        """Initialize the game by creating players and starting first round"""
        # Initialize players with random HP
        starting_hp = random.randint(GameConstants.MIN_START_HP, GameConstants.MAX_START_HP)
        self.players = [Player(i, starting_hp) for i in range(self.num_players)]

        # Start first round
        self.start_new_round()

    def start_new_round(self) -> bool:
        """Start a new round, return False if game should end"""
        if self.current_round >= self.max_rounds:
            self.game_state = GameState.GAME_END
            return False

        self.current_round += 1
        self._setup_round()
        self.game_state = GameState.PLAYING
        return True

    def _setup_round(self):
        """Setup a new round with bullets and items"""
        # Generate random bullets
        total_bullets = random.randint(GameConstants.MIN_BULLETS, GameConstants.MAX_BULLETS)
        damage_bullets = random.randint(1, total_bullets - 1)
        no_damage_bullets = total_bullets - damage_bullets

        # Create bullet sequence
        bullets = ([BulletType.DAMAGE] * damage_bullets +
                   [BulletType.NO_DAMAGE] * no_damage_bullets)
        random.shuffle(bullets)

        self.round_info = RoundInfo(
            round_number=self.current_round,
            total_bullets=total_bullets,
            damage_bullets=damage_bullets,
            no_damage_bullets=no_damage_bullets,
            bullets=bullets,
            current_bullet=0
        )

        # Distribute items
        self._distribute_items()

        # Setup turn order
        self.turn_queue = deque(range(self.num_players))
        self.current_player_idx = self.turn_queue[0]

    def _distribute_items(self):
        """Distribute random items to players"""
        available_items = [
            Cigarette, ShortKnife, Switcher, Stealer,
            Magnifier, Stopper, PainKiller, Bear
        ]

        # Clear previous items
        for player in self.players:
            player.items.clear()

        # Distribute 2-3 random items to each player
        for player in self.players:
            num_items = random.randint(2, 3)
            for _ in range(num_items):
                item_class = random.choice(available_items)
                if item_class == Cellphone:
                    # Cellphone needs special handling for reveal_shot
                    reveal_shot = random.randint(0, self.round_info.total_bullets - 1)
                    item = item_class(reveal_shot)
                else:
                    item = item_class()
                player.items.append(item)

    def get_current_player(self) -> Player:
        """Get the current active player"""
        return self.players[self.current_player_idx]

    def use_item(self, item_idx: int, target_player_idx: int) -> UseItemResult:
        """Use an item from current player's inventory"""
        current_player = self.get_current_player()
        if item_idx >= len(current_player.items):
            return UseItemResult(success=False, message="Invalid item index")

        current_player_skip_effect = current_player.get_status_effect(StatusEffectType.SKIP_TURN)
        if current_player_skip_effect:
            return UseItemResult(success=False, message="Invalid action")

        item = current_player.items[item_idx]
        target = self.players[target_player_idx]

        # Use the item and remove if successful
        use_item_result = item.use(current_player, target, self.round_info)

        # apply item effect
        if type(use_item_result) == BulletType:
            current_player.known_current_bullet = use_item_result

        if type(use_item_result) == RevealedFutureBullet:
            current_player.revealed_future_bullet = use_item_result

        if use_item_result:
            current_player.items.pop(item_idx)
            if type(use_item_result) != bool:
                return UseItemResult(success=True, message=use_item_result)
            return UseItemResult(success=True)
        return UseItemResult(success=False, message="Item usage failed")

    def shoot(self, target_player_idx: int) -> Tuple[bool, str]:
        """
        Execute a shooting action
        Returns: (success, message)
        """
        if self.game_state != GameState.PLAYING:
            return False, "Game is not in playing state"

        if self.round_info.current_bullet >= len(self.round_info.bullets):
            return False, "No bullets remaining"

        current_player = self.get_current_player()
        target_player = self.players[target_player_idx]

        current_player_skip_effect = current_player.get_status_effect(StatusEffectType.SKIP_TURN)
        if current_player_skip_effect:
            return False, "Invalid action"

        # Get current bullet and advance
        bullet = self.round_info.bullets[self.round_info.current_bullet]
        self.round_info.current_bullet += 1

        # Check for bullet reverse effect
        reverse_effect = target_player.get_status_effect(StatusEffectType.BULLET_REVERSE)
        if reverse_effect:
            bullet = (BulletType.NO_DAMAGE if bullet == BulletType.DAMAGE
                      else BulletType.DAMAGE)

        # Apply damage if applicable
        damage = 1 if bullet == BulletType.DAMAGE else 0
        if damage > 0:
            target_player.take_damage(damage)
            if not target_player.is_alive:
                self.game_state = GameState.ROUND_END
                return True, f"Player {target_player_idx} was eliminated!"

        # Handle turn progression
        self._advance_turn(current_player == target_player and damage == 0)

        return True, f"Shot dealt {damage} damage to Player {target_player_idx}"

    def _advance_turn(self, same_player_continues: bool = False):
        """Advance to the next player's turn"""
        if not same_player_continues:
            # Move current player to the end of the queue
            self.turn_queue.rotate(-1)

        # Keep track of players we've checked to avoid infinite loops
        checked_players = set()
        total_players = len(self.players)

        while len(checked_players) < total_players:
            next_player_idx = self.turn_queue[0]

            # If we've already checked this player, all players must be skipped/dead
            if next_player_idx in checked_players:
                # Remove skip effects from all players and start with the first valid player
                for player in self.players:
                    player.status_effects = [effect for effect in player.status_effects
                                             if effect.effect_type != StatusEffectType.SKIP_TURN]
                # Find first alive player
                for idx in range(total_players):
                    if self.players[idx].is_alive:
                        self.current_player_idx = idx
                        self.turn_queue = deque(range(total_players))
                        while self.turn_queue[0] != idx:
                            self.turn_queue.rotate(-1)
                        return
                # If no alive players, game should end
                self.game_state = GameState.ROUND_END
                return

            player = self.players[next_player_idx]
            checked_players.add(next_player_idx)
            player.next_turn()

            # Skip dead players
            if not player.is_alive:
                self.turn_queue.rotate(-1)
                continue

            # Skip players with skip_turn effect
            skip_effect = player.get_status_effect(StatusEffectType.SKIP_TURN)
            if skip_effect:
                self.turn_queue.rotate(-1)
                continue

            # Found a valid player
            self.current_player_idx = next_player_idx
            return

    def check_round_end(self) -> bool:
        """Check if the round should end"""
        if self.game_state == GameState.ROUND_END:
            return True

        # Check if all remaining bullets are 0
        remaining_bullets = self.round_info.bullets[self.round_info.current_bullet:]
        if all(bullet == BulletType.NO_DAMAGE for bullet in remaining_bullets):
            self.game_state = GameState.ROUND_END
            return True

        return False

    def get_round_info(self) -> Dict:
        """Get current round information for players"""
        if not self.round_info:
            return {}

        return {
            "round_number": self.round_info.round_number,
            "total_bullets": self.round_info.total_bullets,
            "damage_bullets": self.round_info.damage_bullets,
            "no_damage_bullets": self.round_info.no_damage_bullets,
            "remaining_bullets": len(self.round_info.bullets) - self.round_info.current_bullet
        }

    def get_game_state(self, player_index: Optional[int] = None) -> Dict:
        """Get current game state information"""
        current_player = self.players[player_index] if player_index is not None else None
        return {
            "current_round": self.current_round,
            "game_state": self.game_state,
            "current_player": self.current_player_idx,
            "players": [
                player.get_info(is_current_player=current_player == player) for player in self.players
            ],
            "round_info": self.get_round_info()
        }
