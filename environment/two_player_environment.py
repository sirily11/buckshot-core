from typing import Tuple

import numpy as np
from numpy import ndarray

from core import GameCore, BulletType, GameState


class TwoPlayerGameEnvironment:
    def __init__(self):
        self.game = GameCore(num_players=2)
        self.game.initialize_game()

    def get_current_player(self) -> int:
        """Get the index of the current player"""
        return self.game.current_player_idx

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset the environment and return initial states for both players"""
        self.game = GameCore(num_players=2)
        self.game.initialize_game()
        return (self._get_state_representation(0), self._get_state_representation(1))

    def _get_state_representation(self, player_idx: int) -> np.ndarray:
        """Convert game state to neural network input from a player's perspective"""
        game_state = self.game.get_game_state(player_idx)
        current_player = self.game.players[player_idx]
        opponent_idx = 1 if player_idx == 0 else 0
        opponent = self.game.players[opponent_idx]

        state = [
            # Player stats
            current_player.current_hp / current_player.max_hp,
            opponent.current_hp / opponent.max_hp,

            # Round info
            self.game.round_info.total_bullets,
            self.game.round_info.damage_bullets / self.game.round_info.total_bullets,
            self.game.round_info.current_bullet / self.game.round_info.total_bullets,

            # Known bullet info
            1.0 if current_player.known_current_bullet == BulletType.DAMAGE else 0.0
            if current_player.known_current_bullet == BulletType.NO_DAMAGE else -1.0,

            # Item counts
            len(current_player.items) / 3,
            len(opponent.items) / 3,
        ]

        # One-hot encode available items
        item_types = ['Cigarette', 'ShortKnife', 'Switcher', 'Stealer',
                      'Magnifier', 'Stopper', 'PainKiller', 'Bear']
        for item_name in item_types:
            state.append(1.0 if any(item.name == item_name for item in current_player.items) else 0.0)

        return np.array(state, dtype=np.float32)

    def step(self, player_idx: int, action: int) -> tuple[
        tuple[ndarray, ndarray], tuple[int, ...], bool, dict[str, bool]]:
        """Execute action for given player and return states, rewards for both players"""
        if player_idx != self.game.current_player_idx:
            raise ValueError(f"Not player {player_idx}'s turn! Current player: {self.game.current_player_idx}")

        current_player = self.game.players[player_idx]
        opponent_idx = 1 if player_idx == 0 else 0

        # Handle action
        num_items = len(current_player.items)
        total_actions = num_items * 2 + 2

        if action >= total_actions:
            return (self._get_state_representation(0), self._get_state_representation(1)), (-1, 0), False, {
                "invalid_action": True}

        # Execute action

        action_rewards = [0, 0]
        if action < num_items * 2:  # Use item
            item_idx = action // 2

            target_idx = player_idx if action % 2 == 0 else opponent_idx

            if item_idx < len(current_player.items):

                result = self.game.use_item(item_idx, target_idx)

                if not result.success:
                    return (self._get_state_representation(0), self._get_state_representation(1)), (-1, 0), False, {
                        "invalid_action": True}
            else:
                return (self._get_state_representation(0), self._get_state_representation(1)), (-1, 0), False, {
                    "invalid_action": True}
        else:  # Shoot
            target_idx = player_idx if action == num_items * 2 else opponent_idx

            success, message = self.game.shoot(target_idx)
            if not success:
                return (self._get_state_representation(0), self._get_state_representation(1)), (-1, 0), False, {
                    "invalid_action": True}

        # Check if round ended

        round_ended = self.game.check_round_end()

        if round_ended:
            # Calculate round end rewards
            if not self.game.players[opponent_idx].is_alive:
                action_rewards[player_idx] = 10  # Current player won
                action_rewards[opponent_idx] = -10  # Opponent lost
            elif not current_player.is_alive:
                action_rewards[player_idx] = -10  # Current player lost
                action_rewards[opponent_idx] = 10  # Opponent won

            # Try to start new round

            game_continues = self.game.start_new_round()
            if not game_continues:
                # Game is over (reached max rounds)
                return (self._get_state_representation(0), self._get_state_representation(1)), \
                    tuple(action_rewards), True, {"game_over": True}

        # Add health-based rewards
        for idx, player in enumerate(self.game.players):
            if player.current_hp < player.max_hp:
                action_rewards[idx] -= 1

        done = self.game.game_state == GameState.GAME_END
        next_states = (self._get_state_representation(0), self._get_state_representation(1))

        return next_states, tuple(action_rewards), done, {"round_ended": round_ended}
