import random
from typing import Union

from core.constants import StatusEffectType, GameConstants, BulletType, StatusEffect
from core.entity import Item


# Specific Item implementations
class Cellphone(Item):
    def __init__(self, reveal_shot: int):
        super().__init__("Cellphone")
        self.reveal_shot = reveal_shot

    def use(self, user, target, game_state) -> bool:
        remaining_shots = len(game_state.bullets) - game_state.current_bullet
        if self.reveal_shot >= remaining_shots:
            return False
        return game_state.bullets[game_state.current_bullet + self.reveal_shot]


class Cigarette(Item):
    def __init__(self):
        super().__init__("Cigarette")

    def use(self, user, target, game_state) -> bool:
        user.heal(GameConstants.CIGARETTE_HEAL_AMOUNT)
        return True


class ShortKnife(Item):
    def __init__(self):
        super().__init__("Short Knife")

    def use(self, user, target, game_state) -> bool:
        target.add_status_effect(
            StatusEffect("damage_multiplier", 1,
                         StatusEffectType.DAMAGE_MULTIPLIER,
                         GameConstants.DAMAGE_MULTIPLIER_VALUE)
        )
        return True


class Switcher(Item):
    def __init__(self):
        super().__init__("Switcher")

    def use(self, user, target, game_state) -> bool:
        target.add_status_effect(
            StatusEffect("bullet_reverse", 1,
                         StatusEffectType.BULLET_REVERSE,
                         GameConstants.BULLET_REVERSE_VALUE)
        )
        return True


class Stealer(Item):
    def __init__(self):
        super().__init__("Stealer")

    def use(self, user, target, game_state) -> bool:
        if not target.items:
            return False
        stolen_item = random.choice(target.items)
        target.items.remove(stolen_item)
        user.items.append(stolen_item)
        return True


class Magnifier(Item):
    def __init__(self):
        super().__init__("Magnifier")

    def use(self, user, target, game_state) -> Union[BulletType, bool]:
        if game_state.current_bullet >= len(game_state.bullets):
            return False
        return game_state.bullets[game_state.current_bullet]


class Stopper(Item):
    def __init__(self):
        super().__init__("Stopper")

    def use(self, user, target, game_state) -> bool:
        existing_effect = target.get_status_effect(StatusEffectType.SKIP_TURN)
        if existing_effect:
            return False
        target.add_status_effect(
            StatusEffect("skip_turn", 1,
                         StatusEffectType.SKIP_TURN,
                         GameConstants.SKIP_TURN_VALUE)
        )
        return True


class PainKiller(Item):
    def __init__(self):
        super().__init__("Pain Killer")

    def use(self, user, target, game_state) -> bool:
        if random.random() < GameConstants.PAINKILLER_HEAL_PROBABILITY:
            user.heal(GameConstants.PAINKILLER_HEAL_AMOUNT)
        else:
            user.take_damage(GameConstants.PAINKILLER_DAMAGE_AMOUNT)
        return True


class Bear(Item):
    def __init__(self):
        super().__init__("Bear")

    def use(self, user, target, game_state) -> bool:
        """
        Bear item effect: Remove current bullet. If all remaining bullets are NO_DAMAGE, set round to end
        :param user:
        :param target:
        :param game_state:
        :return:
        """
        if game_state.current_bullet >= len(game_state.bullets):
            return False
        # Remove current bullet
        game_state.bullets.pop(game_state.current_bullet)
        # Check if remaining bullets are all 0
        all_no_dmg = all(bullet == BulletType.NO_DAMAGE for bullet in game_state.bullets)
        if all_no_dmg:
            game_state.round_number += 1
            game_state.round_end = True
        return True
