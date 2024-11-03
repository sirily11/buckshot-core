from typing import List, Optional

from core.constants import StatusEffectType, BulletType, RevealedFutureBullet
from core.items import StatusEffect


class Player:
    def __init__(self, player_id: int, hp: int):
        self.player_id = player_id
        self.max_hp = hp
        self.current_hp = hp
        self.items: List["Item"] = []
        self.status_effects: List[StatusEffect] = []
        self.is_alive = True
        self.known_current_bullet: Optional[BulletType] = None
        self.revealed_future_bullet: Optional[RevealedFutureBullet] = None

    def next_turn(self):
        self.known_current_bullet = None

    def get_status_effect(self, effect_type: StatusEffectType) -> Optional[StatusEffect]:
        """Get a specific status effect if it exists"""
        return next((effect for effect in self.status_effects
                     if effect.effect_type == effect_type), None)

    def take_damage(self, amount: int) -> bool:
        """Apply damage to player and check if they died"""
        # Apply damage multiplier if it exists
        multiplier_effect = self.get_status_effect(StatusEffectType.DAMAGE_MULTIPLIER)
        final_damage = amount * (multiplier_effect.value if multiplier_effect else 1.0)

        self.current_hp = max(0, self.current_hp - int(final_damage))
        if self.current_hp <= 0:
            self.is_alive = False
        return self.is_alive

    def heal(self, amount: int):
        """Heal the player"""
        self.current_hp = min(self.max_hp, self.current_hp + amount)

    def add_status_effect(self, effect: StatusEffect):
        """Add a status effect to the player"""
        # Remove existing effect of same type if it exists
        self.status_effects = [e for e in self.status_effects
                               if e.effect_type != effect.effect_type]
        self.status_effects.append(effect)

    def update_status_effects(self):
        """Update status effects at the end of turn"""
        for effect in self.status_effects:
            effect.duration -= 1

        self.status_effects = [
            effect for effect in self.status_effects
            if effect.duration > 0
        ]

    def get_info(self, is_current_player: bool = False):
        if not is_current_player:
            return {
                "id": self.player_id,
                "is_alive": self.is_alive,
                "hp": self.current_hp,
                "items": [item.name for item in self.items],
                "status_effects": [
                    (effect.name, effect.duration)
                    for effect in self.status_effects
                ]
            }

        return {
            "id": self.player_id,
            "hp": self.current_hp,
            "is_alive": self.is_alive,
            "items": [item.name for item in self.items],
            "status_effects": [
                (effect.name, effect.duration)
                for effect in self.status_effects
            ],
            "known_current_bullet": self.known_current_bullet,
            "revealed_future_bullet": self.revealed_future_bullet
        }
