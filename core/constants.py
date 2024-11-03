from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class StatusEffectType(Enum):
    DAMAGE_MULTIPLIER = "damage_multiplier"
    BULLET_REVERSE = "bullet_reverse"
    SKIP_TURN = "skip_turn"


class BulletType(Enum):
    NO_DAMAGE = 0
    DAMAGE = 1


class GameConstants:
    # Player Constants
    MIN_PLAYERS = 2
    MAX_PLAYERS = 4
    MIN_START_HP = 3
    MAX_START_HP = 5

    # Status Effect Values
    DAMAGE_MULTIPLIER_VALUE = 2.0
    BULLET_REVERSE_VALUE = 1.0
    SKIP_TURN_VALUE = 1.0

    # Item Constants
    PAINKILLER_HEAL_AMOUNT = 2
    PAINKILLER_DAMAGE_AMOUNT = 1
    CIGARETTE_HEAL_AMOUNT = 1

    # Probability Constants
    PAINKILLER_HEAL_PROBABILITY = 0.5

    # Number of bullets in a round
    MIN_BULLETS = 2
    MAX_BULLETS = 6


class GameState(Enum):
    SETUP = auto()
    PLAYING = auto()
    ROUND_END = auto()
    GAME_END = auto()


@dataclass
class RoundInfo:
    round_number: int
    total_bullets: int
    damage_bullets: int
    no_damage_bullets: int
    bullets: List[BulletType]
    current_bullet: int = 0


@dataclass
class StatusEffect:
    name: str
    duration: int
    effect_type: StatusEffectType
    value: float = 1.0


@dataclass
class UseItemResult:
    success: bool
    message: Optional[str] = None


@dataclass
class RevealedFutureBullet:
    bullet_type: BulletType
    round_number: int