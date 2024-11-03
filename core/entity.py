from abc import ABC, abstractmethod
from typing import Union

from core.constants import RoundInfo, BulletType
from core.player import Player


class Item(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def use(self, user: Player, target: Player, game_state: RoundInfo) -> Union[bool, BulletType]:
        """
        Abstract method to use the item
        Returns True if item was successfully used
        """
        pass
