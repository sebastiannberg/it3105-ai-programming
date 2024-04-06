from __future__ import annotations
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from games.poker.players.player import Player
    from games.poker.poker_game import PokerGame


class Action:

    def __init__(self, name: str, player: Player):
        self.name = name
        self.player = player

    def apply(self, game: PokerGame) -> Dict:
        # Override in subclasses as needed
        pass

    def to_dict(self):
        return {
            'action_name': self.name,
            'player_name': self.player.name
        }
