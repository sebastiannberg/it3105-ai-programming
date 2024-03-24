from typing import Dict

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
            'name': self.name,
            'player': self.player.name
        }
