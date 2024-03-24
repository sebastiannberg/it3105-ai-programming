from dataclasses import dataclass
from typing import List

from games.poker.players.player import Player


@dataclass
class PokerState:
    game_players: List[Player]
    round_players: List[Player]
