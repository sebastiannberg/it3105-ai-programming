from dataclasses import dataclass
from typing import List, Tuple

from games.poker.utils.card import Card


@dataclass
class PokerState:
    public_cards: List[Card]
    player_one_chips: int
    player_one_bet: int
    player_two_chips: int
    player_two_bet: int
    pot: int
    stage: str
    stage_history: List[Tuple[str, str]] # (player, action)
