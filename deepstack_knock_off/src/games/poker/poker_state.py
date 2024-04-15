from dataclasses import dataclass
from typing import List

from games.poker.utils.card import Card


@dataclass
class PokerState:
    public_cards: List[Card]
    chips_player_one: int
    chips_player_two: int
    pot: int
    current_bet: int
    stage: str
    # history: List[Action] # TODO keep this?
