from dataclasses import dataclass
from typing import List, Union

from games.poker.players.player import Player
from games.poker.utils.deck import Deck
from games.poker.utils.card import Card


@dataclass
class PokerState:
    game_players: List[Player]
    round_players: List[Player]
    deck: Deck
    stage: str
    public_cards: List[Card]
    pot: int
    current_bet: int
    small_blind_amount: int
    big_blind_amount: int
    small_blind_player: Player
    big_blind_player: Player
    active_player: Player
