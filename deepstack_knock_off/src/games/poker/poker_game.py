from dataclasses import dataclass
from typing import List, Optional

from games.poker.players.player import Player
from games.poker.utils.deck import Deck
from games.poker.utils.card import Card


@dataclass
class PokerGame:
    game_players: List[Player]
    round_players: List[Player]
    small_blind_player: Player
    big_blind_player: Player
    current_player: Player
    deck: Deck
    public_cards: List[Card]
    stage: str
    pot: int
    current_bet: int
    ai_strategy: Optional[str]
