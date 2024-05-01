from __future__ import annotations
from typing import List, Tuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from games.poker.players.player import Player
    from games.poker.utils.deck import Deck
    from games.poker.utils.card import Card
    from games.poker.players.player import Player
    from games.poker.actions.action import Action

from dataclasses import dataclass


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
    history: List[Tuple[str, Action]] # (stage, Action)
    pot: int
    current_bet: int
