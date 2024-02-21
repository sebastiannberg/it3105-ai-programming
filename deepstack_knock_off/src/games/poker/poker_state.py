from dataclasses import dataclass
from typing import List, Union

from games.poker.players.ai_player import AIPlayer
from games.poker.players.human_player import HumanPlayer
from games.poker.utils.deck import Deck
from games.poker.utils.card import Card


@dataclass
class PokerState:
    game_players: List[Union[AIPlayer, HumanPlayer]]
    round_players: List[Union[AIPlayer, HumanPlayer]]
    deck: Deck
    stage: str
    public_cards: List[Card]
    pot: int
    small_blind_amount: int
    big_blind_amount: int
    small_blind_player: Union[AIPlayer, HumanPlayer]
    big_blind_player: Union[AIPlayer, HumanPlayer]
    active_player: Union[AIPlayer, HumanPlayer]
