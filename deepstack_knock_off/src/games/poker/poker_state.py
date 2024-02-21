from dataclasses import dataclass
from typing import List, Union
from players.ai_player import AIPlayer
from players.human_player import HumanPlayer
from utils.deck import Deck
from utils.card import Card


@dataclass
class PokerState:

    def __init__(self):
        self.players: List[Union[AIPlayer, HumanPlayer]]
        self.deck: Deck
        self.stage: str
        self.public_cards: List[Card]
        self.pot: int
        self.small_blind_amount: int
        self.big_blind_amount: int
        self.dealer: Union[AIPlayer, HumanPlayer]
        self.small_blind_player: Union[AIPlayer, HumanPlayer]
        self.big_blind_player: Union[AIPlayer, HumanPlayer]
