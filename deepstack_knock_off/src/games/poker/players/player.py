from typing import List, Dict, Optional

from games.poker.utils.card import Card
from games.poker.actions.action import Action


class Player:

    def __init__(self, name: str, initial_chips: int):
        self.name: str = name
        self.hand: List[Card] = []
        self.chips: int = initial_chips
        self.player_bet = 0
        self.has_folded = False
        self.has_checked = False
        self.has_called = False
        self.last_raised = False
        self.has_all_in = False
        self.legal_actions: Optional[List[Action]] = None

    def receive_cards(self, *cards: Card):
        self.hand.extend(cards)

    def fold(self):
        self.has_folded = True

    def check(self):
        self.has_checked = True

    def call(self):
        self.has_called = True

    def poker_raise(self):
        self.last_raised = True

    def all_in(self):
        self.has_all_in = True

    def ready_for_new_round(self):
        self.hand = []
        self.player_bet = 0
        self.has_folded = False
        self.has_checked = False
        self.has_called = False
        self.last_raised = False
        self.has_all_in = False

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "hand": [{"rank": card.rank, "suit": card.suit} for card in self.hand],
            "chips": self.chips,
            "player_bet": self.player_bet,
            "has_folded": self.has_folded,
            "has_checked": self.has_checked,
            "has_called": self.has_called,
            "last_raised": self.last_raised,
            "legal_actions": [action.to_dict() for action in self.legal_actions] if self.legal_actions else None,
        }
