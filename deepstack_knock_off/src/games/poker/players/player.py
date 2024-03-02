from typing import List

from games.poker.utils.card import Card


class Player:

    def __init__(self, name, initial_chips):
        self.name: str = name
        self.hand: List[Card] = []
        self.chips: int = initial_chips
        self.player_bet = 0
        self.has_folded = False
        self.has_checked = False
        self.has_called = False
        self.last_raised = False

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

    def ready_for_new_round(self):
        self.hand = []
        self.player_bet = 0
        self.has_folded = False
        self.has_checked = False
        self.has_called = False
        self.last_raised = False
