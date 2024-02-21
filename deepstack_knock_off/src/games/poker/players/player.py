from typing import List

from games.poker.utils.card import Card


class Player:

    def __init__(self, name, initial_chips):
        self.name: str = name
        self.hand: List[Card] = []
        self.chips: int = initial_chips
        self.pot_contribution = 0

    def receive_cards(self, *cards: Card):
        self.hand.extend(cards)
