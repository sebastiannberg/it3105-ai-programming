from typing import List

from games.poker.utils.card import Card


class Player:

    def __init__(self, name, initial_chips):
        self.name = name
        self.hand: List[Card] = []
        self.chips: int = initial_chips

    def receive_cards(self, *cards: Card):
        self.hand.extend(cards)
