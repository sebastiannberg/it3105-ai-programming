import random
from typing import List

from games.poker.utils.card import Card

class Deck:

    def __init__(self, cards: List[Card]):
        self.cards = cards

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_cards(self, num_cards: int) -> List[Card]:
        if num_cards > len(self.cards):
            raise ValueError(f"Not enough cards in the deck to deal {num_cards} cards.")

        dealt_cards = []
        for _ in range(num_cards):
            dealt_cards.append(self.cards.pop())

        return dealt_cards
