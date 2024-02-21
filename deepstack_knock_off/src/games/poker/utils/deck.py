import random
from typing import List
from utils.card import Card

class Deck:

    def __init__(self, cards: List[Card]):
        self.cards = cards

    def shuffle(self):
        random.shuffle(self.cards)
