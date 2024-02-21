from typing import List
from utils.card import Card
from actions.action import Action


class Player:

    def __init__(self, initial_chips, possible_actions):
        self.hand: List[Card] = []
        self.chips: int = initial_chips
        self.possible_actions: List[Action] = possible_actions

    def receive_cards(self, *cards: Card):
        self.hand.extend(cards)
