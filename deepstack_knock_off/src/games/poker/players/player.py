from typing import List

from games.poker.utils.card import Card
from games.poker.actions.action import Action


class Player:

    def __init__(self, name, initial_chips, possible_actions):
        self.name = name
        self.hand: List[Card] = []
        self.chips: int = initial_chips
        self.possible_actions: List[Action] = possible_actions

    def receive_cards(self, *cards: Card):
        self.hand.extend(cards)
