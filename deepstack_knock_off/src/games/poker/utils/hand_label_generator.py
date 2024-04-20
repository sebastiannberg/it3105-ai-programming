from typing import List, Dict
import itertools

from games.poker.utils.card import Card


class HandLabelGenerator:

    @staticmethod
    def get_hand_label(cards: List[Card]) -> str:
        if not all(isinstance(card, Card) for card in cards):
            raise ValueError("All items in the list must be instances of Card")
        sorted_hand = tuple(sorted(cards, key=lambda card: (card.rank, card.suit)))
        return "".join([f"{card.rank}{card.suit}" for card in sorted_hand])
