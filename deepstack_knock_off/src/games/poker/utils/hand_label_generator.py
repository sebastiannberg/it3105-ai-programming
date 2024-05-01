from typing import List

from games.poker.utils.card import Card


class HandLabelGenerator:

    @staticmethod
    def get_hand_label(cards: List[Card]) -> str:
        # Raise error if not all elements in the cards List is of instance Card class
        if not all(isinstance(card, Card) for card in cards):
            raise ValueError("All items in the list must be instances of Card")
        # Sort the hand based on rank and then suit
        sorted_hand = tuple(sorted(cards, key=lambda card: (card.rank, card.suit)))
        # Return a string that represents this hand of cards
        return "".join([f"{card.rank}{card.suit}" for card in sorted_hand])
