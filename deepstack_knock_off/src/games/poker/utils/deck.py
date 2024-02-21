import random


class Deck:
    def __init__(self, cards):
        self.cards = cards

    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)

    def deal(self, num_cards=1):
        """
        Deal cards from the deck.

        Args:
            num_cards (int): Number of cards to deal. Default is 1.

        Returns:
            list: List of dealt cards.
        """
        dealt_cards = []
        for _ in range(num_cards):
            if self.cards:
                dealt_cards.append(self.cards.pop())
            else:
                print("Deck is empty. Cannot deal more cards.")
                break
        return dealt_cards
