from games.poker.utils.deck import Deck
from games.poker.utils.card import Card


class PokerOracle:

    def __init__(self):
        pass

    def gen_deck(self, num_cards: int, shuffled: bool):
        if num_cards % 4:
            raise ValueError(f"The number of cards must be divisible evenly by 4 to form a complete deck but was {num_cards}")
        num_ranks = int(num_cards / 4)
        if num_ranks < 5:
            raise ValueError(f"The number of ranks need to be at least 5 but was {num_ranks}")
        ranks = ("A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2")
        suits = ("spades", "hearts", "diamonds", "clubs")
        cards = []
        for rank in ranks[:num_ranks]:
            for suit in suits:
                cards.append(Card(rank, suit))
        deck = Deck(cards)
        if shuffled:
            deck.shuffle()
        return deck

    def gen_utility_matrix(self, cards):
        # It can also generate utility matrices for any collection of 3, 4 or 5 public cards.
        pass

    def classify_poker_hand(self):
        pass

    def compare_poker_hands(self):
        pass
