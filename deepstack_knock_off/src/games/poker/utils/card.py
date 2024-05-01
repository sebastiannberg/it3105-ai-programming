

class Card:

    def __init__(self, rank: str, suit: str):
        self._validate_card(rank, suit)
        self.rank = rank
        self.suit = suit

    def _validate_card(self, rank: str, suit: str):
        valid_ranks = ("A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2")
        valid_suits = ("spades", "hearts", "diamonds", "clubs")
        if rank not in valid_ranks:
            raise ValueError(f"Invalid rank {rank}")
        if suit not in valid_suits:
            raise ValueError(f"Invalid suit {suit}")

    # Easier to read when printing this class for debugging
    def __repr__(self):
        return f"Card('{self.rank}', '{self.suit}')"

    # Easier to read when printing this class for debugging
    def __str__(self):
        return f"{self.rank}{self.suit}"
