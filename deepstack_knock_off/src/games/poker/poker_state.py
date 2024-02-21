

class PokerState:

    def __init__(self, small_blind_amount, big_blind_amount, deck):
        self.public_cards = None
        self.dealer = None
        self.small_blind_amount = small_blind_amount
        self.big_blind_amount = big_blind_amount
        self.pot = self.small_blind_amount + self.big_blind_amount
        self.small_blind_player = None
        self.big_blind_player = None
        self.deck = deck
