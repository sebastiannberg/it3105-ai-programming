from games.poker.poker_state import PokerState


class PokerGame:

    def __init__(self, state: PokerState):
        self.state = state

    def update_state(self, state: PokerState):
        self.state = state
