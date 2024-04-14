from typing import Optional, List

from games.poker.poker_state import PokerState


class Node:

    def __init__(self, state: PokerState):
        self.state: PokerState = state
        self.parent: Optional['Node'] = None

    def set_parent(self, parent: 'Node'):
        self.parent = parent
