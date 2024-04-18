from typing import Optional, Any

from games.poker.poker_state import PokerState
from resolver.subtree.node import Node


class PlayerNode(Node):

    def __init__(self, state: PokerState, player: str, parent: Optional[Node] = None):
        super().__init__(state, parent)
        self.player = player
        self.strategy = None
