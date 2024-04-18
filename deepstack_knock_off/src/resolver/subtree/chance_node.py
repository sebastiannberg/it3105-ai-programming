from typing import Optional, Any

from games.poker.poker_state import PokerState
from resolver.subtree.node import Node


class ChanceNode(Node):

    def __init__(self, state: PokerState, parent: Optional[Node] = None):
        super().__init__(state, parent)
