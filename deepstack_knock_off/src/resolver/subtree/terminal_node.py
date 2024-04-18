from typing import Optional, Any

from games.poker.poker_state import PokerState
from resolver.subtree.node import Node


class TerminalNode(Node):

    def __init__(self, state: PokerState, parent: Optional[Node] = None):
        super().__init__(state, parent)

    # Override
    def add_child(self, child_node: Node, edge_value: Any):
        raise ValueError("Cannot add children to a terminal node")
