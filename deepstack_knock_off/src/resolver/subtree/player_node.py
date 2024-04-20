from typing import Optional, Any

from games.poker.poker_state import PokerState
from resolver.subtree.node import Node


class PlayerNode(Node):

    def __init__(self, state: PokerState, player: str, parent: Optional[Node] = None, stage_depth: Optional[int] = None):
        super().__init__(state, parent, stage_depth)
        self.player = player
        self.strategy = None

    def get_action_probability(self, hand_label: str, action: str):
        return 0.5 # TODO
        # return self.strategy[hand_index, action]
