from typing import Optional, List, Tuple
import numpy as np

from games.poker.poker_state import PokerState


class Node:

    def __init__(self, state: PokerState, parent: Optional['Node'] = None, stage_depth: Optional[int] = None):
        self.state: PokerState = state
        self.parent: Optional['Node'] = parent
        self.stage_depth = stage_depth
        # Each element in children is a tuple (child_node, edge_value)
        self.children: List[Tuple['Node', str]] = []
        self.utility_matrix: Optional[np.ndarray] = None

    def add_child(self, child_node: 'Node', edge_value: str):
        self.children.append((child_node, edge_value))

    def set_utility_matrix(self, matrix: np.ndarray):
        self.utility_matrix = matrix
