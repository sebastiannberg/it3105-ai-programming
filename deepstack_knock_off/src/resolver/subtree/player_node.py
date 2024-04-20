from typing import Optional, Dict, List
import numpy as np

from games.poker.poker_oracle import PokerOracle
from games.poker.poker_state import PokerState
from resolver.subtree.node import Node


class PlayerNode(Node):

    def __init__(self, state: PokerState, player: str, parent: Optional[Node] = None, stage_depth: Optional[int] = None):
        super().__init__(state, parent, stage_depth)
        self.player = player
        self.strategy_matrix: np.ndarray = None
        self.hand_label_to_index: Dict[str, int] = None
        self.action_to_index: Dict[str, int] = None
        self.index_to_action: List[str] = None

    def init_strategy_matrix(self, deck_size):
        # Assuming self.children is available and populated correctly with action tuples
        if not self.children:
            raise ValueError("No children actions found for initializing strategy matrix.")

        possible_hands, hand_label_to_index, _ = PokerOracle.get_possible_hands_with_indexing(deck_size=deck_size)
        all_actions = [child[1] for child in self.children]

        # Set hand label to index mapping
        self.hand_label_to_index = hand_label_to_index

        # Set action to index mappings
        self.action_to_index = {action: idx for idx, action in enumerate(all_actions)}
        self.index_to_action = all_actions

        if len(all_actions) == 0:
            raise ValueError("No legal actions available to initialize strategy matrix.")

        # Initialize strategy matrix to uniform probability distribution
        self.strategy_matrix = np.full((len(possible_hands), len(all_actions)), 1/len(all_actions))

    def get_action_probability(self, hand_label: str, action: str) -> float:
        hand_index = self.hand_label_to_index[hand_label]
        action_index = self.action_to_index[action]
        if action_index is None:
            raise ValueError(f"Action '{action}' not recognized.")
        return self.strategy_matrix[hand_index, action_index]
