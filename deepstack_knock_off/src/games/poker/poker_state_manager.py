from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from games.poker.poker_state import PokerState
    from games.poker.poker_game import PokerGame


class PokerStateManager:

    def __init__(self, poker_rules: Dict):
        self.poker_rules = poker_rules

    def gen_state_from_game(self, poker_game: PokerGame) -> PokerState:
        pass

    def gen_legal_child_states(parent_state: PokerState) -> List[PokerState]:
        pass
