from typing import List, Dict

from games.poker.poker_state import PokerState
from games.poker.poker_game import PokerGame


class PokerStateManager:

    def __init__(self, poker_rules: Dict):
        self.poker_rules = poker_rules

    # TODO
    def gen_state_from_game(self, poker_game: PokerGame) -> PokerState:
        pass

    def gen_legal_child_states() -> List[PokerState]:
        pass
