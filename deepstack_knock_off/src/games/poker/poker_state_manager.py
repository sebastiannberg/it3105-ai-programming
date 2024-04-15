from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from games.poker.poker_state import PokerState
    from games.poker.poker_game import PokerGame
    from games.poker.actions.action import Action
    from games.poker.players.player import Player


class PokerStateManager:

    def __init__(self, poker_rules: Dict):
        self.poker_rules = poker_rules

    def gen_state_from_game(self, poker_game: PokerGame, player_one_perspective: Player) -> PokerState:
        return PokerState(
            public_cards=poker_game.public_cards,
            chips_player_one=player_one_perspective.chips,
            chips_player_two=[player for player in poker_game.game_players if player is not player_one_perspective][0],
            pot=poker_game.pot,
            current_bet=poker_game.current_bet,
            stage=poker_game.stage
        )

    def gen_child_state(self, parent_state: PokerState, state_type: str, action: Action) -> PokerState:
        pass

    def gen_child_states(self, parent_state: PokerState, state_type: str) -> List[PokerState]:
        pass
