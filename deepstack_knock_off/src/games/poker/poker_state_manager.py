from typing import List

from games.poker.players.player import Player
from games.poker.poker_state import PokerState
from games.poker.actions.action import Action
from games.poker.actions.raise_bet import RaiseBet
from games.poker.actions.fold import Fold


class PokerStateManager:

    @staticmethod
    def gen_init_state(players: List[Player], deck, small_blind_amount, big_blind_amount):
        init_state = PokerState(
            game_players=players,
            round_players=players,
            deck=deck,
            stage="preflop",
            public_cards=[],
            pot=0,
            current_bet=0,
            small_blind_amount=small_blind_amount,
            big_blind_amount=big_blind_amount,
            small_blind_player=None,
            big_blind_player=None,
            active_player=None
        )
        return init_state

    @staticmethod
    def find_all_legal_actions(state: PokerState, player: Player) -> List[Action]:
        legal_actions = []
        legal_actions.append(Fold(player))
        return legal_actions

    def _can_check(self):
        return True

    def _can_raise(self):
        return True

    def _can_all_in(self):
        return False


    @staticmethod
    def apply_action(state: PokerState, action: Action):
        """
        Apply a given action, assuming the action is legal.
        This method updates the game state based on the action.
        """
        if isinstance(action, RaiseBet):
            # Update pot and player state
            state.pot += action.chip_cost
            action.player.chips -= action.chip_cost
            action.player.player_bet += action.chip_cost
            # Ensure current_bet reflects the highest current bet
            state.current_bet = max(state.current_bet, action.player.player_bet)
