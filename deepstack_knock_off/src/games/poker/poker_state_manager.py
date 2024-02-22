from typing import List

from games.poker.players.player import Player
from games.poker.poker_state import PokerState
from games.poker.actions.action import Action
from games.poker.actions.raise_bet import RaiseBet


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
    def find_all_legal_actions(state: PokerState, player: Player):
        # return a list of all possible actions for the player given the player and the state
        # fold (always possible)
        # raise (always possible if have the money) remember rules and fixed amount
        # check (if possible)
        # rules for all in enabled
        pass

    @staticmethod
    def apply_action(state: PokerState, action: Action):
        """
        Apply a given action, assuming the action is legal.
        This method updates the game state based on the action.
        """
        if isinstance(action, RaiseBet):
            # Update pot and player state
            state.pot += action.raise_amount
            action.player.chips -= action.raise_amount
            action.player.player_bet += action.raise_amount
            # Ensure current_bet reflects the highest current bet
            state.current_bet = max(state.current_bet, action.player.player_bet)
