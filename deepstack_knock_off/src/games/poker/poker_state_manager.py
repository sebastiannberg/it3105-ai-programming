from __future__ import annotations
from typing import List, Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from games.poker.poker_game_manager import PokerGameManager

from games.poker.players.player import Player
from games.poker.poker_state import PokerState
from games.poker.actions.action import Action
from games.poker.actions.raise_bet import RaiseBet
from games.poker.actions.fold import Fold
from games.poker.actions.check import Check


class PokerStateManager:

    @staticmethod
    def gen_init_state(players: List[Player], deck, small_blind_amount, big_blind_amount):
        init_state = PokerState(
            game_players=players,
            round_players=players.copy(),
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
    def find_all_legal_actions(state: PokerState, player: Player, rules: Dict) -> List[Action]:
        legal_actions = []

        # Player can always fold
        legal_actions.append(Fold(player))

        check = PokerStateManager._check(state, player, rules)
        if check:
            legal_actions.append(check)

        call = PokerStateManager._call(state, player, rules)
        if call:
            legal_actions.append(call)

        poker_raise = PokerStateManager._raise(state, player, rules)
        if poker_raise:
            legal_actions.append(poker_raise)

        all_in = PokerStateManager._all_in(state, player, rules)
        if all_in:
            legal_actions.append(all_in)

        return legal_actions

    @staticmethod
    def _check(state: PokerState, player: Player, rules: Dict):
        if state.current_bet == player.player_bet:
            return Check(player=player)
        else:
            return None

    @staticmethod
    def _call(state: PokerState, player: Player, rules: Dict):
        if player.player_bet < state.current_bet and player.chips >= (state.current_bet - player.player_bet):
            return RaiseBet(player, chip_cost=(state.current_bet - player.player_bet), raise_amount=0, raise_type="call")
        else:
            return None

    @staticmethod
    def _raise(state: PokerState, player: Player, rules: Dict):
        if player.chips >= rules["fixed_raise"] + (state.current_bet - player.player_bet):
            return RaiseBet(player, chip_cost=(state.current_bet -player.player_bet + rules["fixed_raise"]), raise_amount=rules["fixed_raise"], raise_type="raise")
        else:
            return None

    @staticmethod
    def _all_in(state: PokerState, player: Player, rules: Dict):
        if rules["all_in_disabled"]:
            return None
        else:
            if player.chips - state.current_bet < 0:
                # All in without calling
                raise NotImplementedError("Not implemented")
            else:
                return RaiseBet(player, chip_cost=(player.chips), raise_amount=(player.chips - state.current_bet), raise_type="all_in")

    @staticmethod
    def apply_action(game_manager: PokerGameManager, action: Action):
        """
        Apply a given action, assuming the action is legal.
        This method updates the game state based on the action.
        """
        player = game_manager.find_round_player_by_name(action.player.name)

        if isinstance(action, Fold):
            game_manager.game.round_players.remove(player)

        if isinstance(action, Check):
            player.check()
            if all(p.has_checked for p in game_manager.game.round_players):
                game_manager.proceed()
            else:
                game_manager.assign_active_player()

        if isinstance(action, RaiseBet):
            player.chips -= action.chip_cost
            game_manager.game.current_bet += action.raise_amount
            game_manager.game.pot += action.chip_cost
            player.player_bet = game_manager.game.current_bet
            if action.raise_type == "small_blind" or action.raise_type == "big_blind":
                return
            game_manager.assign_active_player()
