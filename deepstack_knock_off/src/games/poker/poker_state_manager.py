from __future__ import annotations
from typing import List, Dict, Optional, TYPE_CHECKING
import random
import itertools
from copy import deepcopy

from games.poker.poker_oracle import PokerOracle
from games.poker.poker_state import PokerState

if TYPE_CHECKING:
    from games.poker.poker_game import PokerGame
    from games.poker.players.player import Player
    from games.poker.actions.fold import Fold
    from games.poker.actions.check import Check
    from games.poker.actions.raise_bet import RaiseBet


class PokerStateManager:

    def __init__(self, poker_rules: Dict):
        self.poker_rules = poker_rules
        self.oracle: PokerOracle = PokerOracle()

    def gen_state_from_game(self, poker_game: PokerGame, player_one_perspective: Player) -> PokerState:
        state_history = []
        for entry in poker_game.history:
            action = entry[1]
            if action.player == player_one_perspective:
                player = "player_one"
            else:
                player = "player_two"
            if isinstance(action, Fold):
                state_history.append((player, "fold"))
            elif isinstance(action, Check):
                state_history.append((player, "check"))
            elif isinstance(action, RaiseBet):
                if action.raise_type == "call":
                    state_history.append((player, "call"))
                elif action.raise_type == "raise":
                    state_history.append((player, f"raise {action.raise_amount}"))
                elif action.raise_type == "small_blind":
                    state_history.append((player, "small blind"))
                elif action.raise_type == "big_blind":
                    state_history.append((player, "big blind"))
        return PokerState(
            public_cards=poker_game.public_cards,
            player_one_chips=player_one_perspective.chips,
            player_one_bet=player_one_perspective.player_bet,
            player_two_chips=[player for player in poker_game.game_players if player is not player_one_perspective][0].chips,
            player_two_bet=[player for player in poker_game.game_players if player is not player_one_perspective][0].player_bet,
            pot=poker_game.pot,
            stage=poker_game.stage,
            history=state_history
        )

    def find_legal_actions(self, parent_state: PokerState, player: str) -> List[PokerState]:
        if player not in ["player_one", "player_two"]:
            raise ValueError(f"player must be either player_one or player_two but was {player}")
        legal_actions = []
        # Fold
        legal_actions.append("fold")
        # Check
        if parent_state.player_one_bet == parent_state.player_two_bet:
            legal_actions.append("check")
        # Call
        if player == "player_one" and parent_state.player_one_bet < parent_state.player_two_bet and parent_state.player_one_chips >= (parent_state.player_two_bet - parent_state.player_one_bet):
            legal_actions.append("call")
        if player == "player_two" and parent_state.player_two_bet < parent_state.player_one_bet and parent_state.player_two_chips >= (parent_state.player_one_bet - parent_state.player_two_bet):
            legal_actions.append("call")
        # Raise
        if player == "player_one" and parent_state.player_one_chips >= self.poker_rules["fixed_raise"] + (parent_state.player_two_bet - parent_state.player_one_bet):
            current_stage_action_history = [entry[2] for entry in parent_state.history if entry[0] == parent_state.stage]
            if len([action for action in current_stage_action_history if "raise" in action]) < self.poker_rules["max_num_raises_per_stage"]:
                legal_actions.append(f"raise {self.poker_rules['fixed_raise']}")
        if player == "player_two" and parent_state.player_two_chips >= self.poker_rules["fixed_raise"] + (parent_state.player_one_bet - parent_state.player_two_bet):
            current_stage_action_history = [entry[2] for entry in parent_state.history if entry[0] == parent_state.stage]
            if len([action for action in current_stage_action_history if "raise" in action]) < self.poker_rules["max_num_raises_per_stage"]:
                legal_actions.append(f"raise {self.poker_rules['fixed_raise']}")
        return legal_actions

    def gen_player_child_state(self, parent_state: PokerState, player: str, action: str):
        stage_change = {
            "preflop": "flop",
            "flop": "turn",
            "turn": "river",
            "river": "showdown"
        }
        child_state = deepcopy(parent_state)
        if "fold" in action:
            if player == "player_one":
                child_state.player_two_chips += parent_state.pot
            if player == "player_two":
                child_state.player_one_chips += parent_state.pot
            child_state.player_one_bet = 0
            child_state.player_two_bet = 0
            child_state.pot = 0
            child_state.history.append((parent_state.stage, player, action))
        elif "check" in action:
            child_state.history.append((parent_state.stage, player, action))
            double_check = "check" in parent_state.history[-1][2] and parent_state.history[-1][0] == parent_state.stage
            if parent_state.stage == "preflop" or double_check:
                # Next stage
                child_state.stage = stage_change[parent_state.stage]
                child_state.player_one_bet = 0
                child_state.player_two_bet = 0
        elif "call" in action:
            child_state.history.append((parent_state.stage, player, action))
            if player == "player_one":
                child_state.player_one_chips -= (parent_state.player_two_bet - parent_state.player_one_bet)
                child_state.player_one_bet = parent_state.player_two_bet
                child_state.pot += (parent_state.player_two_bet - parent_state.player_one_bet)
            if player == "player_two":
                child_state.player_two_chips -= (parent_state.player_one_bet - parent_state.player_two_bet)
                child_state.player_two_bet = parent_state.player_one_bet
                child_state.pot += (parent_state.player_one_bet - parent_state.player_two_bet)
            if parent_state.stage in ["flop", "turn", "river"] or parent_state.stage == "preflop" and len(parent_state.history) != 2:
                # Next stage
                child_state.stage = stage_change[parent_state.stage]
                child_state.player_one_bet = 0
                child_state.player_two_bet = 0
        elif "raise" in action:
            # Assume raise amount is last char in string
            raise_amount = int(action[-1])
            if player == "player_one":
                child_state.player_one_chips -= raise_amount + (parent_state.player_two_bet - parent_state.player_one_bet)
                child_state.player_one_bet = parent_state.player_two_bet + raise_amount
                child_state.pot += raise_amount + (parent_state.player_two_bet - parent_state.player_one_bet)
            if player == "player_two":
                child_state.player_two_chips -= raise_amount + (parent_state.player_one_bet - parent_state.player_two_bet)
                child_state.player_two_bet = parent_state.player_one_bet + raise_amount
                child_state.pot += raise_amount + (parent_state.player_one_bet - parent_state.player_two_bet)
            child_state.history.append((parent_state.stage, player, action))
        return child_state

    def gen_chance_child_states(self, parent_state: PokerState, max_num_children: Optional[int] = None) -> List[PokerState]:
        public_cards_set = set((card.rank, card.suit) for card in parent_state.public_cards)

        deck = self.oracle.gen_deck(num_cards=self.poker_rules["deck_size"], shuffled=True)
        deck = [card for card in deck.cards if (card.rank, card.suit) not in public_cards_set]

        if parent_state.stage == "flop":
            possible_dealings = list(itertools.combinations(deck, 3))
        elif parent_state.stage in ["turn", "river"]:
            possible_dealings = list(itertools.combinations(deck, 1))
        else:
            raise ValueError("Unexpected chance node appearing in wrong stage")

        if max_num_children and len(possible_dealings) >= max_num_children:
            samples = random.sample(possible_dealings, k=max_num_children)
        else:
            samples = possible_dealings

        child_states = []
        for sample in samples:
            new_public_cards = parent_state.public_cards + list(sample)
            child_states.append(PokerState(
                public_cards=new_public_cards,
                player_one_chips=parent_state.player_one_chips,
                player_one_bet=parent_state.player_one_bet,
                player_two_chips=parent_state.player_two_chips,
                player_two_bet=parent_state.player_two_bet,
                pot=parent_state.pot,
                stage=parent_state.stage,
                history=parent_state.history
            ))
        return child_states
