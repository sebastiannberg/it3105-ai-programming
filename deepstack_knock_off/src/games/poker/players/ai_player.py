from typing import List
import random

from games.poker.players.player import Player
from games.poker.actions.action import Action
from games.poker.utils.card import Card
from games.poker.poker_oracle import PokerOracle
from games.poker.actions.fold import Fold
from games.poker.actions.check import Check
from games.poker.actions.raise_bet import RaiseBet


class AIPlayer(Player):

    def __init__(self, name, initial_chips):
        super().__init__(name, initial_chips)

    def make_decision_rollouts(self, oracle: PokerOracle, public_cards: List[Card], num_opponent_players: int, legal_actions: List[Action]) -> Action:
        win_prob, tie_prob, lose_prob = oracle.perform_rollouts(self.hand, public_cards, num_opponent_players)
        print(win_prob, tie_prob, lose_prob)
        print([(card.rank, card.suit) for card in self.hand])

        # Fold if probability of winning is less than 0.5
        if win_prob < 0.5:
            for action in legal_actions:
                if isinstance(action, Fold):
                    return action

        all_in_action = [action for action in legal_actions if isinstance(action, RaiseBet) and action.raise_type == "all_in"]
        # All in if probability of winning is higher than 0.85
        if all_in_action and win_prob > 0.85:
            return all_in_action[0]

        # Randomly choose check, call or raise when available
        check_action = [action for action in legal_actions if isinstance(action, Check)]
        call_action = [action for action in legal_actions if isinstance(action, RaiseBet) and action.raise_type == "call"]
        raise_action = [action for action in legal_actions if isinstance(action, RaiseBet) and action.raise_type == "raise"]
        available_actions = [action for action in check_action + call_action + raise_action]
        return random.choice(available_actions)

    def make_decision_resolving(self) -> Action:
        pass
