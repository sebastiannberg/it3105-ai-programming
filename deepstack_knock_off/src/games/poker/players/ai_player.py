from typing import List
import numpy as np

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

    def make_decision_rollouts(self, oracle: PokerOracle, public_cards: List[Card], num_opponent_players: int) -> Action:
        win_prob, tie_prob, lose_prob = oracle.perform_rollouts(self.hand, public_cards, num_opponent_players)
        print(win_prob, tie_prob, lose_prob)

        # Normalize probabilities
        total_prob = win_prob + tie_prob + lose_prob
        win_prob /= total_prob
        tie_prob /= total_prob
        lose_prob /= total_prob

        # If both Check and Fold are available, discard Fold action from consideration
        if any(isinstance(action, Check) for action in self.legal_actions) and any(isinstance(action, Fold) for action in self.legal_actions):
            actions_to_consider = [action for action in self.legal_actions if not isinstance(action, Fold)]
        else:
            actions_to_consider = self.legal_actions

        action_scores = {}
        for action in actions_to_consider:
            if isinstance(action, Check):
                action_scores[action] = win_prob
            elif isinstance(action, Fold):
                action_scores[action] = lose_prob * 0.75
            elif isinstance(action, RaiseBet):
                if action.raise_type == "call":
                    action_scores[action] = win_prob + (tie_prob * 0.75)
                elif action.raise_type == "raise":
                    action_scores[action] = win_prob * 0.85 + (tie_prob * 0.5)
                elif action.raise_type == "all_in":
                    action_scores[action] = win_prob * 0.95 + (tie_prob * 0.05)

        actions, scores = zip(*action_scores.items())
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores] if total_score > 0 else [1.0 / len(scores)] * len(scores)

        for action, probability in zip(actions, probabilities):
            print(f"Action: {action.name}, Probability: {probability:.4f}")

        chosen_action = np.random.choice(actions, p=probabilities)
        return chosen_action

    def make_decision_resolving(self) -> Action:
        pass
