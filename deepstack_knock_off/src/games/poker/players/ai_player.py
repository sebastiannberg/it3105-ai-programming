from typing import List
import numpy as np

from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_game import PokerGame
from games.poker.players.player import Player
from games.poker.actions.action import Action
from games.poker.utils.card import Card
from games.poker.poker_oracle import PokerOracle
from games.poker.actions.fold import Fold
from games.poker.actions.check import Check
from games.poker.actions.raise_bet import RaiseBet
from resolver.resolver import Resolver


class AIPlayer(Player):

    def __init__(self, name, initial_chips, state_manager: PokerStateManager):
        super().__init__(name, initial_chips)
        self.resolver = Resolver(state_manager)
        self.state_manager = state_manager
        possible_hands, _, _ = PokerOracle.get_possible_hands_with_indexing(deck_size=self.state_manager.poker_rules["deck_size"])
        self.r1 = np.full((1, len(possible_hands)), 1/len(possible_hands), dtype=np.float64)
        self.r2 = np.full((1, len(possible_hands)), 1/len(possible_hands), dtype=np.float64)

    def make_decision_rollouts(self, public_cards: List[Card], num_cards_deck: int, num_opponent_players: int) -> Action:
        win_prob, tie_prob, lose_prob = PokerOracle.perform_rollouts(self.hand, public_cards, num_cards_deck, num_opponent_players)
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

        actions, scores = zip(*action_scores.items())
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores] if total_score > 0 else [1.0 / len(scores)] * len(scores)

        for action, probability in zip(actions, probabilities):
            print(f"Action: {action.name}, Probability: {probability:.4f}")

        chosen_action = np.random.choice(actions, p=probabilities)
        return chosen_action

    def make_decision_resolving(self, game: PokerGame) -> Action:
        state = self.state_manager.gen_state_from_game(game, player_one_perspective=self)
        # If stage is river, build tree to showdown stage
        if state.stage == "river":
            chosen_action, updated_r1, _, _ = self.resolver.resolve(state, self.r1, self.r2, end_stage="showdown", end_depth=0, T=3, player_hand=self.hand)
        # Else build tree to next stage with depth 1
        else:
            next_stage = self.state_manager.stage_change[state.stage]
            chosen_action, updated_r1, _, _ = self.resolver.resolve(state, self.r1, self.r2, end_stage=next_stage, end_depth=1, T=3, player_hand=self.hand)
        self.r1 = updated_r1

        action = chosen_action.lower()  # Convert to lowercase to ensure consistency in comparison
        if "fold" in chosen_action.lower():
            return next((a for a in self.legal_actions if isinstance(a, Fold)), None)
        if "check" in chosen_action.lower():
            return next((a for a in self.legal_actions if isinstance(a, Check)), None)
        if "call" in chosen_action.lower():
            return next((a for a in self.legal_actions if isinstance(a, RaiseBet) and getattr(a, "raise_type", None) == "call"), None)
        if "raise" in chosen_action.lower():
            # Assuming the raise amount is the last element in the action string
            raise_amount = int(action.split()[-1])
            return next((a for a in self.legal_actions if isinstance(a, RaiseBet) and a.raise_amount == raise_amount), None)
