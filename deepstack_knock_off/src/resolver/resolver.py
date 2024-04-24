from typing import Dict, List
import numpy as np
import time

from games.poker.poker_state import PokerState
from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle
from games.poker.utils.hand_label_generator import HandLabelGenerator
from games.poker.utils.card import Card
from resolver.neural_network.predictor import Predictor
from resolver.subtree.node import Node
from resolver.subtree.player_node import PlayerNode
from resolver.subtree.chance_node import ChanceNode
from resolver.subtree.terminal_node import TerminalNode
from resolver.config import subtree_config


class Resolver:

    def __init__(self, state_manager: PokerStateManager) -> None:
        self.state_manager = state_manager
        possible_hands, hand_label_to_index, _ = PokerOracle.get_possible_hands_with_indexing(deck_size=self.state_manager.poker_rules["deck_size"])
        self.possible_hands = possible_hands
        self.hand_label_to_index = hand_label_to_index
        self.predictor = Predictor()

    def build_initial_subtree(self, state: PokerState, end_stage: str, end_depth: int):
        start_time = time.time()
        # Assumes that the root node is a player node and not a chance node or terminal node
        root = PlayerNode(state, "player_one", stage_depth=0)
        queue = [(root, 0)]  # Queue of tuples (node, current_stage_depth)

        # If root stage is not preflop we have to generate the utility matrix for root
        if root.state.stage != "preflop":
            root.set_utility_matrix(PokerOracle.gen_utility_matrix(root.state.public_cards, self.state_manager.poker_rules["deck_size"]))

        while queue:
            node, current_stage_depth = queue.pop(0)  # Dequeue the first item

            # Stop processing at the end stage and depth for that stage
            if node.state.stage == end_stage and current_stage_depth >= end_depth:
                continue

            # Skip generating children for TerminalNodes
            if isinstance(node, TerminalNode):
                continue

            if isinstance(node, PlayerNode):
                legal_actions = self.state_manager.find_legal_actions(node.state, node.player)
                for action in legal_actions:
                    child_state = self.state_manager.gen_player_child_state(node.state, node.player, action)
                    next_stage_depth = current_stage_depth + 1 if child_state.stage == node.state.stage else 0

                    if node.state.stage != child_state.stage and child_state.stage != "showdown":
                        # Create a Chance Node when transitioning to a new stage
                        child_node = ChanceNode(child_state, parent=node, stage_depth=next_stage_depth)
                    elif child_state.history[-1][2] == "fold" or child_state.stage == "showdown":
                        # Create a Terminal Node for game-ending actions
                        child_node = TerminalNode(child_state, parent=node, stage_depth=next_stage_depth)
                        child_node.set_utility_matrix(node.utility_matrix)
                    else:
                        # Continue with a Player Node if still in the same gameplay phase
                        next_player = self.determine_next_player(state=child_state, current_player=node.player, new_stage=False)
                        child_node = PlayerNode(child_state, player=next_player, parent=node, stage_depth=next_stage_depth)
                        child_node.set_utility_matrix(node.utility_matrix)
                    edge_value = child_state.history[-1][2]
                    node.add_child(child_node, edge_value)
                    queue.append((child_node, next_stage_depth))
                # We can initialize the strategy matrix now that the children are connected to the parent
                node.init_strategy_and_regret_matrix(deck_size=self.state_manager.poker_rules["deck_size"])

            elif isinstance(node, ChanceNode):
                # Handle chance node children, initiating new stage
                child_states = self.state_manager.gen_chance_child_states(node.state, max_num_children=subtree_config["max_chance_node_children"])
                for child_state in child_states:
                    next_player = self.determine_next_player(state=child_state, current_player=None, new_stage=True)
                    child_node = PlayerNode(child_state, player=next_player, parent=node, stage_depth=1)
                    # Generate new utility matrix for descendants
                    utility_matrix = PokerOracle.gen_utility_matrix(child_state.public_cards, deck_size=self.state_manager.poker_rules["deck_size"])
                    child_node.set_utility_matrix(utility_matrix)
                    parent_public_card_set = {(card.rank, card.suit) for card in node.state.public_cards}
                    edge_value = [card for card in child_state.public_cards if (card.rank, card.suit) not in parent_public_card_set]
                    node.add_child(child_node, edge_value)
                    queue.append((child_node, 1))  # Reset stage depth because of new stage

        end_time = time.time()
        duration = end_time - start_time
        duration_minutes = duration / 60
        print(f"build_initial_subtree took {duration_minutes:.2f} minutes to run")
        return root

    def determine_next_player(self, state, current_player, new_stage=False):
        small_blind_player = state.history[0][1]
        if new_stage:
            return small_blind_player
        else:
            return "player_two" if current_player == "player_one" else "player_one"

    def count_nodes(self, root: Node):
        count = 1
        for child, _ in root.children:
            count += self.count_nodes(child)
        return count

    def get_leaf_nodes(self, root: Node):
        leaves = []
        queue = [root]

        while queue:
            node = queue.pop(0)
            if not node.children:
                leaves.append(node)
            else:
                queue.extend(child for child, _ in node.children)

        return leaves

    def find_tree_paths(self, node, path=None):
        if path is None:
            path = []

        # Create a new path up to this node
        current_path = path + [(node, None)]  # Append current node with no edge initially

        # If it's a terminal node , finalize the path
        if not node.children:
            return [current_path]

        # Otherwise, extend DFS to each child
        paths = []
        for child, edge_to_child in node.children:
            # Update the current node's tuple to include the edge leading to the child
            current_path[-1] = (node, edge_to_child)
            paths.extend(self.find_tree_paths(child, current_path))

        return paths

    def print_tree_path(self, path):
        for node, edge in path:
            if isinstance(node, PlayerNode):
                print(node.state, node.__class__.__name__, node.player)
            else:
                print(node.state, node.__class__.__name__)
            if edge:
                print("  |  ")
                print(edge)
                print("  |  ")

    def bayesian_range_update(self, range_prior: np.ndarray, action: str, strategy_matrix: np.ndarray, action_to_index: Dict[str, int]):
        updated_range = np.copy(range_prior)
        for hand in self.possible_hands:
            hand_label = HandLabelGenerator.get_hand_label(hand)
            prob_action_hand = strategy_matrix[self.hand_label_to_index[hand_label], action_to_index[action]]
            prob_hand = range_prior[0, self.hand_label_to_index[hand_label]]
            prob_action = np.sum(strategy_matrix[:, action_to_index[action]]) / np.sum(strategy_matrix)
            updated_range_value = (prob_action_hand * prob_hand) / prob_action
            updated_range[0, self.hand_label_to_index[hand_label]] = updated_range_value
        return updated_range

    def subtree_traversal_rollout(self, node: Node, r1, r2, end_stage, end_depth):
        if isinstance(node, TerminalNode):
            if node.state.stage == "showdown":
                v1 = np.dot(node.utility_matrix, r2.T).T
                v2 = np.dot(-r1, node.utility_matrix)
            else:
                # A player has folded
                loser = node.state.history[-1][1]
                if loser == "player_one":
                    negative_utility_matrix = -np.abs(node.utility_matrix)
                    v1 = np.dot(negative_utility_matrix, r2.T).T
                    v2 = np.dot(-r1, negative_utility_matrix)
                if loser == "player_two":
                    positive_utility_matrix = np.abs(node.utility_matrix)
                    v1 = np.dot(positive_utility_matrix, r2.T).T
                    v2 = np.dot(-r1, positive_utility_matrix)
        elif node.state.stage == end_stage and node.stage_depth == end_depth:
            v1, v2 = self.run_neural_network(node.state.stage, node.state, r1, r2)
        elif isinstance(node, PlayerNode):
            v1 = np.zeros((1, len(self.possible_hands)), dtype=np.float64)
            v2 = np.zeros((1, len(self.possible_hands)), dtype=np.float64)
            for child, action in node.children:
                if node.player == "player_one":
                    updated_range = self.bayesian_range_update(r1, action, node.strategy_matrix, node.action_to_index)
                    v1_action, v2_action = self.subtree_traversal_rollout(child, updated_range, r2, end_stage, end_depth)
                elif node.player == "player_two":
                    updated_range = self.bayesian_range_update(r2, action, node.strategy_matrix, node.action_to_index)
                    v1_action, v2_action = self.subtree_traversal_rollout(child, r1, updated_range, end_stage, end_depth)
                for hand in self.possible_hands:
                    hand_label = HandLabelGenerator.get_hand_label(hand)
                    index = self.hand_label_to_index[hand_label]
                    v1[0, index] += node.get_action_probability(hand_label=hand_label, action=action) * v1_action[0, index]
                    v2[0, index] += node.get_action_probability(hand_label=hand_label, action=action) * v2_action[0, index]
        else:
            # Entering this block if node is a chance node
            v1 = np.zeros((1, len(self.possible_hands)), dtype=np.float64)
            v2 = np.zeros((1, len(self.possible_hands)), dtype=np.float64)
            for child, _ in node.children:
                v1_event, v2_event = self.subtree_traversal_rollout(child, r1, r2, end_stage, end_depth)
                v1 += v1_event/len(node.children)
                v2 += v2_event/len(node.children)
        node.v1 = v1
        node.v2 = v2
        return v1, v2

    def update_strategy(self, node: Node):
        for child in node.children:
            if isinstance(child[0], PlayerNode):
                self.update_strategy(child[0])

        if isinstance(node, PlayerNode):
            all_actions = [child[1] for child in node.children]
            positive_regret_matrix = np.zeros((len(self.possible_hands), len(all_actions)))

            for hand in self.possible_hands:
                for action in all_actions:
                    child_node = [child[0] for child in node.children if child[1] == action][0]
                    hand_label = HandLabelGenerator.get_hand_label(hand)
                    hand_index = node.hand_label_to_index[hand_label]
                    action_index = node.action_to_index[action]
                    if node.player == "player_one":
                        regret = child_node.v1[0, hand_index] - node.v1[0, hand_index]
                    else:  # player_two
                        regret = child_node.v2[0, hand_index] - node.v2[0, hand_index]
                    node.cumulative_regrets[hand_index, action_index] += regret
                    positive_regret_matrix[hand_index, action_index] = max(0, node.cumulative_regrets[hand_index, action_index])

            for hand in self.possible_hands:
                hand_label = HandLabelGenerator.get_hand_label(hand)
                hand_index = node.hand_label_to_index[hand_label]
                total_positive_regrets = np.sum(positive_regret_matrix[hand_index, :])

                if total_positive_regrets == 0:
                    # Distribute probabilities equally across all actions since there are no preferred actions
                    num_actions = len(all_actions)
                    for action_index in range(num_actions):
                        node.strategy_matrix[hand_index, action_index] = 1.0 / num_actions
                else:
                    for action_index in range(len(all_actions)):
                        node.strategy_matrix[hand_index, action_index] = positive_regret_matrix[hand_index, action_index] / total_positive_regrets

        return node.strategy_matrix

    def resolve(self, state: PokerState, r1, r2, end_stage, end_depth, T: int, player_hand: List[Card]):
        root = self.build_initial_subtree(state, end_stage, end_depth)

        strategy_matrices = []
        for _ in range(T):
            v1, v2 = self.subtree_traversal_rollout(root, r1, r2, end_stage, end_depth)
            root_strategy_matrix = self.update_strategy(root)
            strategy_matrices.append(root_strategy_matrix)

        strategy_matrices_np = np.array(strategy_matrices)
        # Compute the average across the matrices
        average_strategy_matrix = np.mean(strategy_matrices_np, axis=0)

        hand_label = HandLabelGenerator.get_hand_label(player_hand)
        hand_index = self.hand_label_to_index[hand_label]
        action_probabilities = average_strategy_matrix[hand_index]
        action_indices = np.arange(len(action_probabilities))
        chosen_action_index = np.random.choice(action_indices, p=action_probabilities)

        chosen_action_probability = average_strategy_matrix[hand_index, chosen_action_index]
        chosen_action = root.index_to_action[chosen_action_index]
        print(chosen_action, chosen_action_probability)

        # Update range
        updated_r1 = self.bayesian_range_update(r1, chosen_action, average_strategy_matrix, root.action_to_index)
        return chosen_action, updated_r1, v1, v2

    def run_neural_network(self, stage, state: PokerState, r1, r2):
        public_cards = [str(card) for card in state.public_cards]

        predicted_v1, predicted_v2 = self.predictor.make_prediction(stage, r1, r2, public_cards, state.pot)

        # Element-wise multiplication to zero out indices where range vectors are zero
        predicted_v1 = predicted_v1 * (r1 != 0)
        predicted_v2 = predicted_v2 * (r2 != 0)

        return predicted_v1, predicted_v2
