import numpy as np
import math
import time

from games.poker.poker_state import PokerState
from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle
from games.poker.utils.hand_label_generator import HandLabelGenerator
from resolver.subtree.node import Node
from resolver.subtree.player_node import PlayerNode
from resolver.subtree.chance_node import ChanceNode
from resolver.subtree.terminal_node import TerminalNode
from resolver.config import subtree_config


class Resolver:

    def __init__(self, state_manager: PokerStateManager) -> None:
        self.state_manager = state_manager

    def build_initial_subtree(self, state: PokerState, end_stage: str, end_depth: int):
        print("Started build_initial_subtree")
        start_time = time.time()
        # Assumes that the root node is a player node and not a chance node or terminal node
        root = PlayerNode(state, "player_one", stage_depth=0)
        queue = [(root, 0)]  # Queue of tuples (node, current_stage_depth)

        i = 0
        while queue:
            i+=1
            print(f"\rNode {i}", end="")
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
                        child_node.set_utility_matrix(node.utility_matrix, node.hand_label_to_index)
                    else:
                        # Continue with a Player Node if still in the same gameplay phase
                        next_player = self.determine_next_player(state=child_state, current_player=node.player, new_stage=False)
                        child_node = PlayerNode(child_state, player=next_player, parent=node, stage_depth=next_stage_depth)
                        child_node.set_utility_matrix(node.utility_matrix, node.hand_label_to_index)
                    edge_value = child_state.history[-1][2]
                    node.add_child(child_node, edge_value)
                    queue.append((child_node, next_stage_depth))

            elif isinstance(node, ChanceNode):
                # Handle chance node children, initiating new stage
                child_states = self.state_manager.gen_chance_child_states(node.state, max_num_children=subtree_config["max_chance_node_children"])
                for child_state in child_states:
                    next_player = self.determine_next_player(state=child_state, current_player=None, new_stage=True)
                    child_node = PlayerNode(child_state, player=next_player, parent=node, stage_depth=1)
                    # Generate new utility matrix for descendants
                    utility_matrix, hand_label_to_index = PokerOracle.gen_utility_matrix(child_state.public_cards, num_cards_deck=self.state_manager.poker_rules["deck_size"])
                    child_node.set_utility_matrix(utility_matrix, hand_label_to_index)
                    parent_public_card_set = {(card.rank, card.suit) for card in node.state.public_cards}
                    edge_value = [card for card in child_state.public_cards if (card.rank, card.suit) not in parent_public_card_set]
                    node.add_child(child_node, edge_value)
                    queue.append((child_node, 1))  # Reset stage depth because of new stage

        end_time = time.time()
        duration = end_time - start_time
        duration_minutes = duration / 60
        print()
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

    def bayesian_range_update(self, range_prior, action, average_strategy):
        pass

    def subtree_traversal_rollout(self, node: Node, r1, r2, end_stage, end_depth):
        possible_hands, hand_label_to_index = PokerOracle.get_possible_hands(num_cards_deck=self.state_manager.poker_rules["deck_size"])

        if isinstance(node, TerminalNode):
            if node.state.stage == "showdown":
                print("showdown")
                v1 = np.dot(node.utility_matrix, r2.T)
                v2 = np.dot(-r1, node.utility_matrix)
            else:
                # A player has folded
                print("Fold node check:", node.state.history[-1][2], node.__class__.__name__)
                loser = node.state.history[-1][1]
                v1 = np.zeros((1, len(possible_hands)), dtype=np.float64)
                v2 = np.zeros((1, len(possible_hands)), dtype=np.float64)
                v_fold = node.state.pot / subtree_config["average_pot_size"]
                public_cards_set = set((card.rank, card.suit) for card in node.state.public_cards)
                for hand in possible_hands:
                    # If a card in the hand is also in the set of public cards, go to the next hand
                    if any((card.rank, card.suit) in public_cards_set for card in hand):
                        continue
                    hand_label = HandLabelGenerator.get_hand_label(hand)
                    index = hand_label_to_index[hand_label]
                    # TODO double check that this indexing is correct
                    if loser == "player_one":
                        v1[0, index] = -v_fold
                        v2[0, index] = v_fold
                    elif loser == "player_two":
                        v1[0, index] = v_fold
                        v2[0, index] = -v_fold
        elif node.state.stage == end_stage and node.stage_depth == end_depth:
            print("Run nerual network...")
            v1, v2 = self.run_neural_network(node.state.stage, node.state, r1, r2)
        elif isinstance(node, PlayerNode):
            v1 = np.zeros((1, len(possible_hands)), dtype=np.float64)
            v2 = np.zeros((1, len(possible_hands)), dtype=np.float64)
            for child, action in node.children:
                if node.player == "player_one":
                    updated_range = self.bayesian_range_update(r1, action, node.strategy)
                    v1, v2 = self.subtree_traversal_rollout(child, updated_range, r2, end_stage, end_depth)
                elif node.player == "player_two":
                    updated_range = self.bayesian_range_update(r2, action, node.strategy)
                    v1_action, v2_action = self.subtree_traversal_rollout(child, r1, updated_range, end_stage, end_depth)
                for hand in possible_hands:
                    hand_label = HandLabelGenerator.get_hand_label(hand)
                    index = hand_label_to_index[hand_label]
                    # TODO double check that this is correct, maybe v1 * v2 instead but not sure
                    v1[0, index] += node.get_strategy(hand, action) * v1_action[0, index]
                    v2[0, index] += node.get_strategy(hand, action) * v2_action[0, index]
        else:
            # Enter this block if node is a chance node
            print("chance node check:", node.__class__.__name__)
            v1 = np.zeros((1, len(possible_hands)), dtype=np.float64)
            v2 = np.zeros((1, len(possible_hands)), dtype=np.float64)
            for child, event in node.children:
                v1_event, v2_event = self.subtree_traversal_rollout(child, r1, r2, end_stage, end_depth)

        return v1, v2


    def resolve(self, state: PokerState, r1, r2, end_stage, end_depth, T):
        root = self.build_initial_subtree(state, end_stage, end_depth)

        for _ in range(T):
            v1, v2 = self.subtree_traversal_rollout(root, r1, r2, end_stage, end_depth)

    def run_neural_network(self, stage, state, r1, r2):
        return r1, r2
