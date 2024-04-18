from games.poker.poker_state import PokerState
from games.poker.poker_state_manager import PokerStateManager
from resolver.subtree.node import Node
from resolver.subtree.player_node import PlayerNode
from resolver.subtree.chance_node import ChanceNode
from resolver.subtree.terminal_node import TerminalNode
from resolver.config import subtree_config


class Resolver:

    def __init__(self, state_manager: PokerStateManager) -> None:
        self.state_manager = state_manager

    def build_initial_subtree(self, state: PokerState, end_stage: str, end_depth: int):
        root = PlayerNode(state, "player_one")
        queue = [(root, 0)]  # Queue of tuples (node, current_stage_depth)

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

                    if node.state.stage != child_state.stage:
                        # Create a Chance Node when transitioning to a new stage
                        child_node = ChanceNode(child_state, parent=node)
                    elif child_state.history[-1][2] == "fold" or child_state.stage == "showdown":
                        # Create a Terminal Node for game-ending actions
                        child_node = TerminalNode(child_state, parent=node)
                    else:
                        # Continue with a Player Node if still in the same gameplay phase
                        next_player = "player_two" if node.player == "player_one" else "player_one"
                        child_node = PlayerNode(child_state, player=next_player, parent=node)
                    edge_value = child_state.history[-1][2]
                    node.add_child(child_node, edge_value)
                    queue.append((child_node, next_stage_depth))

            elif isinstance(node, ChanceNode):
                # Handle chance node children, potentially initiating new stage
                child_states = self.state_manager.gen_chance_child_states(node.state, max_num_children=subtree_config["max_chance_node_children"])
                for child_state in child_states:
                    next_player = "player_two" if node.parent and node.parent.player == "player_one" else "player_one"
                    child_node = PlayerNode(child_state, player=next_player, parent=node)
                    public_card_set = {(card.rank, card.suit) for card in node.state.public_cards}
                    edge_value = [card for card in child_state.public_cards if (card.rank, card.suit) not in public_card_set]
                    node.add_child(child_node, edge_value)
                    queue.append((child_node, 0))  # Reset stage depth because of new stage

        return root

    def count_nodes(self, root: Node):
        count = 1
        for child, _ in root.children:
            count += self.count_nodes(child)
        return count

    def bayesian_range_update(self, range_prior, action_probability, sigma):
