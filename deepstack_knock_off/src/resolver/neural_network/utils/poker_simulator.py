import random
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..", "src")))

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle
from games.poker.utils.hand_label_generator import HandLabelGenerator
from games.poker.poker_state import PokerState
from games.poker.actions.fold import Fold
from resolver.resolver import Resolver

def simulate_poker_game(end_stage: str) -> PokerState:
    game_manager = PokerGameManager(poker_config, poker_rules)
    game_manager.start_game()
    while game_manager.game.stage != end_stage:
        player = game_manager.game.current_player
        game_manager.assign_legal_actions_to_player(player_name=player.name)
        # Remove fold action
        all_actions = [action for action in player.legal_actions if not isinstance(action, Fold)]
        selected_action = random.choice(all_actions)
        game_manager.apply_action(player.name, selected_action.name)
    ai_player = game_manager.game.game_players[0]
    return game_manager.state_manager.gen_state_from_game(game_manager.game, player_one_perspective=ai_player), ai_player.hand


def generate_river_case():
    pass


small_blind_random = random.randint(2, 10)
big_blind_random = small_blind_random * 2
scaler = random.randint(2, 10)
inital_chips_random = scaler * big_blind_random
max_num_raises_per_stage_random = random.randint(2, 4)
poker_config = {
    "num_ai_players": 1,
    "enable_resolver": False,
    "prob_resolver": 0.5,
    "num_human_players": 1,
    "initial_chips": inital_chips_random,
    "small_blind_amount": small_blind_random,
    "big_blind_amount": big_blind_random
}
poker_rules = {
    "deck_size": 24,
    "max_num_raises_per_stage": max_num_raises_per_stage_random,
    "fixed_raise": big_blind_random
}
resolver = Resolver(PokerStateManager(poker_rules))

# River Network
state, player_hand = simulate_poker_game(end_stage="river")
# Generate random random range vectors
public_cards_set = set([(card.rank, card.suit) for card in state.public_cards])
possible_hands, hand_label_to_index, _ = PokerOracle.get_possible_hands_with_indexing(deck_size=poker_rules["deck_size"])
r1 = np.zeros((1, len(possible_hands)), dtype=np.float64)
r2 = np.zeros((1, len(possible_hands)), dtype=np.float64)
for hand in possible_hands:
    hand_label = HandLabelGenerator.get_hand_label(hand)
    hand_index = hand_label_to_index[hand_label]
    # If a card in the hand is also in the set of public cards, go to the next hand
    if any((card.rank, card.suit) in public_cards_set for card in hand):
        continue
    r1[0, hand_index] = random.random()
    r2[0, hand_index] = random.random()
# Normalize r1 and r2
if r1.sum() > 0:
    r1 /= r1.sum()
if r2.sum() > 0:
    r2 /= r2.sum()
print(state)
print(player_hand)
_, _, v1, v2 = resolver.resolve(state, r1, r2, end_stage="showdown", end_depth=0, T=20, player_hand=player_hand)
