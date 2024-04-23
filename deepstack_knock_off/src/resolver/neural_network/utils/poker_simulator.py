import random
import numpy as np
import pandas as pd
import datetime
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..", "src")))

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle
from games.poker.utils.hand_label_generator import HandLabelGenerator
from games.poker.actions.fold import Fold
from resolver.resolver import Resolver

def simulate_poker_game(end_stage: str, poker_config, poker_rules):
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
    small_blind_random = random.randint(2, 5)
    big_blind_random = 2 * small_blind_random
    initial_chips_scaler = random.randint(2, 20)
    initial_chips_random = initial_chips_scaler * big_blind_random
    max_num_raises_per_stage_random = random.randint(2, 4)
    poker_config = {
        "num_ai_players": 1,
        "enable_resolver": False,
        "prob_resolver": 0.5,
        "num_human_players": 1,
        "initial_chips": initial_chips_random,
        "small_blind_amount": small_blind_random,
        "big_blind_amount": big_blind_random
    }
    poker_rules = {
        "deck_size": 24,
        "max_num_raises_per_stage": max_num_raises_per_stage_random,
        "fixed_raise": big_blind_random
    }
    resolver = Resolver(PokerStateManager(poker_rules))

    state, player_hand = simulate_poker_game("river", poker_config, poker_rules)

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
    _, _, v1, v2 = resolver.resolve(state, r1, r2, end_stage="showdown", end_depth=0, T=50, player_hand=player_hand)

    return r1, r2, state.public_cards, state.pot, v1, v2


def generate_river_cases(num_cases: int):
    start_time = time.time()

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"river_cases_{date_str}.csv"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    output_file = os.path.join(parent_dir, "data", filename)

    # Check if the file exists and create it with headers if it does not
    try:
        # If the file exists, we append without writing headers
        pd.read_csv(output_file)
        headers = False
    except FileNotFoundError:
        # If the file does not exist, we create it with headers
        headers = True

    for i in range(num_cases):
        print(f"Case {i+1}")
        r1, r2, public_cards, pot, v1, v2 = generate_river_case()
        # Convert data to a format that can be saved in CSV
        case_dict = {
            "case_id": i,
            "r1": r1.flatten().tolist(),
            "r2": r2.flatten().tolist(),
            "public_cards": [str(card) for card in public_cards],
            "pot": pot,
            "v1": v1.flatten().tolist(),
            "v2": v2.flatten().tolist()
        }
        # Convert the dict to a DataFrame
        case_df = pd.DataFrame([case_dict])
        # Append the single case to the CSV file
        case_df.to_csv(output_file, mode='a', index=False, header=headers)
        # Update headers to False after the first write operation
        headers = False

    end_time = time.time()
    duration = end_time - start_time
    duration_minutes = duration / 60
    print(f"Generating data took {duration_minutes:.2f} minutes to run")

generate_river_cases(num_cases=5000)
