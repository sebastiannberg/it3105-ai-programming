import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import random

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle
from games.poker.players.human_player import HumanPlayer
from games.poker.players.ai_player import AIPlayer


rules = {
    "num_ai_players": 1,
    "ai_strategy": "rollout",
    "num_human_players": 1,
    "deck_size": 52,
    "initial_chips": 100,
    "small_blind_amount": 2,
    "big_blind_amount": 4,
    "fixed_raise": 4,
    "max_num_raises_per_stage": 4,
    "max_pot_size": 20,
    "all_in_disabled": True
}

game_manager = PokerGameManager(rules, PokerOracle())
game_manager.start_game()
current_state = game_manager.game
current_player = game_manager.game.active_player
legal_actions = PokerStateManager.find_all_legal_actions(current_state, current_player, game_manager.rules)
if isinstance(current_player, HumanPlayer):
    selected_action = random.choice(legal_actions)
if isinstance(current_player, AIPlayer):
    selected_action = current_player.make_decision_rollouts(game_manager.oracle, game_manager.game.public_cards, len(game_manager.game.round_players)-1, legal_actions)
print(selected_action.name)
PokerStateManager.apply_action(game_manager.game, current_player, selected_action)
