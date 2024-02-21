import json
import random
import os

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle

random.seed(123)

current_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_dir, "games", "poker", "data", "rules.json")

with open(json_file_path) as f:
    rules = json.load(f)

poker_oracle = PokerOracle()
poker_game_manager = PokerGameManager(rules, poker_oracle)
poker_game_manager.init_poker_game()
print(poker_game_manager.game.state)
print(poker_game_manager.game.state.big_blind_player)
