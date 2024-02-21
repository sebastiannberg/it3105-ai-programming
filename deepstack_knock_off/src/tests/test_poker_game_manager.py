import os
import sys
import json
import random

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle

random.seed(123)

with open("/path/to/your/rules.json") as f:
    rules = json.load(f)

poker_oracle = PokerOracle()
poker_game_manager = PokerGameManager(rules, poker_oracle)
