from poker_game_manager import PokerGameManager
from poker_state_manager import PokerStateManager
import json


class RulesParser:

    def generate_managers(self):
        # Read JSON file
        with open("/data/rules.json", "r") as file:
            rules = json.load(file)

            # Generate PokerGameManager based on extracted data
            poker_game_manager = PokerGameManager(rules=rules)

            # Generate PokerStateManager (if needed) based on extracted data
            poker_state_manager = PokerStateManager()

        # Return the generated managers
        return poker_game_manager, poker_state_manager
