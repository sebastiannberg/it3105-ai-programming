import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_state_manager import PokerStateManager
from games.poker.players.ai_player import AIPlayer
from games.poker.utils.card import Card
from games.poker.poker_state import PokerState
from resolver.resolver import Resolver
from resolver.subtree.terminal_node import TerminalNode

poker_config = {
    "num_ai_players": 1,
    "enable_resolver": False,
    "prob_resolver": 0.5,
    "num_human_players": 1,
    "initial_chips": 20,
    "small_blind_amount": 2,
    "big_blind_amount": 4
}
poker_rules = {
    "deck_size": 52,
    "fixed_raise": 4,
    "max_num_raises_per_stage": 4
}
resolver = Resolver(PokerStateManager(poker_rules))

game_manager = PokerGameManager(poker_config, poker_rules)
game_manager.start_game()
player = game_manager.game.current_player
if isinstance(player, AIPlayer):
    player.make_decision_resolving(game_manager.game)
