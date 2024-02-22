import random

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_oracle import PokerOracle

random.seed(123)

rules = {
    "num_ai_players": 1,
    "num_human_players": 1,
    "initial_chips": 100,
    "deck_size": 52,
    "small_blind_amount": 2,
    "big_blind_amount": 4,
    "fixed_raise": 4,
    "max_num_raises_per_stage": 4,
    "all_in_disabled": True,
    "max_pot_size": 20
}

poker_oracle = PokerOracle()
poker_game_manager = PokerGameManager(rules, poker_oracle)
poker_game_manager.init_poker_game()
print(poker_game_manager.game)
poker_game_manager.assign_blind_roles()
print(poker_game_manager.game.big_blind_player)
poker_game_manager.perform_blind_bets()
print(poker_game_manager.game.pot)
print(poker_game_manager.game.current_bet)
print(poker_game_manager.game.big_blind_player.chips)
