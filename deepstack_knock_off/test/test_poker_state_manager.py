import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_state import PokerState
from games.poker.utils.card import Card

poker_config = {
    "num_ai_players": 1,
    "enable_resolver": False,
    "prob_resolver": 0.5,
    "num_human_players": 1,
    "initial_chips": 100,
    "small_blind_amount": 2,
    "big_blind_amount": 4
}
poker_rules = {
    "deck_size": 52,
    "fixed_raise": 4,
    "all_in_disabled": True,
    "max_num_raises_per_stage": 4
}
game_manager = PokerGameManager(poker_config, poker_rules)
state_manager = PokerStateManager(poker_rules=poker_rules)

game_manager.start_game()
game_manager.assign_legal_actions_to_player("AI Player 1")
game_manager.apply_action("AI Player 1", "Call (2)")
game_manager.assign_legal_actions_to_player("Human Player 1")
game_manager.apply_action("Human Player 1", "Check")
state = state_manager.gen_state_from_game(game_manager.game, game_manager.game.game_players[0])
assert len(state.public_cards) == 3
assert state.pot == 8
assert state.player_one_bet == 0

parent_state = PokerState(
    public_cards=[],
    player_one_chips=85,
    player_one_bet=0,
    player_two_chips=90,
    player_two_bet=0,
    pot=20,
    stage="flop",
    stage_history=[]
)
child_states = state_manager.gen_chance_child_states(parent_state, max_num_children=100)
assert len(child_states) == 100
assert len(child_states[0].public_cards) == 3
assert len(parent_state.public_cards) == 0

parent_state = PokerState(
    public_cards=[Card('7', 'diamonds'), Card('3', 'hearts'), Card('8', 'diamonds')],
    player_one_chips=85,
    player_one_bet=0,
    player_two_chips=90,
    player_two_bet=0,
    pot=20,
    stage="turn",
    stage_history=[]
)
child_states = state_manager.gen_chance_child_states(parent_state)
assert len(child_states) == 49
assert len(child_states[0].public_cards) == 4
assert len(parent_state.public_cards) == 3

game_manager = PokerGameManager(poker_config, poker_rules)
game_manager.start_game()
game_manager.assign_legal_actions_to_player("AI Player 1")
game_manager.apply_action("AI Player 1", "Call (2)")
game_manager.assign_legal_actions_to_player("Human Player 1")
game_manager.apply_action("Human Player 1", "Check")
root_state = state_manager.gen_state_from_game(game_manager.game, player_one_perspective=game_manager.game.game_players[0])
print(root_state)
legal_actions = state_manager.find_legal_actions(root_state, "player_one")
print(legal_actions)
child_state = state_manager.gen_player_child_state(root_state, "player_one", "check")
print(child_state)
child_state = state_manager.gen_player_child_state(child_state, "player_two", "check")
print(child_state)
