import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games.poker.poker_state_manager import PokerStateManager
from games.poker.utils.card import Card
from games.poker.poker_state import PokerState
from resolver.resolver import Resolver


poker_rules = {
    "deck_size": 52,
    "fixed_raise": 4,
    "all_in_disabled": True,
    "max_num_raises_per_stage": 4
}

poker_state = PokerState(
    public_cards=[],
    player_one_chips=95,
    player_one_bet=5,
    player_two_chips=90,
    player_two_bet=10,
    pot=15,
    stage="preflop",
    history=[("preflop", "player_one", "small blind"), ("preflop", "player_two", "big_blind")]
)
resolver = Resolver(PokerStateManager(poker_rules))
root = resolver.build_initial_subtree(poker_state, end_stage="flop", end_depth=1)
print(resolver.count_nodes(root))

state = PokerState(public_cards=[Card('7', 'diamonds'), Card('3', 'hearts'), Card('A', 'clubs')],
                   player_one_chips=0,
                   player_one_bet=0,
                   player_two_chips=0,
                   player_two_bet=0,
                   pot=28,
                   stage='flop',
                   history=[('preflop', 'player_one', 'small blind'), ('preflop', 'player_two', 'big blind'), ('preflop', 'player_one', 'raise 4'), ('preflop', 'player_two', 'raise 4'), ('preflop', 'player_one', 'call')])
root = resolver.build_initial_subtree(state, end_stage="showdown", end_depth=0)
print(root.children)
print(resolver.count_nodes(root))
