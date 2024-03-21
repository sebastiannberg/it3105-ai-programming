import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games.poker.poker_oracle import PokerOracle
from games.poker.utils.card import Card


oracle = PokerOracle()

# Royal Flush
hand = [Card("A", "spades"), Card("K", "spades")]
public_cards = [Card("Q", "spades"), Card("J", "spades"), Card("10", "spades"), Card("9", "spades"), Card("8", "spades")]
hand_type = oracle.classify_poker_hand(hand, public_cards)
assert hand_type.category == "royal_flush", f"Expected 'royal_flush' but was {hand_type.category}"

public_cards = [Card("10", "spades"), Card("2", "clubs"), Card("K", "hearts")]
oracle.gen_utility_matrix(public_cards=public_cards)
