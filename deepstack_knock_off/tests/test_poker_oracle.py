import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games.poker.poker_oracle import PokerOracle
from games.poker.utils.card import Card


oracle = PokerOracle()

royal_flush_tests = (
    (
        [Card("A", "spades"), Card("K", "spades")],
        [Card("Q", "spades"), Card("J", "spades"), Card("10", "spades"), Card("9", "spades"), Card("8", "spades")]
    ),
    (
        [Card("A", "hearts"), Card("K", "hearts")],
        [Card("Q", "hearts"), Card("J", "hearts"), Card("10", "hearts"), Card("2", "diamonds"), Card("4", "clubs")]
    ),
    (
        [Card("A", "diamonds"), Card("K", "diamonds")],
        [Card("Q", "diamonds"), Card("J", "diamonds"), Card("10", "diamonds"), Card("9", "clubs"), Card("8", "clubs")]
    ),
    (
        [Card("A", "clubs"), Card("K", "clubs")],
        [Card("Q", "clubs"), Card("J", "clubs"), Card("10", "clubs"), Card("A", "diamonds"), Card("K", "hearts")]
    ),
    (
        [Card("A", "spades"), Card("K", "spades")],
        [Card("Q", "spades"), Card("J", "spades"), Card("10", "spades"), Card("2", "spades"), Card("3", "spades")]
    ),
    (
        [Card("9", "hearts"), Card("8", "hearts")],
        [Card("A", "diamonds"), Card("K", "diamonds"), Card("Q", "diamonds"), Card("J", "diamonds"), Card("10", "diamonds")]
    ),
)

for hand, public_cards in royal_flush_tests:
    hand_type = oracle.classify_poker_hand(hand, public_cards)
    assert hand_type.category == "royal_flush", f"Expected 'royal_flush' but was '{hand_type.category}' in test with hand {[(card.rank, card.suit) for card in hand]} and public {[(card.rank, card.suit) for card in public_cards]}"

straight_flush_tests = (
    (
        [Card("9", "hearts"), Card("8", "hearts")],
        [Card("7", "hearts"), Card("6", "hearts"), Card("5", "hearts"), Card("4", "clubs"), Card("3", "spades")]
    ),
    (
        [Card("6", "clubs"), Card("5", "clubs")],
        [Card("4", "clubs"), Card("3", "clubs"), Card("2", "clubs"), Card("A", "spades"), Card("K", "hearts")]
    ),
    (
        [Card("5", "diamonds"), Card("4", "diamonds")],
        [Card("3", "diamonds"), Card("2", "diamonds"), Card("A", "diamonds"), Card("K", "diamonds"), Card("Q", "spades")]
    ),
    (
        [Card("10", "spades"), Card("9", "spades")],
        [Card("8", "spades"), Card("7", "spades"), Card("6", "spades"), Card("A", "spades"), Card("2", "hearts")]
    ),
)

for hand, public_cards in straight_flush_tests:
    hand_type = oracle.classify_poker_hand(hand, public_cards)
    assert hand_type.category == "straight_flush", f"Expected 'straight_flush' but was '{hand_type.category}' in test with hand {[(card.rank, card.suit) for card in hand]} and public {[(card.rank, card.suit) for card in public_cards]}"


four_of_a_kind_tests = (
    (
        [Card("9", "hearts"), Card("9", "spades")],
        [Card("7", "hearts"), Card("6", "diamonds"), Card("5", "hearts"), Card("9", "clubs"), Card("9", "diamonds")]
    ),
    (
        [Card("2", "hearts"), Card("7", "spades")],
        [Card("3", "hearts"), Card("3", "diamonds"), Card("5", "hearts"), Card("3", "clubs"), Card("3", "spades")]
    ),
    (
        [Card("2", "hearts"), Card("7", "spades")],
        [Card("2", "spades"), Card("2", "diamonds"), Card("2", "clubs")]
    ),
)

for hand, public_cards in four_of_a_kind_tests:
    hand_type = oracle.classify_poker_hand(hand, public_cards)
    assert hand_type.category == "four_of_a_kind", f"Expected 'four_of_a_kind' but was '{hand_type.category}' in test with hand {[(card.rank, card.suit) for card in hand]} and public {[(card.rank, card.suit) for card in public_cards]}"
