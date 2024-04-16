import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games.poker.poker_oracle import PokerOracle
from games.poker.utils.card import Card
from games.poker.utils.hand_type import HandType
from games.poker.utils.hand_label_generator import HandLabelGenerator


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

full_house_tests = (
    (
        [Card("9", "hearts"), Card("9", "spades")],
        [Card("7", "hearts"), Card("7", "diamonds"), Card("7", "clubs"), Card("2", "clubs"), Card("Q", "diamonds")]
    ),
    (
        [Card("8", "hearts"), Card("2", "clubs")],
        [Card("J", "hearts"), Card("8", "diamonds"), Card("5", "hearts"), Card("2", "hearts"), Card("2", "spades")]
    ),
)

for hand, public_cards in full_house_tests:
    hand_type = oracle.classify_poker_hand(hand, public_cards)
    assert hand_type.category == "full_house", f"Expected 'full_house' but was '{hand_type.category}' in test with hand {[(card.rank, card.suit) for card in hand]} and public {[(card.rank, card.suit) for card in public_cards]}"

flush_tests = (
    (
        [Card("J", "hearts"), Card("A", "spades")],
        [Card("7", "hearts"), Card("3", "diamonds"), Card("5", "hearts"), Card("6", "hearts"), Card("Q", "hearts")]
    ),
    (
        [Card("J", "hearts"), Card("J", "spades")],
        [Card("7", "diamonds"), Card("3", "diamonds"), Card("5", "diamonds"), Card("6", "diamonds"), Card("Q", "diamonds")]
    ),
)

for hand, public_cards in flush_tests:
    hand_type = oracle.classify_poker_hand(hand, public_cards)
    assert hand_type.category == "flush", f"Expected 'flush' but was '{hand_type.category}' in test with hand {[(card.rank, card.suit) for card in hand]} and public {[(card.rank, card.suit) for card in public_cards]}"

straight_tests = (
    (
        [Card("J", "hearts"), Card("A", "spades")],
        [Card("7", "hearts"), Card("8", "diamonds"), Card("9", "diamonds"), Card("10", "hearts"), Card("Q", "hearts")]
    ),
    (
        [Card("A", "hearts"), Card("2", "spades")],
        [Card("3", "spades"), Card("4", "diamonds"), Card("5", "diamonds"), Card("10", "spades"), Card("Q", "clubs")]
    ),
)

for hand, public_cards in straight_tests:
    hand_type = oracle.classify_poker_hand(hand, public_cards)
    assert hand_type.category == "straight", f"Expected 'straight' but was '{hand_type.category}' in test with hand {[(card.rank, card.suit) for card in hand]} and public {[(card.rank, card.suit) for card in public_cards]}"

three_of_a_kind_tests = (
    (
        [Card("J", "hearts"), Card("A", "spades")],
        [Card("7", "hearts"), Card("J", "diamonds"), Card("9", "diamonds"), Card("J", "clubs"), Card("Q", "hearts")]
    ),
    (
        [Card("A", "hearts"), Card("A", "spades")],
        [Card("3", "spades"), Card("4", "diamonds"), Card("A", "diamonds"), Card("7", "spades"), Card("2", "clubs")]
    ),
    (
        [Card("A", "hearts"), Card("2", "spades")],
        [Card("3", "spades"), Card("3", "diamonds"), Card("3", "clubs"), Card("7", "spades"), Card("8", "clubs")]
    ),
)

for hand, public_cards in three_of_a_kind_tests:
    hand_type = oracle.classify_poker_hand(hand, public_cards)
    assert hand_type.category == "three_of_a_kind", f"Expected 'three_of_a_kind' but was '{hand_type.category}' in test with hand {[(card.rank, card.suit) for card in hand]} and public {[(card.rank, card.suit) for card in public_cards]}"

two_pair_tests = (
    (
        [Card("J", "hearts"), Card("A", "spades")],
        [Card("7", "hearts"), Card("7", "diamonds"), Card("9", "diamonds"), Card("9", "clubs"), Card("Q", "hearts")]
    ),
    (
        [Card("3", "hearts"), Card("6", "spades")],
        [Card("3", "spades"), Card("4", "diamonds"), Card("A", "diamonds"), Card("7", "spades"), Card("7", "clubs")]
    ),
)

for hand, public_cards in two_pair_tests:
    hand_type = oracle.classify_poker_hand(hand, public_cards)
    assert hand_type.category == "two_pair", f"Expected 'two_pair' but was '{hand_type.category}' in test with hand {[(card.rank, card.suit) for card in hand]} and public {[(card.rank, card.suit) for card in public_cards]}"

pair_tests = (
    (
        [Card("J", "hearts"), Card("J", "spades")],
        [Card("7", "hearts"), Card("2", "diamonds"), Card("9", "diamonds"), Card("A", "clubs"), Card("Q", "hearts")]
    ),
    (
        [Card("3", "hearts"), Card("6", "spades")],
        [Card("8", "spades"), Card("8", "diamonds"), Card("9", "diamonds"), Card("7", "spades"), Card("A", "clubs")]
    ),
)

for hand, public_cards in pair_tests:
    hand_type = oracle.classify_poker_hand(hand, public_cards)
    assert hand_type.category == "pair", f"Expected 'pair' but was '{hand_type.category}' in test with hand {[(card.rank, card.suit) for card in hand]} and public {[(card.rank, card.suit) for card in public_cards]}"

high_card_tests = (
    (
        [Card("J", "hearts"), Card("K", "spades")],
        [Card("7", "hearts"), Card("2", "diamonds"), Card("9", "diamonds"), Card("A", "clubs"), Card("Q", "hearts")]
    ),
    (
        [Card("3", "hearts"), Card("6", "spades")],
        [Card("8", "spades"), Card("2", "diamonds"), Card("J", "diamonds"), Card("7", "spades"), Card("A", "clubs")]
    ),
)

for hand, public_cards in high_card_tests:
    hand_type = oracle.classify_poker_hand(hand, public_cards)
    assert hand_type.category == "high_card", f"Expected 'high_card' but was '{hand_type.category}' in test with hand {[(card.rank, card.suit) for card in hand]} and public {[(card.rank, card.suit) for card in public_cards]}"

player_win_tests = (
    (
        HandType(category="flush", primary_value=7, kickers=[]),
        HandType(category="two_pair", primary_value=14, kickers=[3, 2])
    ),
    (
        HandType(category="straight_flush", primary_value=9, kickers=[]),
        HandType(category="four_of_a_kind", primary_value=7, kickers=[12])
    ),
    (
        HandType(category="full_house", primary_value=11, kickers=[4]),
        HandType(category="flush", primary_value=14, kickers=[])
    ),
    (
        HandType(category="three_of_a_kind", primary_value=4, kickers=[14, 13]),
        HandType(category="three_of_a_kind", primary_value=3, kickers=[14, 13])
    ),
    (
        HandType(category="two_pair", primary_value=12, kickers=[3, 14]),
        HandType(category="two_pair", primary_value=12, kickers=[3, 2])
    ),
    (
        HandType(category="high_card", primary_value=14, kickers=[13, 12, 11, 9]),
        HandType(category="high_card", primary_value=13, kickers=[12, 11, 10, 9])
    )
)

for player_hand, opponent_hand in player_win_tests:
    result = oracle.compare_poker_hands(player_hand, opponent_hand)
    assert result == "player", f"Expected 'player' but was '{result}' in test with player hand type {[(player_hand.category, player_hand.primary_value, player_hand.kickers)]} and opponent hand type {[(opponent_hand.category, opponent_hand.primary_value, opponent_hand.kickers)]}"

opponent_win_tests = (
    (
        HandType(category="pair", primary_value=7, kickers=[10, 12, 14]),
        HandType(category="two_pair", primary_value=14, kickers=[3, 2])
    ),
    (
        HandType(category="flush", primary_value=9, kickers=[]),
        HandType(category="four_of_a_kind", primary_value=7, kickers=[12])
    )
)

for player_hand, opponent_hand in opponent_win_tests:
    result = oracle.compare_poker_hands(player_hand, opponent_hand)
    assert result == "opponent", f"Expected 'opponent' but was '{result}' in test with player hand type {[(player_hand.category, player_hand.primary_value, player_hand.kickers)]} and opponent hand type {[(opponent_hand.category, opponent_hand.primary_value, opponent_hand.kickers)]}"

tie_tests = (
    (
        HandType(category="straight", primary_value=14, kickers=[]),
        HandType(category="straight", primary_value=14, kickers=[])
    ),
    (
        HandType(category="flush", primary_value=8, kickers=[]),
        HandType(category="flush", primary_value=8, kickers=[])
    ),
)

for player_hand, opponent_hand in tie_tests:
    result = oracle.compare_poker_hands(player_hand, opponent_hand)
    assert result == "tie", f"Expected 'tie' but was '{result}' in test with player hand type {[(player_hand.category, player_hand.primary_value, player_hand.kickers)]} and opponent hand type {[(opponent_hand.category, opponent_hand.primary_value, opponent_hand.kickers)]}"

# Test Utility Matrix
public_cards = [Card("J", "hearts"), Card("J", "spades"), Card("A", "clubs")]
utility_matrix, hand_label_to_index = oracle.gen_utility_matrix(public_cards)

player_hand = [Card("J", "diamonds"), Card("3", "spades")]
player_hand_label = HandLabelGenerator.get_hand_label(player_hand)
opponent_hand = [Card("2", "hearts"), Card("3", "spades")]
opponent_hand_label = HandLabelGenerator.get_hand_label(opponent_hand)
assert utility_matrix[hand_label_to_index[player_hand_label], hand_label_to_index[opponent_hand_label]] == 0

player_hand = [Card("J", "diamonds"), Card("3", "spades")]
player_hand_label = HandLabelGenerator.get_hand_label(player_hand)
opponent_hand = [Card("2", "hearts"), Card("8", "diamonds")]
opponent_hand_label = HandLabelGenerator.get_hand_label(opponent_hand)
assert utility_matrix[hand_label_to_index[player_hand_label], hand_label_to_index[opponent_hand_label]] == 1

player_hand = [Card("A", "diamonds"), Card("7", "hearts")]
player_hand_label = HandLabelGenerator.get_hand_label(player_hand)
opponent_hand = [Card("J", "clubs"), Card("8", "diamonds")]
opponent_hand_label = HandLabelGenerator.get_hand_label(opponent_hand)
assert utility_matrix[hand_label_to_index[player_hand_label], hand_label_to_index[opponent_hand_label]] == -1

# Test Rollouts
player_hand = [Card("2", "hearts"), Card("3", "spades")]
public_cards = []
result = oracle.perform_rollouts(player_hand, public_cards)
print(result)
