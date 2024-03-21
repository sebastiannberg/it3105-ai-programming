from games.poker.poker_oracle import PokerOracle
from games.poker.utils.card import Card

public_cards = [Card("10", "spades"), Card("2", "clubs"), Card("K", "hearts")]


oracle = PokerOracle()
oracle.gen_utility_matrix(public_cards=public_cards)
