from typing import List, Dict
import itertools

from games.poker.poker_oracle import PokerOracle
from games.poker.utils.card import Card


class HandLabelGenerator:
    _cache_hand_label_to_index: Dict[str, int] = {}
    _cache_index_to_hand_label: List[str] = []
    _cache_deck_size: int = 0

    @staticmethod
    def get_possible_hands_with_indexing(deck_size: int):
        if HandLabelGenerator._cache_deck_size != deck_size:
            deck = PokerOracle.gen_deck(num_cards=deck_size, shuffled=False)
            possible_hands = list(itertools.combinations(deck.cards, 2))

            hand_label_to_index = {}
            index_to_hand_label = []

            for idx, hand in enumerate(possible_hands):
                label = HandLabelGenerator.get_hand_label(hand)
                hand_label_to_index[label] = idx
                index_to_hand_label.append(label)

            HandLabelGenerator._cache_hand_label_to_index = hand_label_to_index
            HandLabelGenerator._cache_index_to_hand_label = index_to_hand_label
            HandLabelGenerator._cache_deck_size = deck_size
        return possible_hands, HandLabelGenerator._cache_hand_label_to_index, HandLabelGenerator._cache_index_to_hand_label

    @staticmethod
    def get_hand_label(cards: List[Card]) -> str:
        if not all(isinstance(card, Card) for card in cards):
            raise ValueError("All items in the list must be instances of Card")
        sorted_hand = tuple(sorted(cards, key=lambda card: (card.rank, card.suit)))
        return "".join([f"{card.rank}{card.suit}" for card in sorted_hand])

    @staticmethod
    def get_index_from_hand_label(hand_label: str, deck_size: int) -> int:
        _, hand_label_to_index, _ = HandLabelGenerator.get_possible_hands_with_indexing(deck_size=deck_size)
        return hand_label_to_index.get(hand_label, None)

    @staticmethod
    def get_hand_label_from_index(index: int, deck_size: int) -> str:
        """
        Returns the hand label corresponding to the given index
        """
        _, _, index_to_hand_label = HandLabelGenerator.get_possible_hands_with_indexing(deck_size=deck_size)
        if 0 <= index < len(index_to_hand_label):
            return index_to_hand_label[index]
        raise ValueError(f"Index out of valid range: {index}")
