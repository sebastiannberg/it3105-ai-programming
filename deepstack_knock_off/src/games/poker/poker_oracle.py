from typing import List, Tuple, Optional, Dict
import numpy as np
import itertools
import time
from collections import Counter
from random import sample

from games.poker.utils.deck import Deck
from games.poker.utils.card import Card
from games.poker.utils.hand_type import HandType
from games.poker.utils.hand_label_generator import HandLabelGenerator
from games.poker.players.player import Player


class PokerOracle:

    rank_to_value_mapping = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        "7": 7, "8": 8, "9": 9, "10": 10,
        "J": 11, "Q": 12, "K": 13, "A": 14
    }
    hand_type_ranking = {
        "royal_flush": 10, "straight_flush": 9, "four_of_a_kind": 8,
        "full_house": 7, "flush": 6, "straight": 5,
        "three_of_a_kind": 4, "two_pair": 3, "pair": 2,
        "high_card": 1
    }

    # Caching frequently calculated variables
    _cache_possible_hands: List[Tuple[Card, Card]]
    _cache_hand_label_to_index: Dict[str, int] = {}
    _cache_index_to_hand_label: List[str] = []
    _cache_deck_size: int = 0

    @staticmethod
    def gen_deck(num_cards: int, shuffled: bool) -> Deck:
        if num_cards % 4:
            raise ValueError(f"The number of cards must be divisible evenly by 4 to form a complete deck but was {num_cards}")
        if num_cards < 24:
            raise ValueError(f"The number of cards must be 24 or greater but was {num_cards}")
        ranks = ("A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2")
        suits = ("spades", "hearts", "diamonds", "clubs")
        cards = []
        num_ranks = int(num_cards / 4)
        for rank in ranks[:num_ranks]:
            for suit in suits:
                cards.append(Card(rank, suit))
        deck = Deck(cards)
        if shuffled:
            deck.shuffle()
        return deck

    @staticmethod
    def get_possible_hands_with_indexing(deck_size: int):
        """Generates all possible poker hands from a deck of a specified size, and maintains a mapping
        of these hands to unique indices. This function uses caching to avoid redundant calculations
        when called multiple times with the same deck size.

        Returns
        -------
            A tuple containing three elements:
                - A list of tuples, where each tuple represents a possible hand (combination of two cards).
                - A dictionary mapping each hand's label (as determined by `HandLabelGenerator.get_hand_label`) to its index.
                - A list where each index corresponds to a hand's label, allowing for reverse lookup from index to hand label.

        Purpose
        -------
            The function is designed to facilitate operations that require mapping hand labels to indices and vice versa,
            especially useful in scenarios where hands need to be represented as numerical indices for compatibility with
            data structures like numpy arrays. Caching is employed to optimize performance by storing the generated hands,
            their labels, and the mappings if the deck size has not changed between calls.
        """

        # Only do calculations again for a new deck_size
        if PokerOracle._cache_deck_size != deck_size:
            # Generate unshuffled deck
            deck = PokerOracle.gen_deck(num_cards=deck_size, shuffled=False)
            # Find all possible combinations of choose two
            possible_hands = list(itertools.combinations(deck.cards, 2))

            hand_label_to_index = {}
            index_to_hand_label = []

            for idx, hand in enumerate(possible_hands):
                # Get the hand label string
                label = HandLabelGenerator.get_hand_label(hand)
                # Associate the hand label with the index
                hand_label_to_index[label] = idx
                # Create reverse mapping from index to hand label
                index_to_hand_label.append(label)

            # Cache the new calculations
            PokerOracle._cache_possible_hands = possible_hands
            PokerOracle._cache_hand_label_to_index = hand_label_to_index
            PokerOracle._cache_index_to_hand_label = index_to_hand_label
            PokerOracle._cache_deck_size = deck_size
        return PokerOracle._cache_possible_hands, PokerOracle._cache_hand_label_to_index, PokerOracle._cache_index_to_hand_label

    @staticmethod
    def get_index_from_hand_label(hand_label: str, deck_size: int) -> int:
        _, hand_label_to_index, _ = PokerOracle.get_possible_hands_with_indexing(deck_size=deck_size)
        return hand_label_to_index.get(hand_label, None)

    @staticmethod
    def get_hand_label_from_index(index: int, deck_size: int) -> str:
        _, _, index_to_hand_label = PokerOracle.get_possible_hands_with_indexing(deck_size=deck_size)
        if 0 <= index < len(index_to_hand_label):
            return index_to_hand_label[index]
        raise ValueError(f"Index out of valid range: {index}")

    @staticmethod
    def gen_utility_matrix(public_cards: List[Card], deck_size: int):
        start_time = time.time()

        if len(public_cards) not in (3, 4, 5):
            raise ValueError("Length of public cards must be 3, 4 or 5 when generating utility matrix")

        public_cards_set = set((card.rank, card.suit) for card in public_cards)

        possible_hands, _, _ = PokerOracle.get_possible_hands_with_indexing(deck_size=deck_size)

        # Initialize utility matrix with zeros
        utility_matrix = np.zeros((len(possible_hands), len(possible_hands)), dtype=np.int8)

        for player_hand_index, player_hand in enumerate(possible_hands):
            # If a card in the player's hand is also in the set of public cards, go to the next player hand
            if any((player_card.rank, player_card.suit) in public_cards_set for player_card in player_hand):
                continue
            player_hand_type = PokerOracle.classify_poker_hand(list(player_hand), public_cards)

            for opponent_hand_index, opponent_hand in enumerate(possible_hands):
                # If a card in the opponent's hand is also in the set of public cards, go to next opponent hand
                if any((opponent_card.rank, opponent_card.suit) in public_cards_set for opponent_card in opponent_hand):
                    continue
                # If a card in the player's hand is also in the opponent's hand, go to the next opponent hand
                if any(card in opponent_hand for card in player_hand):
                    continue
                opponent_hand_type = PokerOracle.classify_poker_hand(list(opponent_hand), public_cards)

                result = PokerOracle.compare_poker_hands(player_hand_type, opponent_hand_type)
                if result == "player":
                    # If player wins, mark the corresponding cell with 1
                    utility_matrix[player_hand_index, opponent_hand_index] = 1
                elif result == "opponent":
                    # If opponent wins, mark the corresponding cell with -1
                    utility_matrix[player_hand_index, opponent_hand_index] = -1

        end_time = time.time()
        duration = end_time - start_time
        duration_minutes = duration / 60
        print(f"gen_utility_matrix took {duration_minutes:.2f} minutes to run")

        return utility_matrix

    @staticmethod
    def classify_poker_hand(hand: List[Card], public_cards: List[Card], player: Optional[Player] = None) -> HandType:
        # All cards length is 5, 6 or 7
        all_cards = hand + public_cards

        # Check for a Royal Flush
        rf_success, rf_value, rf_kickers = PokerOracle.is_royal_flush(all_cards)
        if rf_success:
            return HandType(category="royal_flush", primary_value=rf_value, kickers=rf_kickers, player=player)

        # Check for a Straight Flush
        sf_success, sf_value, sf_kickers = PokerOracle.is_straight_flush(all_cards)
        if sf_success:
            return HandType(category="straight_flush", primary_value=sf_value, kickers=sf_kickers, player=player)

        # Check for a Four of a Kind
        foak_success, foak_value, foak_kickers = PokerOracle.is_four_of_a_kind(all_cards)
        if foak_success:
            return HandType(category="four_of_a_kind", primary_value=foak_value, kickers=foak_kickers, player=player)

        # Check for a Full House
        fh_success, fh_value, fh_kickers = PokerOracle.is_full_house(all_cards)
        if fh_success:
            return HandType(category="full_house", primary_value=fh_value, kickers=fh_kickers, player=player)

        # Check for a Flush
        f_success, f_value, f_kickers = PokerOracle.is_flush(all_cards)
        if f_success:
            return HandType(category="flush", primary_value=f_value, kickers=f_kickers, player=player)

        # Check for a Straight
        s_success, s_value, s_kickers =  PokerOracle.is_straight(all_cards)
        if s_success:
            return HandType(category="straight", primary_value=s_value, kickers=s_kickers, player=player)

        # Check for Three of a Kind
        toak_success, toak_value, toak_kickers = PokerOracle.is_three_of_a_kind(all_cards)
        if toak_success:
            return HandType(category="three_of_a_kind", primary_value=toak_value, kickers=toak_kickers, player=player)

        # Check for Two Pair
        tp_success, tp_value, tp_kickers = PokerOracle.is_two_pair(all_cards)
        if tp_success:
            return HandType(category="two_pair", primary_value=tp_value, kickers=tp_kickers, player=player)

        # Check for Pair
        p_success, p_value, p_kickers = PokerOracle.is_pair(all_cards)
        if p_success:
            return HandType(category="pair", primary_value=p_value, kickers=p_kickers, player=player)

        # Check for High Card
        hc_success, hc_value, hc_kickers = PokerOracle.is_high_card(all_cards)
        if hc_success:
            return HandType(category="high_card", primary_value=hc_value, kickers=hc_kickers, player=player)

        raise ValueError("Failure when classifying poker hand")

    @staticmethod
    def is_high_card(cards: List[Card]) -> Tuple[bool, int, List[int]]:
        if len(cards) < 5:
            raise ValueError(f"Unexpected behaviour, less than 5 cards: {len(cards)} cards")
        cards.sort(key=lambda card: PokerOracle.rank_to_value_mapping[card.rank], reverse=True)
        high_card = PokerOracle.rank_to_value_mapping[cards[0].rank]
        kickers = [PokerOracle.rank_to_value_mapping[card.rank] for card in cards[1:5]]
        return True, high_card, kickers

    @staticmethod
    def is_pair(cards: List[Card]) -> Tuple[bool, int, List[int]]:
        card_values = [PokerOracle.rank_to_value_mapping[card.rank] for card in cards]
        value_counter = Counter(card_values)
        pairs = [value for value, count in value_counter.items() if count == 2]

        # Identifies if there's exactly one pair among the cards, optimal for up to 7 cards
        if len(pairs) == 1:
            pair_value = pairs[0]
            kickers = sorted([value for value in card_values if value != pair_value], reverse=True)[:3]
            return True, pair_value, kickers
        return False, 0, []

    @staticmethod
    def is_two_pair(cards: List[Card]) -> Tuple[bool, int, List[int]]:
        card_values = [PokerOracle.rank_to_value_mapping[card.rank] for card in cards]
        value_counter = Counter(card_values)
        pairs = [value for value, count in value_counter.items() if count == 2]
        # In theory possible to have three pairs when seven cards
        if len(pairs) >= 2:
            # Sort pairs to have the highest pair values first
            pairs.sort(reverse=True)
            highest_pair, second_highest_pair = pairs[:2]
            # Exclude all cards that are part of any pair for kicker consideration
            kicker_candidates = [value for value in card_values if value not in pairs]
            # Sort remaining values to find the highest for the kicker
            kicker_candidates.sort(reverse=True)
            kickers = kicker_candidates[:1]  # There should always be at least one kicker if there are additional cards

            if not kickers:  # Safety check to ensure there is always a kicker
                kickers = [min(card_values)]  # Use the lowest card value as a fallback kicker

            # Return highest pair as primary value, second highest in kickers along with the top kicker
            return True, highest_pair, [second_highest_pair] + kickers

        return False, 0, []

    @staticmethod
    def is_three_of_a_kind(cards: List[Card]) -> Tuple[bool, int, List[int]]:
        rank_counter = Counter(card.rank for card in cards)
        threes = [(rank, count) for rank, count in rank_counter.items() if count == 3]
        # In theory possible to have two three of a kinds when seven cards
        if threes:
            # Sort them by their rank's value, highest first
            threes.sort(key=lambda x: PokerOracle.rank_to_value_mapping[x[0]], reverse=True)
            highest_three_of_a_kind_rank, _ = threes[0]

            three_of_a_kind_value = PokerOracle.rank_to_value_mapping[highest_three_of_a_kind_rank]

            # Exclude cards of the highest Three of a Kind rank, then take the two highest values as kickers
            kickers = sorted([PokerOracle.rank_to_value_mapping[card.rank] for card in cards if card.rank != highest_three_of_a_kind_rank], reverse=True)[:2]
            return True, three_of_a_kind_value, kickers

        return False, 0, []

    @staticmethod
    def is_straight(cards: List[Card]) -> Tuple[bool, int, List[int]]:
        values = sorted({PokerOracle.rank_to_value_mapping[card.rank] for card in cards})

        # Add a check for the low Ace straight (A-2-3-4-5)
        if set([14, 2, 3, 4, 5]).issubset(values):
            # Adding '1' to represent Ace as the low card for straight calculation
            values.append(1)
            values.sort()

        consecutive_count = 1
        max_straight_high_card = 0

        for i in range(1, len(values)):
            # Check if the current card is consecutive to the previous
            if values[i] - values[i - 1] == 1:
                consecutive_count += 1
                # If we've found 5 consecutive cards, update the highest card of the straight found so far
                if consecutive_count >= 5:
                    max_straight_high_card = values[i]
            else:
                # Reset the count if the sequence is broken
                consecutive_count = 1

        if max_straight_high_card > 0:
            return True, max_straight_high_card, []
        else:
            return False, 0, []

    @staticmethod
    def is_flush(cards: List[Card]) -> Tuple[bool, int, List[int]]:
        suit_counter = Counter(card.suit for card in cards)
        for suit, count in suit_counter.items():
            if count >= 5:
                flush_cards = [card for card in cards if card.suit == suit]
                # Sort in descending order on values
                flush_cards.sort(key=lambda card: PokerOracle.rank_to_value_mapping[card.rank], reverse=True)
                high_card = PokerOracle.rank_to_value_mapping[flush_cards[0].rank]
                return True, high_card, []
        return False, 0, []

    @staticmethod
    def is_full_house(cards: List[Card]) -> Tuple[bool, int, List[int]]:
        rank_counter = Counter(card.rank for card in cards)
        # Extract ranks that have three or more occurrences
        three_counts = {rank for rank, count in rank_counter.items() if count >= 3}
        # Pairs could also be part of another Three of a Kind not selected as the primary Three of a Kind
        pair_counts = {rank for rank, count in rank_counter.items() if count == 2 or (count >= 3 and rank not in three_counts)}

        if three_counts:
            # Convert ranks to values and sort to find the highest Three of a Kind
            three_values = sorted([PokerOracle.rank_to_value_mapping[rank] for rank in three_counts], reverse=True)
            highest_three_value = three_values[0]

            # Handle case where there's another Three of a Kind or a Pair
            if len(three_values) > 1 or pair_counts:
                # If there's a second Three of a Kind, use it as the pair for the Full House
                # Otherwise, use the highest pair available
                second_set_value = three_values[1] if len(three_values) > 1 else max(PokerOracle.rank_to_value_mapping[rank] for rank in pair_counts)
                return True, highest_three_value, [second_set_value]

        return False, 0, []

    @staticmethod
    def is_four_of_a_kind(cards: List[Card]) -> Tuple[bool, int, List[int]]:
        rank_counter = Counter(card.rank for card in cards)
        fours = [(rank, count) for rank, count in rank_counter.items() if count == 4]
        if fours:
            four_of_a_kind_rank = fours[0][0]
            four_of_a_kind_value = PokerOracle.rank_to_value_mapping[four_of_a_kind_rank]

            # Find the highest card outside the Four of a Kind to serve as the kicker
            kickers = sorted([PokerOracle.rank_to_value_mapping[card.rank] for card in cards if card.rank != four_of_a_kind_rank], reverse=True)[:1]

            return True, four_of_a_kind_value, kickers

        return False, 0, []

    @staticmethod
    def is_straight_flush(cards: List[Card]) -> Tuple[bool, int, List[int]]:
        suits = {card.suit for card in cards}
        for suit in suits:
            suited_cards = [card for card in cards if card.suit == suit]
            if len(suited_cards) < 5:
                # Not enough cards of the same suit for a straight flush
                continue
            # Now check if these suited cards form a straight
            s_success, s_value, _ = PokerOracle.is_straight(suited_cards)
            if s_success:
                return True, s_value, []
        return False, 0, []

    @staticmethod
    def is_royal_flush(cards: List[Card]) -> Tuple[bool, int, List[int]]:
        sf_success, sf_value, _ = PokerOracle.is_straight_flush(cards)
        if sf_success and sf_value == 14: # High card of Ace indicates a royal flush
            return True, sf_value, []
        return False, 0, []

    @staticmethod
    def compare_poker_hands(player_hand: HandType, opponent_hand: HandType) -> str:
        """
        Returns 'player' if player_hand beats the opponent_hand,
        'opponent' if opponent_hand beats the player_hand,
        and 'tie' if neither hand wins.
        """
        # Compare the hand types based on their ranking
        player_rank = PokerOracle.hand_type_ranking[player_hand.category]
        opponent_rank = PokerOracle.hand_type_ranking[opponent_hand.category]

        if player_rank > opponent_rank:
            return "player"
        elif player_rank < opponent_rank:
            return "opponent"
        else:
            # If the hand types are the same, further comparison is needed
            if player_hand.primary_value > opponent_hand.primary_value:
                return "player"
            elif player_hand.primary_value < opponent_hand.primary_value:
                return "opponent"
            else:
                # Ensure both hands have the same number of kickers before comparing
                if len(player_hand.kickers) != len(opponent_hand.kickers):
                    raise ValueError(f"Kicker lists are not of the same size. {player_hand.category}, {opponent_hand.category}, {player_hand.kickers}, {opponent_hand.kickers}")

                # Kickers decide outcome
                # Compare kickers sequentially from highest to lowest
                for player_kicker, opponent_kicker in zip(player_hand.kickers, opponent_hand.kickers):
                    if player_kicker > opponent_kicker:
                        return "player"
                    elif player_kicker < opponent_kicker:
                        return "opponent"
                # If all primary values and kickers are equal, it's a tie
                return "tie"

    @staticmethod
    def perform_rollouts(player_hand: List[Card], public_cards: List[Card], num_cards_deck: int,  num_opponent_players: int = 1, num_rollouts: int = 5000):
        if len(player_hand) != 2:
            raise ValueError("Length of player hand should be 2 when performing rollouts")

        wins, ties, losses = 0, 0, 0
        for _ in range(num_rollouts):
            # Generate new deck for rollout
            deck = PokerOracle.gen_deck(num_cards=num_cards_deck, shuffled=True)
            # Create a set for easy comparison
            player_and_public_cards_set = {(card.rank, card.suit) for card in player_hand + public_cards}
            # Remove player cards and public cards from deck
            deck.cards = [card for card in deck.cards if (card.rank, card.suit) not in player_and_public_cards_set]

            # Deal hands to opponents
            opponent_cards = sample(deck.cards, 2 * num_opponent_players)
            deck.cards = [card for card in deck.cards if card not in opponent_cards]
            opponent_hands = [opponent_cards[i:i+2] for i in range(0, len(opponent_cards), 2)]

            # Deal additional public cards if necessary
            additional_cards_needed = 5 - len(public_cards)
            final_board = public_cards + sample(deck.cards, additional_cards_needed) if additional_cards_needed > 0 else public_cards

            # Evaluate the player's hand against each opponent
            player_hand_type = PokerOracle.classify_poker_hand(player_hand, final_board)
            all_hands = [PokerOracle.classify_poker_hand(hand, final_board) for hand in opponent_hands] + [player_hand_type]

            # Determine the best hand and possible wins or ties
            best_hand = max(all_hands, key=lambda hand: (PokerOracle.hand_type_ranking[hand.category], hand.primary_value, hand.kickers))
            winner_count = 0
            for hand in all_hands:
                if hand.category == best_hand.category and hand.primary_value == best_hand.primary_value and hand.kickers == best_hand.kickers:
                    winner_count += 1

            if player_hand_type.category == best_hand.category and player_hand_type.primary_value == best_hand.primary_value and player_hand_type.kickers == best_hand.kickers:
                if winner_count == 1:  # No ties, player wins
                    wins += 1
                else:  # Player's hand is part of a tie
                    ties += 1
            else:
                losses += 1

        # Calculate probabilities
        win_probability = wins / num_rollouts
        tie_probability = ties / num_rollouts
        lose_probability = losses / num_rollouts
        return win_probability, tie_probability, lose_probability

    @staticmethod
    def gen_cheat_sheet(deck_size: int):
        start_time = time.time()

        possible_hands, _, _ = PokerOracle.get_possible_hands_with_indexing(deck_size)

        # Initialize cheat sheet to empty dictionary
        cheat_sheet = {}

        # Iterate over all possible hands
        for hand in possible_hands:
            # Calculate win probability for this hand
            win_prob, _, _ = PokerOracle.perform_rollouts(list(hand), public_cards=[], num_cards_deck=deck_size)

            hand_label = HandLabelGenerator.get_hand_label(hand)
            # Set cheat sheet to win probability for this hand
            cheat_sheet[hand_label] = win_prob

        end_time = time.time()
        duration = end_time - start_time
        duration_minutes = duration / 60
        print(f"gen_cheat_sheet took {duration_minutes:.2f} minutes to run")

        return cheat_sheet
