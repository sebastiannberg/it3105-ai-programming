from typing import List, Tuple
import pandas as pd
import itertools
import time
from collections import Counter

from games.poker.utils.deck import Deck
from games.poker.utils.card import Card
from games.poker.utils.hand_type import HandType

class PokerOracle:

    def __init__(self):
        self.rank_to_value_mapping = {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14
        }

    def gen_deck(self, num_cards: int, shuffled: bool) -> Deck:
        if num_cards % 4:
            raise ValueError(f"The number of cards must be divisible evenly by 4 to form a complete deck but was {num_cards}")
        num_ranks = int(num_cards / 4)
        if num_ranks < 5:
            raise ValueError(f"The number of ranks need to be at least 5 but was {num_ranks}")
        ranks = ("A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2")
        suits = ("spades", "hearts", "diamonds", "clubs")
        cards = []
        for rank in ranks[:num_ranks]:
            for suit in suits:
                cards.append(Card(rank, suit))
        deck = Deck(cards)
        if shuffled:
            deck.shuffle()
        return deck

    def gen_utility_matrix(self, public_cards: List[Card]) -> pd.DataFrame:
        print("Started gen_utility_matrix")
        start_time = time.time()

        if len(public_cards) not in (3, 4, 5):
            raise ValueError("Length of public cards must be 3, 4 or 5 when generating utility matrix")

        public_cards_set = set((card.rank, card.suit) for card in public_cards)

        deck = self.gen_deck(num_cards=52, shuffled=False)
        # Possible combination of 2 cards in a deck
        possible_hands = list(itertools.combinations(deck.cards, 2))
        hand_labels = ["".join([f"{card.rank}{card.suit}" for card in hand]) for hand in possible_hands]

        # Initialize utility matrix with zeros
        utility_matrix = pd.DataFrame(0, index=hand_labels, columns=hand_labels)

        for player_hand_index, player_hand in enumerate(possible_hands):
            print(f"\rPlayer hand {player_hand_index}", end="")
            # If a card in the player's hand is also in the set of public cards, go to the next player hand
            if any((player_card.rank, player_card.suit) in public_cards_set for player_card in player_hand):
                continue
            player_hand_type = self.classify_poker_hand(list(player_hand), public_cards)
            player_hand_label = hand_labels[player_hand_index]


            for opponent_hand_index, opponent_hand in enumerate(possible_hands):
                # If a card in the opponent's hand is also in the set of public cards, go to next opponent hand
                if any((opponent_card.rank, opponent_card.suit) in public_cards_set for opponent_card in opponent_hand):
                    continue
                # If a card in the player's hand is alsi in the opponent's hand, go to the next opponent hand
                if any(card in opponent_hand for card in player_hand):
                    continue

                opponent_hand_type = self.classify_poker_hand(list(opponent_hand), public_cards)
                opponent_hand_label = hand_labels[opponent_hand_index]

                result = self.compare_poker_hands(player_hand_type, opponent_hand_type)
                if result == "player":
                    # If player wins, mark the corresponding cell with 1
                    utility_matrix.loc[player_hand_label, opponent_hand_label] = 1
                elif result == "opponent":
                    # If opponent wins, mark the corresponding cell with -1
                    utility_matrix.loc[player_hand_label, opponent_hand_label] = -1
                # If tie just let it be 0
        print()
        end_time = time.time()
        duration = end_time - start_time
        duration_minutes = duration / 60
        print(f"gen_utility_matrix took {duration_minutes:.2f} minutes to run")

        return utility_matrix

    def classify_poker_hand(self, hand: List[Card], public_cards: List[Card]) -> HandType:
        # All cards length is 5, 6 or 7
        all_cards = hand + public_cards

        # Check for a Royal Flush
        rf_success, rf_value, rf_kickers = self.is_royal_flush(all_cards)
        if rf_success:
            return HandType(category="royal_flush", primary_value=rf_value, kickers=rf_kickers)

        # Check for a Straight Flush
        sf_success, sf_value, sf_kickers = self.is_straight_flush(all_cards)
        if sf_success:
            return HandType(category="straight_flush", primary_value=sf_value, kickers=sf_kickers)

        # Check for a Four of a Kind
        foak_success, foak_value, foak_kickers = self.is_four_of_a_kind(all_cards)
        if foak_success:
            return HandType(category="four_of_a_kind", primary_value=foak_value, kickers=foak_kickers)

        # Check for a Full House
        fh_success, fh_value, fh_kickers = self.is_full_house(all_cards)
        if fh_success:
            return HandType(category="full_house", primary_value=fh_value, kickers=fh_kickers)

        # Check for a Flush
        f_success, f_value, f_kickers = self.is_flush(all_cards)
        if f_success:
            return HandType(category="flush", primary_value=f_value, kickers=f_kickers)

        # Check for a Straight
        s_success, s_value, s_kickers =  self.is_straight(all_cards)
        if s_success:
            return HandType(category="straight", primary_value=s_value, kickers=s_kickers)

        # Check for Three of a Kind
        toak_success, toak_value, toak_kickers = self.is_three_of_a_kind(all_cards)
        if toak_success:
            return HandType(category="three_of_a_kind", primary_value=toak_value, kickers=toak_kickers)

        # Check for Two Pair
        tp_success, tp_value, tp_kickers = self.is_two_pair(all_cards)
        if tp_success:
            return HandType(category="two_pair", primary_value=tp_value, kickers=tp_kickers)

        # Check for Pair
        p_success, p_value, p_kickers = self.is_pair(all_cards)
        if p_success:
            return HandType(category="pair", primary_value=p_value, kickers=p_kickers)

        # Check for High Card
        hc_success, hc_value, hc_kickers = self.is_high_card(all_cards)
        if hc_success:
            return HandType(category="high_card", primary_value=hc_value, kickers=hc_kickers)

        raise ValueError("Failure when classifying poker hand")

    def is_high_card(self, cards: List[Card]) -> Tuple[bool, int, List[int]]:
        if len(cards) < 5:
            raise ValueError(f"Unexpected behaviour, less than 5 cards: {len(cards)} cards")
        cards.sort(key=lambda card: self.rank_to_value_mapping[card.rank], reverse=True)
        high_card = self.rank_to_value_mapping[cards[0].rank]
        kickers = [self.rank_to_value_mapping[card.rank] for card in cards[1:5]]
        return True, high_card, kickers

    def is_pair(self, cards: List[Card]) -> Tuple[bool, int, List[int]]:
        card_values = [self.rank_to_value_mapping[card.rank] for card in cards]
        value_counter = Counter(card_values)
        pairs = [value for value, count in value_counter.items() if count == 2]

        # Identifies if there's exactly one pair among the cards, optimal for up to 7 cards
        if len(pairs) == 1:
            pair_value = pairs[0]
            kickers = sorted([value for value in card_values if value != pair_value], reverse=True)[:3]
            return True, pair_value, kickers
        return False, 0, []

    def is_two_pair(self, cards: List[Card]) -> Tuple[bool, int, List[int]]:
        card_values = [self.rank_to_value_mapping[card.rank] for card in cards]
        value_counter = Counter(card_values)
        pairs = [value for value, count in value_counter.items() if count == 2]
        # In theory possible to have three pairs when seven cards
        if len(pairs) >= 2:
            # Sort pairs to have the highest pair values first
            pairs.sort(reverse=True)
            highest_pair, second_highest_pair = pairs[:2]
            kickers = sorted([value for value in card_values if value not in pairs], reverse=True)[:1]
            # Consider the highest pair as the primary value and second highest in the kickers
            return True, highest_pair, [second_highest_pair] + kickers
        return False, 0, []

    def is_three_of_a_kind(self, cards: List[Card]) -> Tuple[bool, int, List[int]]:
        rank_counter = Counter(card.rank for card in cards)
        threes = [(rank, count) for rank, count in rank_counter.items() if count == 3]
        # In theory possible to have two three of a kinds when seven cards
        if threes:
            # Sort them by their rank's value, highest first
            threes.sort(key=lambda x: self.rank_to_value_mapping[x[0]], reverse=True)
            highest_three_of_a_kind_rank, _ = threes[0]

            three_of_a_kind_value = self.rank_to_value_mapping[highest_three_of_a_kind_rank]

            # Exclude cards of the highest Three of a Kind rank, then take the two highest values as kickers
            kickers = sorted([self.rank_to_value_mapping[card.rank] for card in cards if card.rank != highest_three_of_a_kind_rank], reverse=True)[:2]
            return True, three_of_a_kind_value, kickers

        return False, 0, []

    def is_straight(self, cards: List[Card]) -> Tuple[bool, int, List[int]]:
        values = sorted({self.rank_to_value_mapping[card.rank] for card in cards})

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

    def is_flush(self, cards: List[Card]) -> Tuple[bool, int, List[int]]:
        suit_counter = Counter(card.suit for card in cards)
        for suit, count in suit_counter.items():
            if count >= 5:
                flush_cards = [card for card in cards if card.suit == suit]
                # Sort in descending order on values
                flush_cards.sort(key=lambda card: self.rank_to_value_mapping[card.rank], reverse=True)
                high_card = self.rank_to_value_mapping[flush_cards[0].rank]
                return True, high_card, []
        return False, 0, []

    def is_full_house(self, cards: List[Card]) -> Tuple[bool, int, List[int]]:
        rank_counter = Counter(card.rank for card in cards)
        # Extract ranks that have three or more occurrences
        three_counts = {rank for rank, count in rank_counter.items() if count >= 3}
        # Pairs could also be part of another Three of a Kind not selected as the primary Three of a Kind
        pair_counts = {rank for rank, count in rank_counter.items() if count == 2 or (count >= 3 and rank not in three_counts)}

        if three_counts:
            # Convert ranks to values and sort to find the highest Three of a Kind
            three_values = sorted([self.rank_to_value_mapping[rank] for rank in three_counts], reverse=True)
            highest_three_value = three_values[0]

            # Handle case where there's another Three of a Kind or a Pair
            if len(three_values) > 1 or pair_counts:
                # If there's a second Three of a Kind, use it as the pair for the Full House
                # Otherwise, use the highest pair available
                second_set_value = three_values[1] if len(three_values) > 1 else max(self.rank_to_value_mapping[rank] for rank in pair_counts)
                return True, highest_three_value, [second_set_value]

        return False, 0, []

    def is_four_of_a_kind(self, cards: List[Card]) -> Tuple[bool, int, List[int]]:
        rank_counter = Counter(card.rank for card in cards)
        fours = [(rank, count) for rank, count in rank_counter.items() if count == 4]
        if fours:
            four_of_a_kind_rank = fours[0][0]
            four_of_a_kind_value = self.rank_to_value_mapping[four_of_a_kind_rank]

            # Find the highest card outside the Four of a Kind to serve as the kicker
            kickers = sorted([self.rank_to_value_mapping[card.rank] for card in cards if card.rank != four_of_a_kind_rank], reverse=True)[:1]

            return True, four_of_a_kind_value, kickers

        return False, 0, []

    def is_straight_flush(self, cards: List[Card]) -> Tuple[bool, int, List[int]]:
        suits = {card.suit for card in cards}
        for suit in suits:
            suited_cards = [card for card in cards if card.suit == suit]
            if len(suited_cards) < 5:
                # Not enough cards of the same suit for a straight flush
                continue
            # Now check if these suited cards form a straight
            s_success, s_value, _ = self.is_straight(suited_cards)
            if s_success:
                return True, s_value, []
        return False, 0, []

    def is_royal_flush(self, cards: List[Card]) -> Tuple[bool, int, List[int]]:
        sf_success, sf_value, _ = self.is_straight_flush(cards)
        if sf_success and sf_value == 14: # High card of Ace indicates a royal flush
            return True, sf_value, []
        return False, 0, []

    def compare_poker_hands(self, player_hand: HandType, opponent_hand: HandType) -> str:
        """
        Returns 'player' if player_hand beats the opponent_hand,
        'opponent' if opponent_hand beats the player_hand,
        and 'tie' if neither hand wins.
        """
        hand_type_ranking = {
            "royal_flush": 10,
            "straight_flush": 9,
            "four_of_a_kind": 8,
            "full_house": 7,
            "flush": 6,
            "straight": 5,
            "three_of_a_kind": 4,
            "two_pair": 3,
            "pair": 2,
            "high_card": 1
        }

        # Compare the hand types based on their ranking
        player_rank = hand_type_ranking[player_hand.category]
        opponent_rank = hand_type_ranking[opponent_hand.category]

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
                    raise ValueError("Kicker lists are not of the same size.")

                # Kickers decide outcome
                # Compare kickers sequentially from highest to lowest
                for player_kicker, opponent_kicker in zip(player_hand.kickers, opponent_hand.kickers):
                    if player_kicker > opponent_kicker:
                        return "player"
                    elif player_kicker < opponent_kicker:
                        return "opponent"
                # If all primary values and kickers are equal, it's a tie
                return "tie"
