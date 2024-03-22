from typing import Dict, List, Optional

from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle
from games.poker.poker_state import PokerState
from games.poker.players.player import Player
from games.poker.players.ai_player import AIPlayer
from games.poker.players.human_player import HumanPlayer
from games.poker.actions.raise_bet import RaiseBet


class PokerGameManager:

    def __init__(self, rules: Dict, oracle: PokerOracle):
        self.rules = rules
        self.oracle = oracle

    def init_poker_game(self):
        deck = self.oracle.gen_deck(self.rules["deck_size"], shuffled=True)
        players = self.gen_poker_players(num_ai_players=self.rules["num_ai_players"], num_human_players=self.rules["num_human_players"])
        init_state = PokerStateManager.gen_init_state(players, deck, self.rules["small_blind_amount"], self.rules["big_blind_amount"])
        self.game: PokerState = init_state

    def start_game(self):
        # TODO
        pass

    def gen_poker_players(self, num_ai_players, num_human_players):
        if num_ai_players + num_human_players > 6:
            raise ValueError("Total amount of players should be less than or equal to 6")
        if num_ai_players + num_human_players < 2:
            raise ValueError("Total amount of players should be at least 2")
        players = []
        initial_chips = self.rules["initial_chips"]
        # Generate AI players
        for i in range(num_ai_players):
            players.append(AIPlayer(name=f"AI Player {i+1}", initial_chips=initial_chips))
        # Generate human players
        for i in range(num_human_players):
            players.append(HumanPlayer(name=f"Human Player {i+1}", initial_chips=initial_chips))
        return players

    def find_round_player_by_name(self, player_name: str) -> Player:
        for player in self.game.round_players:
            if player.name == player_name:
                return player
        return None

    def assign_blind_roles(self):
        current_small_blind_player = self.game.small_blind_player
        current_big_blind_player = self.game.big_blind_player
        players = self.game.round_players

        if current_small_blind_player and current_big_blind_player:
            small_blind_index = players.index(current_small_blind_player)

            next_small_blind_index = (small_blind_index + 1) % len(players)
            next_small_blind_player = players[next_small_blind_index]

            next_big_blind_index = (next_small_blind_index + 1) % len(players)
            next_big_blind_player = players[next_big_blind_index]
        else:
            next_small_blind_player = players[0]
            next_big_blind_player = players[1]

        self.game.small_blind_player = next_small_blind_player
        self.game.big_blind_player = next_big_blind_player


    def perform_blind_bets(self):
        current_small_blind_player = self.game.small_blind_player
        current_big_blind_player = self.game.big_blind_player
        if not current_small_blind_player or not current_big_blind_player:
            raise ValueError("Either small blind or big blind is not assigned to a player")
        # Small blind action
        raise_amount = (current_small_blind_player.player_bet + self.game.small_blind_amount) - self.game.current_bet
        small_blind_action = RaiseBet(player=current_small_blind_player,
                                      chip_cost=self.game.small_blind_amount,
                                      raise_amount=raise_amount,
                                      raise_type="small_blind")
        PokerStateManager.apply_action(self.game, current_small_blind_player, small_blind_action)
        # Big blind action
        raise_amount = (current_big_blind_player.player_bet + self.game.big_blind_amount) - self.game.current_bet
        big_blind_action = RaiseBet(player=current_big_blind_player,
                                    chip_cost=self.game.big_blind_amount,
                                    raise_amount=raise_amount,
                                    raise_type="big_blind")
        PokerStateManager.apply_action(self.game, current_big_blind_player, big_blind_action)

    def deal_cards(self):
        if self.game.stage == "preflop":
            for player in self.game.round_players:
                cards = self.game.deck.deal_cards(num_cards=2)
                player.receive_cards(*cards)
        elif self.game.stage == "flop":
            cards = self.game.deck.deal_cards(num_cards=3)
            self.game.public_cards.extend(cards)
        elif self.game.stage == "turn" or self.game.stage == "river":
            cards = self.game.deck.deal_cards(num_cards=1)
            self.game.public_cards.extend(cards)

    def assign_active_player(self, stage_change=False):
        if not self.game.active_player:
            # Assign player after big blind
            current_big_blind_player = self.game.big_blind_player
            big_blind_index = self.game.round_players.index(current_big_blind_player)
            next_player_index = (big_blind_index + 1) % len(self.game.round_players)
            self.game.active_player = self.game.round_players[next_player_index]
        elif stage_change:
            # Assign first player after dealer
            if not self.game.small_blind_player.has_folded:
                self.game.active_player = self.game.small_blind_player
            else:
                small_blind_index = self.game.game_players.index(self.game.small_blind_player)
                next_player_index = (small_blind_index + 1) % len(self.game.game_players)

                while self.game.game_players[next_player_index].has_folded:
                    next_player_index = (next_player_index + 1) % len(self.game.game_players)
                    if next_player_index == current_active_player_index:
                        break

                self.game.active_player = self.game.game_players[next_player_index]
        else:
            # Assign player after current player
            current_active_player = self.game.active_player
            current_active_player_index = self.game.game_players.index(current_active_player)
            next_player_index = (current_active_player_index + 1) % len(self.game.game_players)

            while self.game.game_players[next_player_index].has_folded:
                next_player_index = (next_player_index + 1) % len(self.game.game_players)
                if next_player_index == current_active_player_index:
                    break

            self.game.active_player = self.game.game_players[next_player_index]

    def check_for_proceed_stage(self):
        if self.game.stage == "preflop":
            if not any(player.last_raised for player in self.game.round_players):
                if self.game.big_blind_player.has_checked and (all(player.has_called or player.has_folded) for player in self.game.round_players):
                    return True
            else:
                # First, check if all players have checked
                all_checked = all(player.has_checked for player in self.game.round_players)
                # Then, check for the scenario where one player has raised and the others have called
                raised_and_called = False
                if any(player.last_raised for player in self.game.round_players):
                    # Find the player who last raised
                    last_raised_player = next(player for player in self.game.round_players if player.last_raised)
                    # Check if all other players have called
                    raised_and_called = all((player.has_called or player is last_raised_player) for player in self.game.round_players)
                if all_checked or raised_and_called:
                    return True
        else:
            # First, check if all players have checked
            all_checked = all(player.has_checked for player in self.game.round_players)
            # Then, check for the scenario where one player has raised and the others have called
            raised_and_called = False
            if any(player.last_raised for player in self.game.round_players):
                # Find the player who last raised
                last_raised_player = next(player for player in self.game.round_players if player.last_raised)
                # Check if all other players have called
                raised_and_called = all(player.has_called or player is last_raised_player for player in self.game.round_players)
            if all_checked or raised_and_called:
                return True

    def proceed_stage(self) -> Optional[List[Dict]]:
        self.game.current_bet = 0
        for player in self.game.round_players:
            player.player_bet = 0
            player.has_checked = False
            player.has_called = False
            player.last_raised = False

        winners = None
        if self.game.stage in ["preflop", "flop", "turn"]:
            # Advance the game stage
            stage_transitions = {"preflop": "flop", "flop": "turn", "turn": "river"}
            self.game.stage = stage_transitions.get(self.game.stage, self.game.stage)
            # Deal cards and assign the active player for the new stage
            self.deal_cards()
            self.assign_active_player(stage_change=True)
        elif self.game.stage == "river":
            # Showdown and determine winners
            winners = self.showdown()

        return winners

    def showdown(self) -> List[Dict]:
        # Classify each player's hand and associate it with the player
        classified_hands = {
            player: self.oracle.classify_poker_hand(player.hand, self.game.public_cards, player)
            for player in self.game.round_players
        }

        # Find the best hand to compare with others
        best_hand = max(classified_hands.values(), key=lambda hand: (
            self.oracle.hand_type_ranking[hand.category],
            hand.primary_value,
            hand.kickers
        ))

        # Filter winners based on the comparison, preserving original order
        winners = []
        for player, hand in classified_hands.items():
            comparison_result = self.oracle.compare_poker_hands(hand, best_hand)
            if comparison_result in ["player", "tie"]:
                winners.append(player)

        # Update round_players to include only the winners, preserving original order
        self.game.round_players = winners

        winners_details = [
            {
                "player": player.name,
                "hand_category": classified_hands[player].category,
                "primary_value": classified_hands[player].primary_value,
                "kickers": classified_hands[player].kickers
            }
            for player in winners
        ]
        return winners_details

    def check_for_early_round_winner(self):
        """
        Checks for winner and return player.
        If no winner it returns None
        """
        if len(self.game.round_players) == 1:
            return self.game.round_players[0]
        else:
            return None


    def process_winnings(self):
        winnings_per_player = self.game.pot // len(self.game.round_players)

        for player in self.game.round_players:
            player.chips += winnings_per_player

        # If the pot cannot be evenly divided, add remainder to pot
        total_distributed = winnings_per_player * len(self.game.round_players)
        self.game.pot -= total_distributed

    def end_round_next_round(self):
        """
        Ends the current round and starts the next round
        """
        # Reset every player in the game for a new round
        for player in self.game.game_players:
            player.ready_for_new_round()
        self.game.round_players = self.game.game_players.copy()
        # Reset state variables
        self.game.active_player = None
        self.game.current_bet = 0
        self.game.public_cards = []
        self.game.stage = "preflop"
        self.deck = self.oracle.gen_deck(num_cards=52, shuffled=True)
        # Init a new round
        self.assign_blind_roles()
        self.perform_blind_bets()
        self.deal_cards()
        self.assign_active_player()

    def remove_busted_players(self):
        for player in self.game.game_players:
            if player.chips <= 0:
                self.game.game_players.remove(player)

    def check_for_game_winner(self):
        if len(self.game.game_players) == 1:
            return self.game.game_players[0]
        else:
            return None

    def jsonify_poker_game(self):
        if self.game.game_players:
            game_players_dict = {}
            for player in self.game.game_players:
                game_players_dict[player.name] = {
                    "hand": [{"rank": card.rank, "suit": card.suit} for card in player.hand],
                    "chips": player.chips,
                    "player_bet": player.player_bet
                }
        else:
            game_players_dict = None

        if self.game.round_players:
            round_players_dict = {}
            for player in self.game.round_players:
                round_players_dict[player.name] = {
                    "hand": [{"rank": card.rank, "suit": card.suit} for card in player.hand],
                    "chips": player.chips,
                    "player_bet": player.player_bet
                }
        else:
            round_players_dict = None

        if self.game.deck:
            deck_dict = {"cards": [{"rank": card.rank, "suit": card.suit} for card in self.game.deck.cards]}
        else:
            deck_dict = None

        if self.game.small_blind_player and self.game.big_blind_player:
            small_blind_player_dict = {self.game.small_blind_player.name: {
                "hand": [{"rank": card.rank, "suit": card.suit} for card in self.game.small_blind_player.hand],
                "chips": self.game.small_blind_player.chips,
                "player_bet": self.game.small_blind_player.player_bet
            }}
            big_blind_player_dict = {self.game.big_blind_player.name: {
                "hand": [{"rank": card.rank, "suit": card.suit} for card in self.game.big_blind_player.hand],
                "chips": self.game.big_blind_player.chips,
                "player_bet": self.game.big_blind_player.player_bet
            }}
        else:
            small_blind_player_dict = None
            big_blind_player_dict = None

        if self.game.active_player:
            active_player_dict = {self.game.active_player.name: {
                "hand": [{"rank": card.rank, "suit": card.suit} for card in self.game.active_player.hand],
                "chips": self.game.active_player.chips,
                "player_bet": self.game.active_player.player_bet
            }}
        else:
            active_player_dict = None

        return {
            "game_players": game_players_dict,
            "round_players": round_players_dict,
            "deck": deck_dict,
            "stage": self.game.stage,
            "public_cards": [{"rank": card.rank, "suit": card.suit} for card in self.game.public_cards],
            "pot": self.game.pot,
            "current_bet": self.game.current_bet,
            "small_blind_amount": self.game.small_blind_amount,
            "big_blind_amount": self.game.big_blind_amount,
            "small_blind_player": small_blind_player_dict,
            "big_blind_player": big_blind_player_dict,
            "active_player": active_player_dict
        }
