from typing import Dict, List, Optional
import random

from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle
from games.poker.poker_game import PokerGame
from games.poker.players.player import Player
from games.poker.players.ai_player import AIPlayer
from games.poker.players.human_player import HumanPlayer
from games.poker.actions.action import Action
from games.poker.actions.raise_bet import RaiseBet
from games.poker.actions.fold import Fold
from games.poker.actions.check import Check


class PokerGameManager:

    def __init__(self, poker_config: Dict, poker_rules: Dict) -> None:
        self.poker_config: Dict = poker_config
        self.poker_rules: Dict = poker_rules
        self.state_manager: PokerStateManager = PokerStateManager(poker_rules)
        players = self.gen_poker_players(num_ai_players=self.poker_config["num_ai_players"], num_human_players=self.poker_config["num_human_players"])
        if len(players) > 2 and self.poker_config["enable_resolver"] == True:
            raise ValueError("Cannot use resolver when there are more than 2 players")
        self.game: PokerGame = PokerGame(
            game_players=players,
            round_players=players.copy(),
            small_blind_player=None,
            big_blind_player=None,
            current_player=None,
            deck=PokerOracle.gen_deck(self.poker_rules["deck_size"], shuffled=True),
            public_cards=[],
            stage="preflop",
            history=[],
            pot=0,
            current_bet=0,
            ai_strategy=None
        )

    def start_game(self) -> None:
        self.assign_blind_roles()
        self.perform_blind_bets()
        self.deal_cards()
        self.assign_next_player()

    def assign_legal_actions_to_player(self, player_name: str) -> None:
        player = self.find_round_player_by_name(player_name)
        legal_actions: List[Action] = []
        # Fold
        legal_actions.append(Fold(player))
        # Check
        if self.game.current_bet == player.player_bet:
            legal_actions.append(Check(player=player))
        # Call
        if player.player_bet < self.game.current_bet and player.chips >= (self.game.current_bet - player.player_bet):
            legal_actions.append(RaiseBet(player, chip_cost=(self.game.current_bet - player.player_bet), raise_amount=0, raise_type="call"))
        # Raise
        if player.chips >= self.poker_rules["fixed_raise"] + (self.game.current_bet - player.player_bet) and not any(player.has_all_in or player.chips == 0 for player in self.game.round_players):
            current_stage_history = [entry[1] for entry in self.game.history if entry[0] == self.game.stage]
            if len([action for action in current_stage_history if isinstance(action, RaiseBet) and action.raise_type=="raise"]) < self.poker_rules["max_num_raises_per_stage"]:
                legal_actions.append(RaiseBet(player, chip_cost=(self.game.current_bet -player.player_bet + self.poker_rules["fixed_raise"]), raise_amount=self.poker_rules["fixed_raise"], raise_type="raise"))
        # All In (if player has chips but cannot call)
        if  player.player_bet < self.game.current_bet and player.chips < (self.game.current_bet - player.player_bet):
            legal_actions.append(RaiseBet(player, chip_cost=player.chips, raise_amount=0, raise_type="all_in"))
        player.legal_actions = legal_actions

    def apply_action(self, player_name: str, action_name: str) -> Dict:
        # Find the player by name
        player = self.find_round_player_by_name(player_name)
        if not player:
            return {"error": "Player not found"}

        # Find action by player name and action name
        selected_action = None
        for legal_action in  player.legal_actions:
            if player_name == legal_action.player.name and action_name == legal_action.name:
                selected_action = legal_action
                break

        action_result = selected_action.apply(self.game)
        # Reset player's legal actions after applying
        player.legal_actions = None

        post_action_result = self.post_action()
        if not post_action_result:
            return action_result

        return post_action_result

    def post_action(self) -> Optional[Dict]:
        early_round_winner = self.check_for_early_round_winner()
        if early_round_winner:
            self.process_winnings()
            self.remove_busted_players()

            game_winner = self.check_for_game_winner()
            if game_winner:
                return {"game_winner": game_winner.name}

            return {"round_winners":[
                {
                    "player": early_round_winner.name,
                    "early_win": True
                }
            ]}

        if self.check_for_proceed_stage():
            # Edge case for all in
            if any(player.has_all_in for player in self.game.round_players):
                print("Inside all in edge case post action")
                # Deal out remainder of cards
                if len(self.game.public_cards) < 5:
                    cards = self.game.deck.deal_cards(5-len(self.game.public_cards))
                    self.game.public_cards.extend(cards)
                print(f"Public cards: {len(self.game.public_cards)}")
                # Showdown
                winners_details = self.showdown()
                self.process_winnings()
                self.remove_busted_players()
                game_winner = self.check_for_game_winner()
                if game_winner:
                    return {"game_winner": game_winner.name}
                return {"round_winners": winners_details}
            else:
                winners_details = self.proceed_stage()
                if winners_details:
                    self.process_winnings()
                    self.remove_busted_players()
                    game_winner = self.check_for_game_winner()
                    if game_winner:
                        return {"game_winner": game_winner.name}
                    return {"round_winners": winners_details}
        else:
            self.assign_next_player()

    def get_ai_decision(self) -> Dict:
        if not isinstance(self.game.current_player, AIPlayer):
            return None

        self.assign_legal_actions_to_player(self.game.current_player.name)

        if self.poker_config["enable_resolver"]:
            if random.random() < self.poker_config["prob_resolver"]:
                selected_action = self.game.current_player.make_decision_resolving()
            else:
                selected_action = self.game.current_player.make_decision_rollouts(self.game.public_cards, self.poker_rules["deck_size"], len(self.game.round_players)-1)
        else:
            selected_action = self.game.current_player.make_decision_rollouts(self.game.public_cards, self.poker_rules["deck_size"], len(self.game.round_players)-1)
        return selected_action.to_dict()

    def gen_poker_players(self, num_ai_players: int, num_human_players: int) -> List[Player]:
        if num_ai_players + num_human_players > 6:
            raise ValueError("Total amount of players should be less than or equal to 6")
        if num_ai_players + num_human_players < 2:
            raise ValueError("Total amount of players should be at least 2")

        players = []
        # Generate AI players
        for i in range(num_ai_players):
            players.append(AIPlayer(name=f"AI Player {i+1}", initial_chips=self.poker_config["initial_chips"], state_manager=self.state_manager))
        # Generate human players
        for i in range(num_human_players):
            players.append(HumanPlayer(name=f"Human Player {i+1}", initial_chips=self.poker_config["initial_chips"]))
        return players

    def find_round_player_by_name(self, player_name: str) -> Optional[Player]:
        for player in self.game.round_players:
            if player.name == player_name:
                return player
        raise ValueError(f"{player_name} not found in game round")

    def assign_blind_roles(self):
        # Get the current players in the round and the game
        round_players = self.game.round_players
        game_players = set(self.game.game_players)  # Convert to set for faster lookup

        # Check if there are current blind players set
        if self.game.small_blind_player and self.game.big_blind_player:
            # Find the current small blind player in the round
            current_small_blind_index = round_players.index(self.game.small_blind_player)

            # Initialize indices for searching next blinds
            next_small_blind_index = (current_small_blind_index + 1) % len(round_players)
            next_big_blind_index = (next_small_blind_index + 1) % len(round_players)

            # Find the next small blind player who is still in the game
            while round_players[next_small_blind_index] not in game_players:
                next_small_blind_index = (next_small_blind_index + 1) % len(round_players)
            next_small_blind_player = round_players[next_small_blind_index]

            # Find the next big blind player who is still in the game
            next_big_blind_index = (next_small_blind_index + 1) % len(round_players)
            while round_players[next_big_blind_index] not in game_players:
                next_big_blind_index = (next_big_blind_index + 1) % len(round_players)
            next_big_blind_player = round_players[next_big_blind_index]

            # Set the new blind players in the game
            self.game.small_blind_player = next_small_blind_player
            self.game.big_blind_player = next_big_blind_player
        else:
            next_small_blind_player = round_players[0]
            next_big_blind_player = round_players[1]

        self.game.small_blind_player = next_small_blind_player
        self.game.big_blind_player = next_big_blind_player

    def perform_blind_bets(self):
        current_small_blind_player = self.game.small_blind_player
        current_big_blind_player = self.game.big_blind_player
        if not current_small_blind_player or not current_big_blind_player:
            raise ValueError("Either small blind or big blind is not assigned to a player")
        # Small blind action
        raise_amount = (current_small_blind_player.player_bet + self.poker_config["small_blind_amount"]) - self.game.current_bet
        small_blind_action = RaiseBet(player=current_small_blind_player,
                                      chip_cost=self.poker_config["small_blind_amount"],
                                      raise_amount=raise_amount,
                                      raise_type="small_blind")
        small_blind_action.apply(self.game)
        # Big blind action
        raise_amount = (current_big_blind_player.player_bet + self.poker_config["big_blind_amount"]) - self.game.current_bet
        big_blind_action = RaiseBet(player=current_big_blind_player,
                                    chip_cost=self.poker_config["big_blind_amount"],
                                    raise_amount=raise_amount,
                                    raise_type="big_blind")
        big_blind_action.apply(self.game)

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

    def assign_next_player(self, stage_change=False):
        if not self.game.current_player:
            # Assign player after big blind
            current_big_blind_player = self.game.big_blind_player
            big_blind_index = self.game.round_players.index(current_big_blind_player)
            next_player_index = (big_blind_index + 1) % len(self.game.round_players)
            self.game.current_player = self.game.round_players[next_player_index]
        elif stage_change:
            # Assign first player after dealer
            if not self.game.small_blind_player.has_folded:
                self.game.current_player = self.game.small_blind_player
            else:
                small_blind_index = self.game.game_players.index(self.game.small_blind_player)
                next_player_index = (small_blind_index + 1) % len(self.game.game_players)

                while self.game.game_players[next_player_index].has_folded:
                    next_player_index = (next_player_index + 1) % len(self.game.game_players)
                    if next_player_index == small_blind_index:
                        break
                self.game.current_player = self.game.game_players[next_player_index]
        else:
            # Assign player after current player
            current_player = self.game.current_player
            current_player_index = self.game.game_players.index(current_player)
            next_player_index = (current_player_index + 1) % len(self.game.game_players)
            while self.game.game_players[next_player_index].has_folded:
                next_player_index = (next_player_index + 1) % len(self.game.game_players)
                if next_player_index == current_player_index:
                    break
            self.game.current_player = self.game.game_players[next_player_index]

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
            # Reset stage variables
            self.game.current_bet = 0
            # Deal cards and assign the active player for the new stage
            self.deal_cards()
            self.assign_next_player(stage_change=True)
        elif self.game.stage == "river":
            # Showdown and determine winners
            winners = self.showdown()

        return winners

    def showdown(self) -> List[Dict]:
        # Classify each player's hand and associate it with the player
        classified_hands = {
            player: PokerOracle.classify_poker_hand(player.hand, self.game.public_cards, player)
            for player in self.game.round_players
        }

        # Find the best hand to compare with others
        best_hand = max(classified_hands.values(), key=lambda hand: (
            PokerOracle.hand_type_ranking[hand.category],
            hand.primary_value,
            hand.kickers
        ))

        # Filter winners based on the comparison, preserving original order
        winners = []
        for player, hand in classified_hands.items():
            comparison_result = PokerOracle.compare_poker_hands(hand, best_hand)
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
        # Reset every player still in the game
        for player in self.game.game_players:
            player.ready_for_new_round()
        self.game.round_players = self.game.game_players.copy()
        # Reset round variables
        self.game.current_player = None
        self.game.current_bet = 0
        self.game.public_cards = []
        self.game.stage = "preflop"
        self.deck = PokerOracle.gen_deck(num_cards=52, shuffled=True)
        # Start a new round
        self.assign_blind_roles()
        self.perform_blind_bets()
        self.deal_cards()
        self.assign_next_player()

    def remove_busted_players(self):
        for player in self.game.game_players:
            if player.chips <= 0:
                self.game.game_players.remove(player)

    def check_for_game_winner(self):
        if len(self.game.game_players) == 1:
            return self.game.game_players[0]
        else:
            return None

    def jsonify_poker_game(self) -> Dict:
        """
        Generates a dictionary containing information about the game
        """
        return {
            "game_players": [player.to_dict() for player in self.game.game_players],
            "round_players": [player.to_dict() for player in self.game.round_players],
            "small_blind_player": self.game.small_blind_player.to_dict() if self.game.small_blind_player else None,
            "big_blind_player": self.game.big_blind_player.to_dict() if self.game.big_blind_player else None,
            "current_player": self.game.current_player.to_dict() if self.game.current_player else None,
            "deck": {"cards": [{"rank": card.rank, "suit": card.suit} for card in self.game.deck.cards]},
            "public_cards": [{"rank": card.rank, "suit": card.suit} for card in self.game.public_cards],
            "stage": self.game.stage,
            "pot": self.game.pot,
            "current_bet": self.game.current_bet,
            "ai_strategy": self.game.ai_strategy
        }
