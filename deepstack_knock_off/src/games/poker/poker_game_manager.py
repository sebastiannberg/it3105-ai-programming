from typing import Dict

from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle
from games.poker.poker_state import PokerState
from games.poker.players.ai_player import AIPlayer
from games.poker.players.human_player import HumanPlayer
from games.poker.actions.raise_bet import RaiseBet


class PokerGameManager:

    def __init__(self, rules: Dict, oracle: PokerOracle):
        self.rules = rules
        self.oracle = oracle

    def init_poker_game(self):
        deck = self.oracle.gen_deck(self.rules["deck_size"])
        deck.shuffle()
        players = self.gen_poker_players(num_ai_players=self.rules["num_ai_players"], num_human_players=self.rules["num_human_players"])
        init_state = PokerStateManager.gen_init_state(players, deck, self.rules["small_blind_amount"], self.rules["big_blind_amount"])
        self.game: PokerState = init_state

    def gen_poker_players(self, num_ai_players, num_human_players):
        if num_ai_players + num_human_players > 6:
            raise ValueError("Total amount of players should be less than 6")
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

        small_blind_action = RaiseBet(player=current_small_blind_player, raise_amount=self.game.small_blind_amount)
        big_blind_action = RaiseBet(player=current_big_blind_player, raise_amount=self.game.big_blind_amount)

        PokerStateManager.apply_action(self.game, small_blind_action)
        PokerStateManager.apply_action(self.game, big_blind_action)

    def deal_cards(self):
        if self.game.stage == "preflop":
            pass
        elif self.game.stage == "flop":
            pass
        elif self.game.stage == "turn":
            pass
        elif self.game.stage == "river":
            pass

    def assign_active_player(self):
        pass

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
            "public_cards": [(card.rank, card.suit) for card in self.game.public_cards],
            "pot": self.game.pot,
            "current_bet": self.game.current_bet,
            "small_blind_amount": self.game.small_blind_amount,
            "big_blind_amount": self.game.big_blind_amount,
            "small_blind_player": small_blind_player_dict,
            "big_blind_player": big_blind_player_dict,
            "active_player": active_player_dict
        }
