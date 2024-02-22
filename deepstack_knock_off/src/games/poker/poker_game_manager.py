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
            players.append(AIPlayer(name=f"AI Player {i}", initial_chips=initial_chips))
        # Generate human players
        for i in range(num_human_players):
            players.append(HumanPlayer(name=f"Human Player {i}", initial_chips=initial_chips))
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
        stage = self.game.stage
        if stage == "preflop":
            pass
        elif stage == "flop":
            pass
        elif stage == "turn":
            pass
        elif stage == "river":
            pass

    def assign_active_player(self):
        pass
