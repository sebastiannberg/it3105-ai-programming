from players.ai_player import AIPlayer
from players.human_player import HumanPlayer
from poker_game import PokerGame
from poker_state import PokerState
from poker_oracle import PokerOracle
from typing import Dict


class PokerGameManager:

    def __init__(self, rules: Dict, oracle: PokerOracle):
        self.rules = rules
        self.oracle = oracle

        players = self.gen_poker_players(num_ai_players=rules["num_ai_players"], num_human_players=rules["num_human_players"])
        deck = self.oracle.gen_deck(rules["deck_size"]).shuffle()
        init_state = self.gen_init_state(players, deck)

        self.game = PokerGame(state=init_state)

    def get_current_game(self):
        return self.game

    def gen_poker_players(self, num_ai_players, num_human_players):
        if num_ai_players + num_human_players > 6:
            raise ValueError("Total amount of players should be less than 6")
        if num_ai_players + num_human_players < 2:
            raise ValueError("Total amount of players should be at least 2")

        players = []
        # Generate AI players
        for _ in range(num_ai_players):
            players.append(AIPlayer())
        # Generate human players
        for _ in range(num_human_players):
            players.append(HumanPlayer())

        return players

    def gen_init_state(self, players, deck):
        init_state = PokerState(
            players=players,
            deck=deck,
            stage="preflop",
            public_cards=[],
            pot=0,
            small_blind_amount=self.rules["small_blind_amount"],
            big_blind_amount=self.rules["big_blind_amount"],
            dealer=players[0],
            # TODO deal with less than three players
            small_blind_player=players[1],
            big_blind_player=players[2]
        )
        return init_state

    def find_dealer_indexes(self, players):
        pass
