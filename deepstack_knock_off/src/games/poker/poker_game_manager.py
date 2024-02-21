from games.poker.players.ai_player import AIPlayer
from games.poker.players.human_player import HumanPlayer
from games.poker.poker_game import PokerGame
from games.poker.poker_state import PokerState
from games.poker.poker_oracle import PokerOracle
from games.poker.actions.action import Action
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

    def init_poker_game(self):
        pass

    def deal_cards(self):
        pass
        # TODO different stages

    def apply_player_action(self, action: Action):
        pass

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

    def gen_player_actions(self):
        # TODO based on rules, and create a list of actions to be passed to players
        pass

    def gen_init_state(self, players, deck):
        init_state = PokerState(
            players=players,
            deck=deck,
            stage="preflop",
            public_cards=[],
            pot=0,
            small_blind_amount=self.rules["small_blind_amount"],
            big_blind_amount=self.rules["big_blind_amount"],
            # TODO deal with less than three players
            small_blind_player=players[1],
            big_blind_player=players[2],
            current_player=players[1]
        )
        return init_state

    def find_dealer_indexes(self, players):
        pass
