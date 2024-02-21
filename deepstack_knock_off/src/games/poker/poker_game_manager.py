from typing import Dict

from games.poker.players.ai_player import AIPlayer
from games.poker.players.human_player import HumanPlayer
from games.poker.poker_game import PokerGame
from games.poker.poker_state import PokerState
from games.poker.poker_oracle import PokerOracle
from games.poker.actions.action import Action


class PokerGameManager:

    def __init__(self, rules: Dict, oracle: PokerOracle):
        self.rules = rules
        self.oracle = oracle

    def init_poker_game(self):
        deck = self.oracle.gen_deck(self.rules["deck_size"])
        deck.shuffle()
        players = self.gen_poker_players(num_ai_players=self.rules["num_ai_players"], num_human_players=self.rules["num_human_players"])
        init_state = self.gen_init_state(players, deck, self.rules["small_blind_amount"], self.rules["big_blind_amount"])
        self.game = PokerGame(state=init_state)

    def run_poker_game(self):
        while len(self.game.state.players) >= 2:
            print("hello")
        print("Winner: ", self.game.state.players[0])

    def run_one_round(self):
        pass

    def assign_blind_roles(self):
        pass
        # TODO return small_blind_player, big_blind_player

    def perform_blind_bets(self):
        pass

    def deal_cards(self):
        stage = self.game.state.stage
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

    def apply_player_action(self, action: Action):
        pass

    def advance_stage(self):
        pass

    def gen_poker_players(self, num_ai_players, num_human_players):
        if num_ai_players + num_human_players > 6:
            raise ValueError("Total amount of players should be less than 6")
        if num_ai_players + num_human_players < 2:
            raise ValueError("Total amount of players should be at least 2")
        players = []
        initial_chips = self.rules["initial_chips"]
        possible_actions = self.gen_player_actions(all_in_disabled=self.rules["all_in_disabled"])
        # Generate AI players
        for i in range(num_ai_players):
            players.append(AIPlayer(name=f"AI Player {i}", initial_chips=initial_chips, possible_actions=possible_actions))
        # Generate human players
        for i in range(num_human_players):
            players.append(HumanPlayer(name=f"Human Player {i}", initial_chips=initial_chips, possible_actions=possible_actions))
        return players

    def gen_player_actions(self, all_in_disabled):
        # TODO maybe not needed
        actions = []
        if all_in_disabled:
            pass
        else:
            pass
        return actions

    def gen_init_state(self, players, deck, small_blind_amount, big_blind_amount):
        init_state = PokerState(
            players=players,
            deck=deck,
            stage="preflop",
            public_cards=[],
            pot=0,
            small_blind_amount=small_blind_amount,
            big_blind_amount=big_blind_amount,
            small_blind_player=None,
            big_blind_player=None,
            active_player=None
        )
        return init_state
