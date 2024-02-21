from typing import Dict

from games.poker.players.ai_player import AIPlayer
from games.poker.players.human_player import HumanPlayer
from games.poker.players.player import Player
from games.poker.poker_game import PokerGame
from games.poker.poker_state import PokerState
from games.poker.poker_oracle import PokerOracle


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

    # def run_poker_game(self):
    #     while len(self.game.state.players) >= 2:
    #         print("hello")
    #     print("Winner: ", self.game.state.players[0])

    # def run_one_round(self):
    #     pass

    def assign_blind_roles(self):
        current_small_blind_player = self.game.state.small_blind_player
        current_big_blind_player = self.game.state.big_blind_player
        players = self.game.state.round_players
        if current_small_blind_player and current_big_blind_player:
            small_blind_index = players.index(current_small_blind_player)

            next_small_blind_index = (small_blind_index + 1) % len(players)
            next_small_blind_player = players[next_small_blind_index]

            next_big_blind_index = (next_small_blind_index + 1) % len(players)
            next_big_blind_player = players[next_big_blind_index]
        else:
            next_small_blind_player = players[0]
            next_big_blind_player = players[1]

        self.game.state.small_blind_player = next_small_blind_player
        self.game.state.big_blind_player = next_big_blind_player


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

    def gen_all_available_actions(self, player: Player):
        # return a list of all possible actions for the player given the player and the state
        # fold (always possible)
        # raise (always possible if have the money)
        # check (if possible)
        pass

    def apply_player_action(self):
        pass

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

    def gen_init_state(self, players, deck, small_blind_amount, big_blind_amount):
        init_state = PokerState(
            game_players=players,
            round_players=players,
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
