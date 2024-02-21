from players.ai_player import AIPlayer
from players.human_player import HumanPlayer
from poker_game import PokerGame
from poker_state import PokerState
from utils.deck import Deck


class PokerGameManager:

    def __init__(self, rules):
        self.rules = rules
        players = self.gen_poker_players(num_ai_players=rules["num_ai_players"], num_human_players=["num_human_players"])
        init_state = self.gen_init_state()
        self.game = PokerGame(players=players, state=init_state)

    def gen_init_state(self):
        init_state = PokerState(
            small_blind_amount=self.rules["small_blind_amount"],
            big_blind_amount=self.rules["big_blind_amount"],
            deck=self.gen_deck(self.rules["num_cards_deck"])
        )
        return init_state

    def gen_deck(self, num_cards):
        rest = num_cards % 4
        if rest:
            raise ValueError(f"num_cards_in_deck is wrongly configured in rules")
        num_ranks = num_cards / 4
        ranks = ("A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2")

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

        self.game.players = players

    def get_current_game(self):
        return self.game
