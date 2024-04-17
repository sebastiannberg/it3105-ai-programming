from typing import Dict

from games.poker.actions.action import Action
from games.poker.players.player import Player
from games.poker.poker_game import PokerGame


class RaiseBet(Action):

    def __init__(self, player: Player, chip_cost, raise_amount, raise_type):
        self.chip_cost: int = chip_cost
        self.raise_amount: int = raise_amount
        self.raise_type: str = raise_type
        name = ""
        if raise_type == "small_blind":
            name = f"Small Blind ({chip_cost})"
        elif raise_type == "big_blind":
            name = f"Big Blind ({chip_cost})"
        elif raise_type == "call":
            name = f"Call ({chip_cost})"
        elif raise_type == "raise":
            name = f"Raise {raise_amount} ({chip_cost})"
        elif raise_type == "all_in":
            name = "All In"
        else:
            raise ValueError(f"Received unexpected raise_type {raise_type}")
        super().__init__(name=name, player=player)

    def apply(self, game: PokerGame) -> Dict:
        game.stage_history.append(self)
        self.player.chips -= self.chip_cost
        game.current_bet += self.raise_amount
        game.pot += self.chip_cost
        self.player.player_bet += self.chip_cost
        if self.raise_type == "call":
            self.player.call()
        elif self.raise_type == "raise":
            for player_temp in game.round_players:
                player_temp.has_called = False
                player_temp.last_raised = False
            self.player.poker_raise()
        elif self.raise_type == "all_in":
            self.player.all_in()
        return {"message": f"{self.player.name} raise bet"}
