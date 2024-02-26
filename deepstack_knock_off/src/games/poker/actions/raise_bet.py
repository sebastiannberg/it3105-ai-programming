from games.poker.actions.action import Action
from games.poker.players.player import Player


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
            raise NotImplementedError("All in action not implemented yet")
        else:
            raise ValueError(f"Received unexpected raise_type {raise_type}")
        super().__init__(name=name, player=player)
