from games.poker.actions.action import Action
from games.poker.players.player import Player


class RaiseBet(Action):

    def __init__(self, player, raise_amount):
        self.player: Player = player
        self.raise_amount: int = raise_amount
