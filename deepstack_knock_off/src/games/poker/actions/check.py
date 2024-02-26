from games.poker.actions.action import Action
from games.poker.players.player import Player


class Check(Action):

    def __init__(self, player: Player):
        super().__init__(name="Check", player=player)
