from games.poker.actions.action import Action
from games.poker.players.player import Player


class Fold(Action):

    def __init__(self, player: Player):
        super().__init__(name="Fold", player=player)
