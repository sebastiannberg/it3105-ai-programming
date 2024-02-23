from games.poker.actions.action import Action
from games.poker.players.player import Player


class Fold(Action):

    def __init__(self, player: Player):
        self.player = player
        super().__init__(name="Fold")
