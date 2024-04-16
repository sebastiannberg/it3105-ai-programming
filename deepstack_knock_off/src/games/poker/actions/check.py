from typing import Dict

from games.poker.actions.action import Action
from games.poker.players.player import Player
from games.poker.poker_game import PokerGame


class Check(Action):

    def __init__(self, player: Player):
        super().__init__(name="Check", player=player)

    def apply(self, game: PokerGame) -> Dict:
        game.stage_history.append(self)
        self.player.check()
        return {"message": f"{self.player.name} check"}
