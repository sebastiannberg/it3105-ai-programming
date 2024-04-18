from typing import Dict

from games.poker.actions.action import Action
from games.poker.players.player import Player
from games.poker.poker_game import PokerGame


class Fold(Action):

    def __init__(self, player: Player):
        super().__init__(name="Fold", player=player)

    def apply(self, game: PokerGame) -> Dict:
        game.history.append((game.stage, self))
        game.round_players.remove(self.player)
        self.player.fold()
        return {"message": f"{self.player.name} folded"}
