from games.poker.players.player import Player


class AIPlayer(Player):

    def __init__(self, name, initial_chips):
        super().__init__(name, initial_chips)
