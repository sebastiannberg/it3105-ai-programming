from games.poker.players.player import Player


class HumanPlayer(Player):

    def __init__(self, name, initial_chips):
        super().__init__(name, initial_chips)
