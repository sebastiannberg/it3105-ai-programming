from games.poker.players.player import Player


class HumanPlayer(Player):

    def __init__(self, name, initial_chips, possible_actions):
        super().__init__(name, initial_chips, possible_actions)
