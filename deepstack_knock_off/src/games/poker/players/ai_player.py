from games.poker.players.player import Player


class AIPlayer(Player):

    def __init__(self, name, initial_chips):
        super().__init__(name, initial_chips)

    def make_decision_rollouts(self):
        pass

    def make_decision_resolving(self):
        pass
