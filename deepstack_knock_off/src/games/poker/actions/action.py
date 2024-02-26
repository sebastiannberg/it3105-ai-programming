from games.poker.players.player import Player


class Action:

    def __init__(self, name: str, player: Player):
        self.name = name
        self.player = player

    def to_dict(self):
        return {
            'name': self.name,
            'player': self.player.name
        }
