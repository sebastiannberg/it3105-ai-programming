from games.poker.poker_game_manager import PokerGameManager


class PokerGameService:
    """
    Providing an interface for the API to interact with the
    underlying poker game mechanics encapsulated in the PokerGameManager.
    """

    def __init__(self, game_manager: PokerGameManager):
        self.game_manager = game_manager

    def start_game(self):
        pass

    def apply_action(self):
        pass
