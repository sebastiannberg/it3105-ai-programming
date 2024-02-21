from games.poker.players.player import Player
from games.poker.poker_state import PokerState


class Action:


    @staticmethod
    def check(player: Player, state: PokerState):
        state.round_players.remove(player)
        return state
