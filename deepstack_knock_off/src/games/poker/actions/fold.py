from games.poker.actions.action import Action
from games.poker.players.player import Player
from games.poker.poker_state import PokerState


class Fold(Action):

    def fold(player: Player, state: PokerState):
        state.round_players.remove(player)
        return state
