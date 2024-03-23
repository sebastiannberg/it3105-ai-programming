from typing import Dict, List

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle
from games.poker.players.ai_player import AIPlayer


class PokerGameService:
    """
    Providing an interface for the API to interact with the
    underlying poker game mechanics encapsulated in the PokerGameManager.
    """

    def __init__(self, game_manager: PokerGameManager):
        self.game_manager = game_manager

    def start_game(self):
        self.game_manager.init_poker_game()
        self.game_manager.assign_blind_roles()
        self.game_manager.perform_blind_bets()
        self.game_manager.deal_cards()
        self.game_manager.assign_active_player()

    def legal_actions(self) -> List[Dict]:
        legal_actions = PokerStateManager.find_all_legal_actions(self.game_manager.game, self.game_manager.game.active_player, self.game_manager.rules)
        return [action.to_dict() for action in legal_actions]

    def apply_action(self, data):
        action_name = data.get("name")
        player_name = data.get("player")

        selected_action = None
        for legal_action in  PokerStateManager.find_all_legal_actions(self.game_manager.game, self.game_manager.game.active_player, self.game_manager.rules):
            if action_name == legal_action.name and player_name == legal_action.player.name:
                selected_action = legal_action
                break

        if selected_action:
            player = self.game_manager.find_round_player_by_name(player_name)
            PokerStateManager.apply_action(self.game_manager.game, player, selected_action)

            game_winner = self.game_manager.check_for_game_winner()
            if game_winner:
                return {"winner": game_winner.name}

            early_round_winner = self.game_manager.check_for_early_round_winner()
            if early_round_winner:
                self.game_manager.process_winnings()
                self.game_manager.remove_busted_players()

                game_winner = self.game_manager.check_for_game_winner()
                if game_winner:
                    return {"winner": game_winner.name}
                else:
                    return {"round_winners":[
                        {
                            "player": early_round_winner.name,
                            "early_win": True
                        }
                    ]}

            if self.game_manager.check_for_proceed_stage():
                winners_details = self.game_manager.proceed_stage()
                if winners_details:
                    self.game_manager.process_winnings()
                    self.game_manager.remove_busted_players()

                    game_winner = self.game_manager.check_for_game_winner()
                    if game_winner:
                        return {"winner": game_winner.name}

                    return {"round_winners": winners_details}
            else:
                self.game_manager.assign_active_player()

            return {"message": "Action applied successfully"}
        else:
            return None

    def next_round(self):
        self.game_manager.end_round_next_round()

    def ai_decision(self) -> Dict:
        if not isinstance(self.game_manager.game.active_player, AIPlayer):
            return None

        legal_actions = PokerStateManager.find_all_legal_actions(self.game_manager.game, self.game_manager.game.active_player, self.game_manager.rules)

        if self.game_manager.rules["ai_strategy"] == "rollout":
            selected_action = self.game_manager.game.active_player.make_decision_rollouts(self.game_manager.oracle, self.game_manager.game.public_cards, len(self.game_manager.game.round_players)-1, legal_actions)
        elif self.game_manager.rules["ai_strategy"] == "resolve":
            raise NotImplementedError()
        return selected_action.to_dict()

    def jsonify_poker_game(self) -> Dict:
        """
        Generates a dictionary containing information about the game state
        """
        game = self.game_manager.game
        if game.game_players:
            game_players_dict = {}
            for player in game.game_players:
                game_players_dict[player.name] = {
                    "hand": [{"rank": card.rank, "suit": card.suit} for card in player.hand],
                    "chips": player.chips,
                    "player_bet": player.player_bet
                }
        else:
            game_players_dict = None

        if game.round_players:
            round_players_dict = {}
            for player in game.round_players:
                round_players_dict[player.name] = {
                    "hand": [{"rank": card.rank, "suit": card.suit} for card in player.hand],
                    "chips": player.chips,
                    "player_bet": player.player_bet
                }
        else:
            round_players_dict = None

        if game.deck:
            deck_dict = {"cards": [{"rank": card.rank, "suit": card.suit} for card in game.deck.cards]}
        else:
            deck_dict = None

        if game.small_blind_player and game.big_blind_player:
            small_blind_player_dict = {game.small_blind_player.name: {
                "hand": [{"rank": card.rank, "suit": card.suit} for card in game.small_blind_player.hand],
                "chips": game.small_blind_player.chips,
                "player_bet": game.small_blind_player.player_bet
            }}
            big_blind_player_dict = {game.big_blind_player.name: {
                "hand": [{"rank": card.rank, "suit": card.suit} for card in game.big_blind_player.hand],
                "chips": game.big_blind_player.chips,
                "player_bet": game.big_blind_player.player_bet
            }}
        else:
            small_blind_player_dict = None
            big_blind_player_dict = None

        if game.active_player:
            active_player_dict = {game.active_player.name: {
                "hand": [{"rank": card.rank, "suit": card.suit} for card in game.active_player.hand],
                "chips": game.active_player.chips,
                "player_bet": game.active_player.player_bet
            }}
        else:
            active_player_dict = None

        return {
            "game_players": game_players_dict,
            "round_players": round_players_dict,
            "deck": deck_dict,
            "stage": game.stage,
            "public_cards": [{"rank": card.rank, "suit": card.suit} for card in game.public_cards],
            "pot": game.pot,
            "current_bet": game.current_bet,
            "small_blind_amount": game.small_blind_amount,
            "big_blind_amount": game.big_blind_amount,
            "small_blind_player": small_blind_player_dict,
            "big_blind_player": big_blind_player_dict,
            "active_player": active_player_dict
        }
