from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_caching import Cache
from typing import List

import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle
from games.poker.actions.action import Action
from games.poker.players.ai_player import AIPlayer

app = Flask(__name__)
CORS(app)
cache = Cache(app, config={"CACHE_TYPE": "simple"})

@app.route("/rules", methods=["GET"])
def rules():
    try:
        return send_from_directory(directory=os.path.join(app.root_path, "data"), path="rules.json", as_attachment=False)
    except:
        return jsonify({"error": "File not found."}), 404


@app.route("/start-game", methods=["POST"])
def start_game():
    rules = request.json
    try:
        # Create new game manager object
        game_manager = PokerGameManager(rules, PokerOracle())
        # Start game
        game_manager.start_game()
        # Save game manager for later use
        cache.set("game_manager", pickle.dumps(game_manager))
        return jsonify({"message": "Game started successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/game-state", methods=["GET"])
def game_state():
    serialized_game_manager = cache.get("game_manager")
    if serialized_game_manager:
        game_manager: PokerGameManager = pickle.loads(serialized_game_manager)
        game_state_json = game_manager.jsonify_poker_game()
        return jsonify(game_state_json), 200
    else:
        return jsonify({"error": "Game not started."}), 404

@app.route("/legal-actions", methods=["GET"])
def legal_actions():
    serialized_game_manager = cache.get("game_manager")
    if serialized_game_manager:
        game_manager: PokerGameManager = pickle.loads(serialized_game_manager)
        current_state = game_manager.game
        current_player = game_manager.game.active_player
        legal_actions = PokerStateManager.find_all_legal_actions(state=current_state, player=current_player, rules=game_manager.rules)
        cache.set("legal_actions", pickle.dumps(legal_actions))
        legal_actions_dicts = [action.to_dict() for action in legal_actions]
        return jsonify(legal_actions_dicts), 200
    else:
        return jsonify({"error": "Failed to fetch possible actions"}), 404

@app.route("/apply-action", methods=["POST"])
def apply_action():
    action_data = request.json
    serialized_game_manager = cache.get("game_manager")
    serialized_legal_actions = cache.get("legal_actions")

    if serialized_game_manager and serialized_legal_actions:
        game_manager: PokerGameManager = pickle.loads(serialized_game_manager)
        legal_actions: List[Action] = pickle.loads(serialized_legal_actions)

        selected_action = None
        for legal_action in legal_actions:
            # TODO maybe check player name as well
            if legal_action.name == action_data.get("name"):
                selected_action = legal_action
                break

        if selected_action:
            player = game_manager.find_round_player_by_name(selected_action.player.name)
            PokerStateManager.apply_action(game_manager.game, player, selected_action)

            game_winner = game_manager.check_for_game_winner()
            if game_winner:
                cache.set("game_manager", pickle.dumps(game_manager))
                return jsonify({"winner": game_winner.name})

            early_round_winner = game_manager.check_for_early_round_winner()
            if early_round_winner:
                game_manager.process_winnings()
                game_manager.remove_busted_players()

                game_winner_after_early_round = game_manager.check_for_game_winner()
                if game_winner_after_early_round:
                    cache.set("game_manager", pickle.dumps(game_manager))
                    return jsonify({"winner": game_winner_after_early_round.name})

                cache.set("game_manager", pickle.dumps(game_manager))
                return jsonify({"round_winners": [
                    {
                        "player": early_round_winner.name,
                        "early_win": True
                    }
                ]})

            if game_manager.check_for_proceed_stage():
                winners_details = game_manager.proceed_stage()
                # If winners details is not None it means we had a showdown
                if winners_details:
                    game_manager.process_winnings()
                    game_manager.remove_busted_players()

                    game_winner = game_manager.check_for_game_winner()
                    if game_winner:
                        cache.set("game_manager", pickle.dumps(game_manager))
                        return jsonify({"winner": game_winner.name})

                    cache.set("game_manager", pickle.dumps(game_manager))
                    return jsonify({"round_winners": winners_details})
            else:
                game_manager.assign_active_player()

            cache.set("game_manager", pickle.dumps(game_manager))
            return jsonify({"message": "Action applied successfully."}), 200
        else:
            return jsonify({"error": "Action not found or not allowed."}), 404
    else:
        return jsonify({"error": "Game not started or legal actions not found."}), 404

@app.route("/next-round", methods=["POST"])
def next_round():
    serialized_game_manager = cache.get("game_manager")
    if serialized_game_manager:
        game_manager: PokerGameManager = pickle.loads(serialized_game_manager)
        game_manager.end_round_next_round()
        cache.set("game_manager", pickle.dumps(game_manager))
        return jsonify({"message": "Next round started successfully."}), 200
    else:
        return jsonify({"error": "Game not started or not found."}), 404

@app.route("/ai-decision", methods=["POST"])
def ai_decision():
    serialized_game_manager = cache.get("game_manager")
    if not serialized_game_manager:
        return jsonify({"error": "Game not started."}), 404

    game_manager: PokerGameManager = pickle.loads(serialized_game_manager)
    if not isinstance(game_manager.game.active_player, AIPlayer):
        return jsonify({"error": "It's not AI's turn."}), 400

    decision = game_manager.game.active_player.make_decision_rollouts(game_manager.game)
    # Apply the decision TODO

    cache.set("game_manager", pickle.dumps(game_manager))
    return jsonify({"message": "AI decision made.", "decision": decision}), 200




if __name__ == "__main__":
    app.run(debug=True)
