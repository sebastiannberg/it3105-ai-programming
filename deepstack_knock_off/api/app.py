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
from games.poker_game_service import PokerGameService

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
        game_manager = PokerGameManager(rules, PokerOracle())
        game_service = PokerGameService(game_manager)

        game_service.start_game()

        cache.set("game_manager", pickle.dumps(game_manager))
        return jsonify({"message": "Game started successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/game-state", methods=["GET"])
def game_state():
    try:
        serialized_game_manager = cache.get("game_manager")
        game_manager = pickle.loads(serialized_game_manager)
        game_service = PokerGameService(game_manager)

        game_state_json = game_service.jsonify_poker_game()

        return jsonify(game_state_json), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/legal-actions", methods=["GET"])
def legal_actions():
    try:
        serialized_game_manager = cache.get("game_manager")
        game_manager = pickle.loads(serialized_game_manager)
        game_service = PokerGameService(game_manager)

        legal_actions = game_service.legal_actions()

        return jsonify(legal_actions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/apply-action", methods=["POST"])
def apply_action():
    data = request.json
    try:
        serialized_game_manager = cache.get("game_manager")
        game_manager = pickle.loads(serialized_game_manager)
        game_service = PokerGameService(game_manager)

        result = game_service.apply_action(data)

        cache.set("game_manager", pickle.dumps(game_manager))
        if result:
            return jsonify(result), 200
        else:
            return jsonify({"message": "Action not found or not allowed"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/next-round", methods=["POST"])
def next_round():
    try:
        serialized_game_manager = cache.get("game_manager")
        game_manager = pickle.loads(serialized_game_manager)
        game_service = PokerGameService(game_manager)

        game_service.next_round()

        cache.set("game_manager", pickle.dumps(game_manager))
        return jsonify({"message": "Round ended successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ai-decision", methods=["GET"])
def ai_decision():
    try:
        serialized_game_manager = cache.get("game_manager")
        game_manager = pickle.loads(serialized_game_manager)
        game_service = PokerGameService(game_manager)

        ai_action = game_service.ai_decision()

        return jsonify(ai_action), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


    if not isinstance(game_manager.game.active_player, AIPlayer):
        return jsonify({"error": "It's not AI's turn."}), 400

    legal_actions = PokerStateManager.find_all_legal_actions(game_manager.game, game_manager.game.active_player, game_manager.rules)

    if game_manager.rules["ai_strategy"] == "rollout":
        selected_action = game_manager.game.active_player.make_decision_rollouts(game_manager.oracle, game_manager.game.public_cards, len(game_manager.game.round_players)-1, legal_actions)

    print(selected_action.name, selected_action.player)

    # Apply the action
    cache.set("game_manager", pickle.dumps(game_manager))
    return jsonify({"message": "AI decision made.", "action_name": selected_action.name}), 200


if __name__ == "__main__":
    app.run(debug=True)
