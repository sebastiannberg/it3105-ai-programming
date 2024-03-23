from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_caching import Cache

import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_oracle import PokerOracle
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
        if ai_action:
            return jsonify(ai_action), 200
        else:
            raise ValueError("It is not AI's turn")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
