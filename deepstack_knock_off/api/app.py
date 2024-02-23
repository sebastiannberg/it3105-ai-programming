from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_caching import Cache

import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_state_manager import PokerStateManager
from games.poker.poker_oracle import PokerOracle

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
        game_manager.init_poker_game()
        game_manager.assign_blind_roles()
        game_manager.perform_blind_bets()
        game_manager.deal_cards()
        game_manager.assign_active_player()
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
def legal_action():
    serialized_game_manager = cache.get("game_manager")
    if serialized_game_manager:
        game_manager: PokerGameManager = pickle.loads(serialized_game_manager)
        current_state = game_manager.game
        current_player = game_manager.game.active_player
        legal_actions = PokerStateManager.find_all_legal_actions(state=current_state, player=current_player, json=True)
        return jsonify(legal_actions), 200
    else:
        return jsonify({"error": "Failed to fetch possible actions"}), 404

if __name__ == "__main__":
    app.run(debug=True)
