from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from flask_caching import Cache

from collections import OrderedDict
import pickle
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games.poker.poker_game_manager import PokerGameManager

app = Flask(__name__)
CORS(app)
cache = Cache(app, config={"CACHE_TYPE": "simple"})

@app.route("/placeholders", methods=["GET"])
def rules():
    try:
        directory = os.path.join(app.root_path, "data")

        with open(os.path.join(directory, "poker_rules.json"), 'r') as rules_file:
            rules_data = json.load(rules_file, object_pairs_hook=OrderedDict)

        with open(os.path.join(directory, "poker_config.json"), 'r') as config_file:
            config_data = json.load(config_file, object_pairs_hook=OrderedDict)

        combined_data = OrderedDict([
            ("config", config_data),
            ("rules", rules_data)
        ])
        return Response(json.dumps(combined_data, ensure_ascii=False), mimetype='application/json')
    except FileNotFoundError:
        return jsonify({"error": "File not found."}), 404

@app.route("/start-game", methods=["POST"])
def start_game():
    data = request.json
    try:
        game_manager = PokerGameManager(data["config"], data["rules"])
        game_manager.start_game()

        cache.set("game_manager", pickle.dumps(game_manager))
        return jsonify({"message": "Game started successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/game-state", methods=["GET"])
def game_state():
    try:
        serialized_game_manager = cache.get("game_manager")
        game_manager = pickle.loads(serialized_game_manager)

        game_state_json = game_manager.jsonify_poker_game()

        return jsonify(game_state_json), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/legal-actions", methods=["POST"])
def legal_actions():
    data = request.json
    try:
        serialized_game_manager = cache.get("game_manager")
        game_manager = pickle.loads(serialized_game_manager)

        game_manager.assign_legal_actions_to_player(data["name"])

        cache.set("game_manager", pickle.dumps(game_manager))
        return jsonify({"message": "Legal actions assigned to player"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/apply-action", methods=["POST"])
def apply_action():
    data = request.json
    try:
        serialized_game_manager = cache.get("game_manager")
        game_manager = pickle.loads(serialized_game_manager)

        result = game_manager.apply_action(data["player"], data["name"])

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

        game_manager.end_round_next_round()

        cache.set("game_manager", pickle.dumps(game_manager))
        return jsonify({"message": "Round ended successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ai-method", methods=["GET"])
def ai_method():
    try:
        serialized_game_manager = cache.get("game_manager")
        game_manager = pickle.loads(serialized_game_manager)

        ai_method = game_manager.get_ai_method()

        cache.set("game_manager", pickle.dumps(game_manager))
        if ai_method:
            return jsonify(ai_method), 200
        else:
            raise ValueError("It is not AI's turn")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ai-decision", methods=["GET"])
def ai_decision():
    try:
        # Retrieve the AI method from query parameters
        ai_method = request.args.get("method", default=None, type=str)
        if ai_method not in ["rollout", "resolver"]:
            return jsonify({"error": "Invalid AI method. Choose 'rollout' or 'resolver'."}), 400

        serialized_game_manager = cache.get("game_manager")
        game_manager = pickle.loads(serialized_game_manager)

        ai_action = game_manager.get_ai_decision(ai_method)
        print(ai_action)

        cache.set("game_manager", pickle.dumps(game_manager))
        if ai_action:
            return jsonify(ai_action), 200
        else:
            return jsonify({"message": "AI decision could not be made"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
