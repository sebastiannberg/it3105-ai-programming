from flask import Flask, jsonify, request, send_from_directory

import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_oracle import PokerOracle

app = Flask(__name__)

@app.route("/rules", methods=["GET", "POST"])
def rules():
    if request.method == "GET":
        try:
            return send_from_directory(directory=os.path.join(app.root_path, "data"), path="rules.json", as_attachment=False)
        except:
            return jsonify({"error": "Rules file not found."}), 404
    if request.method == "POST":
        new_rules = request.json
        try:
            with open(os.path.join(app.root_path, "data", "rules.json"), 'w') as rules_file:
                json.dump(new_rules, rules_file, indent=4)
            return jsonify({"message": "Rules updated successfully."}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/start-game", methods=["POST"])
def start_game():
    rules = request.json
    try:
        game_manager = PokerGameManager(rules, PokerOracle())
        return jsonify({"message": "Game started successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
