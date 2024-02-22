from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games.poker.poker_game_manager import PokerGameManager
from games.poker.poker_oracle import PokerOracle

app = Flask(__name__)
CORS(app)

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
        return jsonify({"message": "Game started successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
