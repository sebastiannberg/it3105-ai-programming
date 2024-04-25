import torch
import numpy as np
import os
import json
import re

from resolver.neural_network.models.river_network import RiverNetwork


class Predictor:

    def __init__(self):
        self.saved_models_path = os.path.join(os.path.dirname(__file__), "saved_models")
        self.river_model = "24-04-2024_19-53-45_epoch_125.pt"

    def make_prediction(self, stage, r1, r2, public_cards, pot):
        if stage == "river":
            model_path = os.path.join(self.saved_models_path, "river", self.river_model)
            model = RiverNetwork()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            normalization_params = os.path.join(os.path.dirname(__file__), "data", f"{stage}_normalization_params.json")
            with open(normalization_params, 'r') as f:
                params = json.load(f)
        else:
            raise ValueError("{stage} not recognized.")

        # One hot encoding for public cards
        public_cards_one_hot = np.zeros(24, dtype=np.int8)
        for card in public_cards:
            index = self.card_to_index(card)
            public_cards_one_hot[index] = 1

        # Normalize pot
        pot = np.array([pot], dtype=np.float64)
        pot = (pot - params["pot_mean"]) / params["pot_std"]

        # Create tensors for r1, r2, public cards, pot
        r1_tensor = torch.from_numpy(r1).float()
        r2_tensor = torch.from_numpy(r2).float()
        public_cards_tensor = torch.from_numpy(public_cards_one_hot).float().unsqueeze(0)
        pot_tensor = torch.from_numpy(pot).float().unsqueeze(0)

        with torch.no_grad():
            predicted_v1, predicted_v2, _ = model(r1_tensor, r2_tensor, public_cards_tensor, pot_tensor)

        return predicted_v1.numpy(), predicted_v2.numpy()

    def card_to_index(self, card: str) -> int:
        rank_values = {"9": 0, "10": 1, "J": 2, "Q": 3, "K": 4, "A": 5}
        suit_values = {"diamonds": 0, "hearts": 6, "clubs": 12, "spades": 18}
        # Using regex to match the rank and suit
        match = re.match(r"(\d+|[AJKQ])(diamonds|hearts|clubs|spades)$", card)
        if match:
            rank, suit = match.groups()
            return rank_values[rank] + suit_values[suit]
        raise ValueError(f"Invalid card format: {card}")
