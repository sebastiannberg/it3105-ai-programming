import re
import ast
import json
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PokerDataset(Dataset):

    def __init__(self, csv_filename: str, mode: str, param_file: str = None):
        super().__init__()
        self.current_dir = os.path.dirname(__file__)
        self.dataset_path = os.path.join(self.current_dir, "raw", csv_filename)
        self.mode = mode

        TEST_SIZE = 0.1
        VALIDATION_SIZE = 0.1
        RANDOM_STATE = 10

        full_data = pd.read_csv(self.dataset_path)

        total_count = len(full_data)
        test_count = int(np.floor(total_count * TEST_SIZE))
        validation_count = int(np.floor(total_count * VALIDATION_SIZE))

        train_data, temp_data = train_test_split(full_data, test_size=test_count + validation_count, shuffle=True, random_state=RANDOM_STATE)
        validation_data, test_data = train_test_split(temp_data, test_size=test_count, shuffle=False, random_state=RANDOM_STATE)

        self.train = train_data
        self.validation = validation_data
        self.test = test_data

        if mode == "train":
            self.calculate_normalization_params()
            self.data = self.train
        elif mode in ["validation", "test"]:
            if param_file is None:
                raise ValueError("Parameter file must be provided for validation and test modes")
            with open(param_file, 'r') as f:
                self.params = json.load(f)
            self.data = self.validation if mode == "validation" else self.test
        else:
            raise ValueError("Mode should be 'train', 'validation', or 'test'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        public_cards_string = self.data.iloc[idx]["public_cards"]
        public_cards_list = ast.literal_eval(public_cards_string)
        public_cards_vector = self.public_cards_to_one_hot(public_cards_list)

        # Deserialize r1, r2, v1, and v2 from string representations of lists to numpy arrays
        r1 = np.array(ast.literal_eval(self.data.iloc[idx]["r1"]), dtype=np.float64)
        r2 = np.array(ast.literal_eval(self.data.iloc[idx]["r2"]), dtype=np.float64)
        v1 = np.array(ast.literal_eval(self.data.iloc[idx]["v1"]), dtype=np.float64)
        v2 = np.array(ast.literal_eval(self.data.iloc[idx]["v2"]), dtype=np.float64)

        pot = np.array([self.data.iloc[idx]["pot"]], dtype=np.float64)
        pot = self.normalize(pot, self.params['pot_mean'], self.params['pot_std'])

        return torch.from_numpy(r1).float(), torch.from_numpy(r2).float(), torch.from_numpy(public_cards_vector).float(), torch.from_numpy(pot).float(), torch.from_numpy(v1).float(), torch.from_numpy(v2).float()

    def card_to_index(self, card: str) -> int:
        rank_values = {"9": 0, "10": 1, "J": 2, "Q": 3, "K": 4, "A": 5}
        suit_values = {"diamonds": 0, "hearts": 6, "clubs": 12, "spades": 18}
        # Using regex to match the rank and suit
        match = re.match(r"(\d+|[AJKQ])(diamonds|hearts|clubs|spades)$", card)
        if match:
            rank, suit = match.groups()
            return rank_values[rank] + suit_values[suit]
        raise ValueError(f"Invalid card format: {card}")

    def public_cards_to_one_hot(self, public_cards) -> np.ndarray:
        one_hot_vector = np.zeros(24, dtype=np.int8)
        for card in public_cards:
            index = self.card_to_index(card)
            one_hot_vector[index] = 1

        return one_hot_vector

    def calculate_normalization_params(self):
        pots = self.train["pot"].values

        self.params = {
            'pot_mean': np.mean(pots),
            'pot_std': np.std(pots)
        }

        # Save parameters to a file
        param_file = os.path.join(self.current_dir, "normalization_params.json")
        with open(param_file, 'w') as f:
            json.dump(self.params, f)

    def normalize(self, data, mean, std):
        if isinstance(std, (list, np.ndarray)):
            std = np.array(std)
            zero_std_mask = (std == 0)
            std[zero_std_mask] = 1  # Avoid division by zero
            return (data - mean) / std
        else:
            return (data - mean) / std if std > 0 else data
