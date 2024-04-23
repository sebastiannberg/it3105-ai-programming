import re
import ast
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class RiverDataset(Dataset):

    def __init__(self, csv_filename: str, mode: str):
        super().__init__()
        self.dataset_path = os.path.join(os.path.dirname(__file__), "raw", csv_filename)
        self.mode = mode

        TEST_SIZE = 0.2
        VALIDATION_SIZE = 0.1
        RANDOM_STATE = 10

        full_data = pd.read_csv(self.dataset_path)

        # Calculate the exact counts for each subset
        total_count = len(full_data)
        test_count = int(np.floor(total_count * TEST_SIZE))
        validation_count = int(np.floor(total_count * VALIDATION_SIZE))

        train_data, temp_data = train_test_split(full_data, test_size=test_count + validation_count, shuffle=True, random_state=RANDOM_STATE)
        validation_data, test_data = train_test_split(temp_data, test_size=test_count, shuffle=False, random_state=RANDOM_STATE)

        self.train = train_data
        self.validation = validation_data
        self.test = test_data

        if mode == "train":
            self.data = self.train
        elif mode == "validation":
            self.data = self.validation
        elif mode == "test":
            self.data = self.test
        else:
            raise ValueError("Mode should be 'train', 'validation', or 'test'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        public_cards_string = self.data.iloc[idx]["public_cards"]
        public_cards_list = ast.literal_eval(public_cards_string)
        public_cards_vector = self.public_cards_to_one_hot(public_cards_list)

        pot = np.array([self.data.iloc[idx]["pot"]], dtype=np.int8)

        # Deserialize r1, r2, v1, and v2 from string representations of lists to NumPy arrays
        r1 = np.array(ast.literal_eval(self.data.iloc[idx]["r1"]), dtype=np.float64)
        r2 = np.array(ast.literal_eval(self.data.iloc[idx]["r2"]), dtype=np.float64)
        v1 = np.array(ast.literal_eval(self.data.iloc[idx]["v1"]), dtype=np.float64)
        v2 = np.array(ast.literal_eval(self.data.iloc[idx]["v2"]), dtype=np.float64)

        features = np.concatenate([r1, r2, public_cards_vector, pot])
        targets = np.concatenate([v1, v2])

        return torch.from_numpy(features).float(), torch.from_numpy(targets).float()

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


train_dataset = RiverDataset(csv_filename="river_cases_2024-04-22_19-45-01.csv", mode="train")
validation_dataset = RiverDataset(csv_filename="river_cases_2024-04-22_19-45-01.csv", mode="validation")
test_dataset = RiverDataset(csv_filename="river_cases_2024-04-22_19-45-01.csv", mode="test")

train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_data_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
