import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "src")))

from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from resolver.neural_network.data.poker_dataset import PokerDataset
from resolver.neural_network.models.turn_network import TurnNetwork
from resolver.neural_network.utils.custom_loss import CustomLoss


def train():
    print("\033[1;32m" + "="*15 + " Setup " + "="*15 + "\033[0m")
    NUM_EPOCHS = 200
    LOSS_FUNCTION = nn.MSELoss()
    LR = 0.001
    WEIGHT_DECAY = 0.00001
    BATCH_SIZE = 16
    CSV_FILENAME = "turn_cases_2024-04-24_21-53-46.csv"
    NORMALIZATION_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "data", "turn_normalization_params.json")
    SAVED_MODELS_PATH = os.path.join(os.path.dirname(__file__), "saved_models")

    print("Creating datasets")
    train_dataset = PokerDataset(stage= "turn", csv_filename=CSV_FILENAME, mode="train", param_file=NORMALIZATION_PARAMS_PATH)
    validation_dataset = PokerDataset(stage="turn", csv_filename=CSV_FILENAME, mode="validation", param_file=NORMALIZATION_PARAMS_PATH)
    print(f"Train dataset created with {len(train_dataset)} samples")
    print(f"Validation dataset created with {len(validation_dataset)} samples")

    print("Creating data loaders")
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TurnNetwork()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = CustomLoss(loss_function=LOSS_FUNCTION)
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Loss Function: {LOSS_FUNCTION.__class__.__name__}")

    # Initialize variable to track the best validation loss
    best_validation_loss = float('inf')

    # Training phase
    print("\033[1;32m" + "="*15 + " Training " + "="*15 + "\033[0m")
    start_time = time.time()

    total_batches = len(train_data_loader)
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = []

        for batch_idx, (r1, r2, public_cards, pot, v1, v2) in enumerate(train_data_loader):
            print(f"\rBatch {batch_idx+1} of {total_batches}", end="")
            # Zeroing gradients for each minibatch
            optimizer.zero_grad()

            predicted_v1, predicted_v2, utility_sum = model(r1, r2, public_cards, pot)

            loss = criterion(predicted_v1, predicted_v2, v1, v2, utility_sum)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

        average_loss = sum(running_loss) / len(running_loss)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {average_loss}")

        # Validation phase
        model.eval()
        running_validation_loss = []
        with torch.no_grad():
            for r1, r2, public_cards, pot, v1, v2 in validation_data_loader:
                predicted_v1, predicted_v2, utility_sum = model(r1, r2, public_cards, pot)
                loss = criterion(predicted_v1, predicted_v2, v1, v2, utility_sum)
                running_validation_loss.append(loss.item())

        average_validation_loss = sum(running_validation_loss) / len(running_validation_loss)
        print(f"Average Validation Loss: {average_validation_loss}")

        # Model saving logic
        if average_validation_loss < best_validation_loss and (epoch + 1) % 5 == 0:
            best_validation_loss = average_validation_loss
            current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            model_save_path = os.path.join(SAVED_MODELS_PATH, "turn", f"{current_time}_epoch_{epoch+1}.pt")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    end_time = time.time()
    duration = end_time - start_time
    duration_minutes = duration / 60
    print(f"Total training time: {duration_minutes:.2f} minutes")

def test(model_filename: str):
    print("\033[1;32m" + "="*15 + " Testing " + "="*15 + "\033[0m")
    LOSS_FUNCTION = nn.MSELoss()
    BATCH_SIZE = 16
    CSV_FILENAME = "turn_cases_2024-04-24_21-53-46.csv"
    NORMALIZATION_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "data", "turn_normalization_params.json")
    SAVED_MODELS_PATH = os.path.join(os.path.dirname(__file__), "saved_models")

    print("Creating dataset")
    test_dataset = PokerDataset(stage="turn", csv_filename=CSV_FILENAME, mode="test", param_file=NORMALIZATION_PARAMS_PATH)
    print(f"Test dataset created with {len(test_dataset)} samples")

    print("Creating data loaders")
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_path = os.path.join(SAVED_MODELS_PATH, "turn", model_filename)
    model = TurnNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = CustomLoss(loss_function=LOSS_FUNCTION)
    print(f"Model: {model.__class__.__name__}")
    print(f"Loss Function: {LOSS_FUNCTION.__class__.__name__}")

    running_test_loss = []
    comparing = []
    with torch.no_grad():
        for r1, r2, public_cards, pot, v1, v2 in test_data_loader:
            predicted_v1, predicted_v2, utility_sum = model(r1, r2, public_cards, pot)
            comparing.append((v1.numpy(), predicted_v1.numpy()))
            loss = criterion(predicted_v1, predicted_v2, v1, v2, utility_sum)
            running_test_loss.append(loss.item())
            #print(utility_sum)

    average_test_loss = sum(running_test_loss) / len(running_test_loss)
    print(f"Average Test Loss: {average_test_loss}")

    # for v1, predicted_v1 in comparing:
    #     for i in range(BATCH_SIZE):
    #         for j in range(276):
    #             print(v1[i, j], "|", predicted_v1[i, j])

# train()
test("25-04-2024_16-28-13_epoch_180.pt")
