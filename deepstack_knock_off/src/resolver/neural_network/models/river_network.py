import torch
import torch.nn as nn


class RiverNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_layer_module = nn.Sequential(
            nn.Linear(in_features=577, out_features=1154),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=1154, out_features=800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=552)
        )

    def forward(self, r1, r2, public_cards, pot):
        combined_features = torch.concat([r1, r2, public_cards, pot], dim=1)

        v1_v2 = self.hidden_layer_module(combined_features)

        v1, v2 = v1_v2[:, 0:276], v1_v2[:, 276:]

        player_one_expected_utility = torch.sum(r1*v1, dim=1, keepdim=True)
        player_two_expected_utility = torch.sum(r2*v2, dim=1, keepdim=True)

        utility_sum = player_one_expected_utility + player_two_expected_utility

        return v1, v2, utility_sum
