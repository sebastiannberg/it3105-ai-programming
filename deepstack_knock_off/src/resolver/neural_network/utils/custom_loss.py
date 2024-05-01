import torch
import torch.nn as nn


class CustomLoss(nn.Module):

    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function

    def forward(self, predicted_v1, predicted_v2, v1, v2, utility_sum):
        # Standard loss for regression
        loss_v1 = self.loss_function(predicted_v1, v1)
        loss_v2 = self.loss_function(predicted_v2, v2)
        loss_utility_sum = self.loss_function(utility_sum, torch.zeros_like(utility_sum))

        # Additional penalty for non-zero predictions where v1 or v2 is zero
        # Mask v1 and v2 creates boolean list
        mask_v1 = (v1 == 0)
        mask_v2 = (v2 == 0)
        # Multiplying with mask retrieves predicted values that should be zero
        # Added since network in general was bad at predicting zeros
        penalized_loss_v1 = self.loss_function(predicted_v1 * mask_v1, torch.zeros_like(predicted_v1))
        penalized_loss_v2 = self.loss_function(predicted_v2 * mask_v2, torch.zeros_like(predicted_v2))

        # Combine the losses
        total_loss = loss_v1 + loss_v2 + loss_utility_sum + penalized_loss_v1 + penalized_loss_v2
        return total_loss
