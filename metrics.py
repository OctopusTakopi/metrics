"""Weighted R² functions."""

import numpy as np
import torch
import torch.nn as nn


def r2_weighted(
    y_true: np.array,
    y_pred: np.array,
    sample_weight: np.array
) -> float:
    """Compute the weighted R² score.

    Args:
        y_true (np.array): Ground truth values.
        y_pred (np.array): Predicted values.
        sample_weight (np.array): Weights for each observation.

    Returns:
        float: Weighted R² score.
    """
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (
        np.average((y_true) ** 2, weights=sample_weight) + 1e-38
    )
    return r2

def r2_weighted_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sample_weight: torch.Tensor
) -> torch.Tensor:
    """Compute the weighted R² score using PyTorch tensors.

    Args:
        y_true (torch.Tensor): Ground truth tensor.
        y_pred (torch.Tensor): Predicted tensor.
        sample_weight (torch.Tensor): Weights for each observation (same shape as y_true).

    Returns:
        torch.Tensor: Weighted R² score.
    """
    numerator = torch.sum(sample_weight * (y_pred - y_true) ** 2)
    denominator = torch.sum(sample_weight * (y_true) ** 2) + 1e-38
    r2 = 1 - (numerator / denominator)
    return r2

class WeightedR2Loss(nn.Module):
    """PyTorch loss function for weighted R²."""
    def __init__(self, epsilon: float = 1e-38) -> None:
        """
        Initialize the WeightedR2Loss class.

        Args:
            epsilon (float, optional): Small constant added to the denominator 
                for numerical stability. Defaults to 1e-38.
        """
        super(WeightedR2Loss, self).__init__()
        self.epsilon = epsilon

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute the weighted R² loss.

        Args:
            y_true (torch.Tensor): Ground truth tensor.
            y_pred (torch.Tensor): Predicted tensor.
            weights (torch.Tensor): Weights for each observation (same shape as y_true).

        Returns:
            torch.Tensor: Computed weighted R² loss.
        """
        numerator = torch.sum(weights * (y_pred - y_true) ** 2)
        denominator = torch.sum(weights * (y_true) ** 2) + 1e-38
        loss = numerator / denominator
        return loss

### Custom loss function AdjLoss2 of the article https://ssrn.com/abstract=3973086
# Sigmoid-like adjustment
# In the article : beta is 2.5 or 2.25 
class AdjMSELoss2(nn.Module):
    def __init__(self):
        super(AdjMSELoss2, self).__init__()
                
    def forward(self, outputs, labels):
        outputs = torch.squeeze(outputs)
        beta = 2.5
        loss = (outputs - labels)**2
        adj_loss = beta - (beta - 0.5) / (1 + torch.exp(10000 * torch.mul(outputs, labels)))
        loss = beta * loss /(1+adj_loss) 
        return torch.mean(loss)