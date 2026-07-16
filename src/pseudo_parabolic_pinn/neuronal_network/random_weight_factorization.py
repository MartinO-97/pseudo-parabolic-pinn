import torch
import torch.nn as nn

class RandomWeightFactorization(nn.Module):

    def __init__(self, 
                 weight_matrix: torch.Tensor, 
                 mu: float,
                 sigma: float = 0.1) -> None:
        
        r""" Initialization function for the random weight 
        factorization.
        
        Args:
            weight_matrix (torch.Tensor): The weight matrix of the layer
            mu (float): Mean ($\mu$)
            sigma (float): Standard Deviation"""

        super().__init__()

        # Number of rows in the weight matrix
        number_rows = weight_matrix.shape[0]

        # Intialize paramater s
        self.s = nn.Parameter(torch.randn(number_rows)*sigma + mu)

        # Initialize matrix V
        V_init = torch.exp(-self.s)[:,None] * weight_matrix
        self.V = nn.Parameter(V_init)

    def forward(self) -> torch.Tensor:
        
        """ Parametarization of the weight matrix 
        
        Returns:
            torch.Tensor: The factorized weight matrix
        """

        return torch.exp(self.s)[:,None]*self.V