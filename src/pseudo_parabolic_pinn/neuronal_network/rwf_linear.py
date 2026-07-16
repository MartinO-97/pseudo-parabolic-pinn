import torch 
import torch.nn as nn

class RWFLinear(nn.Module):

    def __init__(self, 
                 in_features: int,
                 out_features: int, 
                 bias: bool, 
                 mean: float,
                 std: float) -> None:
        
        r""" Initilization of a linear layer in a network employing
        random weight facotrization.

        Args:
            in_features (int): Number of input neurons.
            out_features (int): Number of output neurons.
            bias (bool): If ``True``, a bias is intialized.
            mean (float): Mean value for normal distribution.
            std (float): Standard deviation for normal distribution.
        """

        super(RWFLinear, self).__init__()

        # Initialize weight matrix
        W = torch.empty((out_features, in_features))
        nn.init.xavier_uniform_(W)

        # Initialize scale factors s ~ N(\mu, \sigma)
        s_init = torch.randn(out_features) * std + mean

        # Initialize matrix V
        V_init = torch.exp(-s_init)[:, None] * W

        # Save parameters s and V
        self.s = nn.Parameter(s_init)
        self.V = nn.Parameter(V_init)

        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = self.register_parameter("bias", None)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        r""" Forward function
        
        Args: 
            x (torch.Tensor): Input Tensor.

        Returns
            torch.Tensor: Output of the linear layer using Random Weight Factorization.
        """

        return nn.functional.linear(x, 
                                    torch.exp(self.s)[:,None]*self.V, 
                                    self.bias)