import torch
import torch.nn as nn

class PinnNetwork(nn.Module):

    """ A Phsysical-Informed Neuronal Network (PINN) to compute an approximate solution 
    for a pseudo-parabolic equation in one spatial dimension. 
    
    The network is a fully connected architecture and uses the Tanh activation function.
    """

    def __init__(self, 
                 number_hidden_layers: int, 
                 number_neurons: int):
        
        """ Initialization of the PINN network with a dynamic number of hidden layers
        and neurons in the hidden layers.
        
        Args:
            number_hidden_layers (int): The number of hidden layers
            number_neurons (int): The number of neurons per hidden layer. 
        """

        super(PinnNetwork, self).__init__()
        
        # ASSEMBLE LAYERS
        layer_framework = []

        # Input Layer 
        # Since we consider a pseudo-parabolic equation on \Omega x (0,T],
        # where \Omega is an interval, we need two neurons in the input 
        # layer
        layer_framework.append(nn.Linear(2, number_neurons, bias=True))
        layer_framework.append(nn.Tanh())        

        # Hidden Layers
        for _ in range(number_hidden_layers):
            layer_framework.append(nn.Linear(number_neurons, number_neurons, bias=True))
            layer_framework.append(nn.Tanh())

        # Output Layer
        # Since our solution u(x,t) maps to \RR, we need only one neuron in the output layer
        layer_framework.append(nn.Linear(number_neurons, 1))

        self.pinn = nn.Sequential(*layer_framework)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:

        """ The forward function of the PINN network. 
        
        Args:
            x_t (torch.Tensor): Tensor of shape [any, 2] containing the 
                spatial and temporal coordinates (x,t).
                
        Returns:
            torch.Tensor: Prediction of the network based on the x_t input. """

        result = self.pinn(x_t)
        return result