import torch
import torch.nn as nn
from .rwf_linear import RWFLinear
from .random_fourier_feature_embeddings import RFFEmbedding

class PinnNetwork(nn.Module):

    """ A Phsysical-Informed Neuronal Network (PINN) to compute an approximate solution 
    for a pseudo-parabolic equation in one spatial dimension. 
    
    The network is a fully connected architecture and uses the Tanh activation function.
    """

    def __init__(self, 
                 number_hidden_layers: int, 
                 number_neurons_hidden_layers: int,
                 bias: bool = True,
                 use_rwf: bool = False,
                 rwf_mean: float = 1.0,
                 rwf_std: float = 0.1,
                 spatial_embeddings: int = 0,
                 temporal_embeddings: int = 0,
                 rffe_std: float = 1.0) -> None:
        
        """ Initialization of the PINN network with a dynamic number of hidden layers
        and neurons in the hidden layers.
        
        Args:
            number_hidden_layers (int): The number of hidden layers
            number_neurons_hidden_layers (int): The number of neurons per hidden layer.
            bias (bool): If ``True`` a bias is used. Defaults to ``True``.
            use_rwf (bool): Shall random weight factorization be used?
                - Yes -> ``True``
                - No -> ``False``
            rwf_mean (float, optional): Mean of the normal distribution used to
                initialize the RWF scaling parameters. Defaults to ``1.0``.
            rwf_std (float, optional): Standard deviation of the normal
                distribution used to initialize the RWF scaling parameters.
                Defaults to ``0.1``
            spatial_embeddings (int): Number of embeddings for 
                the random Fourier feature embeddings for the spatial
                variable. Defaults to ``0``.
            temporal_embeddings (int): Number of embeddings for 
                the random Fourier feature embeddings for the temporal
                variable. Defaults to ``0``.
            rffe_std (float): Standard deviation of the normal 
                distribution employed to intialize the random Fourier
                feature embedding class. Defaults to ``1.0``.
        """

        super(PinnNetwork, self).__init__()

        # INITIALIZE CLASS FOR RANDOM FOURIER FEATURE EMBEDDING
        self.embedding = RFFEmbedding(spatial_embeddings, temporal_embeddings, rffe_std)

        # ASSEMBLE LAYERS
        layer_framework = []

        # Input Layer 
        # Since we consider a pseudo-parabolic equation on \Omega x (0,T],
        # where \Omega is an interval, we need two neurons in the input 
        # layer
        layer_framework.append(self.create_linear_layer(self.embedding.output_dimension, 
                                                        number_neurons_hidden_layers, 
                                                        bias, use_rwf, rwf_mean, rwf_std))
        layer_framework.append(nn.Tanh())        

        # Hidden Layers
        for _ in range(number_hidden_layers):
            layer_framework.append(self.create_linear_layer(number_neurons_hidden_layers,
                                                            number_neurons_hidden_layers,
                                                            bias, use_rwf, rwf_mean, rwf_std))
            layer_framework.append(nn.Tanh())

        # Output Layer
        # Since our solution u(x,t) maps to \RR, we need only one neuron in the output layer
        layer_framework.append(self.create_linear_layer(number_neurons_hidden_layers,
                                                        1, bias, use_rwf, rwf_mean, rwf_std))

        self.pinn = nn.Sequential(*layer_framework)

    @staticmethod
    def create_linear_layer(in_features: int, 
                            out_features: int,
                            bias: bool = True,
                            use_rwf: bool = False,
                            mean: float = 1.0,
                            std: float = 0.1) -> nn.Module:
        
        r""" Create a linear layer.

        Args:
            in_features (int): Number of input neurons
            out_features (int): Number of output neurons
            bias (bool): If ``True`` a bias is used. Defaults to ``True``.
            use_rwf (bool): Shall random weight factorization be used?
                - Yes -> ``True``
                - No -> ``False``
            mean (float, optional): Mean of the normal distribution used to
            initialize the RWF scaling parameters. Defaults to ``1.0``.
            std (float, optional): Standard deviation of the normal
            distribution used to initialize the RWF scaling parameters.
            Defaults to ``0.1``.

        Returns:
            nn.Module: Linear layer, optionally employing random
            weight factorization 
        """

        if use_rwf:
            layer = RWFLinear(in_features, out_features, bias, mean, std)
        else:
            layer = nn.Linear(in_features, out_features, bias)

        return layer


    def forward(self, x_t: torch.Tensor) -> torch.Tensor:

        """ The forward function of the PINN network. 
        
        Args:
            x_t (torch.Tensor): Tensor of shape [any, 2] containing the 
                spatial and temporal coordinates (x,t).
                
        Returns:
            torch.Tensor: Prediction of the network based on the x_t input. """

        result = self.pinn(self.embedding(x_t))
        return result