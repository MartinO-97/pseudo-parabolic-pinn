import torch 
import torch.nn as nn
from ..pinn_network import PinnNetwork
from typing import Callable

def intial_loss(func_u0: Callable[[torch.Tensor], torch.Tensor],
                xt_points_init: torch.Tensor,
                pinn_network: PinnNetwork) -> torch.Tensor:

    r""" The loss function for the initial condition. To be 
    more precise, let u_{nn} be the prediction of the model.
    Then, we consider

    loss_init := 1/N \sum_{i=1}^N |u_0(x_i) - u_{nn}(x_i, t_i)|

    Args:
        func_u0 (Callable[[torch.Tensor], torch.Tensor]): The initial condition
        device (str): The device where tensores shall be stored. Can be either 'cpu' or 'cuda'
        xt_points_init (torch.Tensor): Training data
        pinn_network (PinnNetwork): The PINN network

    Returns:
        loss_init (torch.Tensor): The computed loss Loss_{init}
    """

    # Compute predicition of the model
    u_nn = pinn_network(xt_points_init)

    # Compute loss
    loss_fn = nn.MSELoss()

    loss_init = loss_fn(func_u0(xt_points_init[:,0:1]), u_nn)

    return loss_init