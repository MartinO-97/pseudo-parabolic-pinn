import torch 
import torch.nn as nn
from ..pinn_network import PinnNetwork
from typing import Callable

def boundary_loss(func_Psi: Callable[[torch.Tensor], torch.Tensor],
                  xt_points_boundary: torch.Tensor,
                  pinn_network: PinnNetwork) -> torch.Tensor:

    r""" The loss function for the initial condition. To be 
    more precise, let u_{nn} be the prediction of the model.
    Then, we consider

    loss_boundary := 1/N \sum_{i=1}^N |\Psi(x_i, t_i) - u_{nn}(x_i, t_i)|

    Args:
        func_Psi (Callable[[torch.Tensor], torch.Tensor]): The boundary condition
        xt_points_boundary (torch.Tensor): Training data
        pinn_network (PinnNetwork): The PINN network

    Returns:
        loss_init (torch.Tensor): The computed loss Loss_{init}
    """

    # Compute predicition of the model
    u_nn = pinn_network(xt_points_boundary)

    # Compute loss
    loss_fn = nn.MSELoss()

    loss_boundary = loss_fn(func_Psi(xt_points_boundary), u_nn)

    return loss_boundary