import torch
import torch.nn as nn
from typing import Callable
from ..pinn_network import PinnNetwork
from ...pde_classes import TimeDependentPDE

def pde_loss(pinn_network: PinnNetwork,
             pde_problem: TimeDependentPDE,
             xt_points_ppp: torch.Tensor) -> torch.Tensor:

    r""" 'Phyiscal' loss function based on the given time dependent PDE. First, given the approximation 
    u_{nn} of the network, the residual is comptuded by applying the PDE operator
    to (u-u_{nn}). Then, the loss function is defined by
    
    Loss_{ppp} = 1/N \sum_{i=1}^N |F(x_i, t_i) - F_{nn}(x_i, t_i)|^2
    
    where F is the source function and F_{nn} is the residual. 

    Args:
        pinn_network (PinnNetwork): The PINN network
        pde_problem (TimeDependentPDE): The time-dependent PDE.
        xt_points_ppp (torch.Tensor): The training data

    Returns:
        torch.Tensor: The computed loss Loss_{ppp}
    """

    # Compute residual
    F_nn = pde_problem.pde_operator_to_nn(xt_points_ppp, pinn_network)
    
    loss_fn = nn.MSELoss()

    Loss_ppp = loss_fn(F_nn, pde_problem.func_F(xt_points_ppp))

    return Loss_ppp
