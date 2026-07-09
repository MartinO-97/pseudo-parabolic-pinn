import torch
import torch.nn as nn
from typing import Callable
from .pinn_network import PinnNetwork

def ppp_loss(device: str,
             pinn_network: PinnNetwork,
             func_a: Callable[[torch.Tensor], torch.Tensor],
             func_c: Callable[[torch.Tensor], torch.Tensor],
             func_f: Callable[[torch.Tensor], torch.Tensor],
             xt_points_ppp: torch.Tensor) -> torch.Tensor:

    r""" 'Phyiscal' loss function based on the pseudo-parabolic equation. Given the approximation 
    u_{nn} of the network, we define
    
    F_{nn} := L \partial_t u_{nn} + Mu_{nn},
    
    where L and M are the two elliptic operators. Then, the loss function is defined by
    
    Loss_{ppp} = 1/N \sum_{i=1}^N |F(x_i, t_i) - F_{nn}(x_i, t_i)|
    
    Args:
        device (str): The device where tensores shall be stored. Can be either 'cpu' or 'cuda'
        pinn_network (PinnNetwork): The PINN network
        func_a (Callable[[torch.Tensor], torch.Tensor]): The function a in the operator L
        func_c (Callable[[torch.Tensor], torch.Tensor]): The function c in the operator M
        func_f (Callable[[torch.Tensor], torch.Tensor]): The source function f or, precisely, F
        xt_points_ppp (torch.Tensor): The training data

    Returns:
        torch.Tensor: The computed loss Loss_{ppp}
    """

    # Extract spatial and temporal points
    x_points = xt_points_ppp[:,0:1].to(device=device).clone().requires_grad_(True)
    t_points = xt_points_ppp[:,1:2].to(device=device).clone().requires_grad_(True)

    xt_points = torch.cat((x_points, t_points), dim=1)

    # Compute predicition of the model
    u_nn = pinn_network(xt_points)

    # Compute loss
    du_nn_dx = torch.autograd.grad(u_nn, x_points, grad_outputs=torch.ones_like(u_nn), create_graph=True)[0]
    du_nn_dt = torch.autograd.grad(u_nn, t_points, grad_outputs=torch.ones_like(u_nn), create_graph=True)[0]
    du_nn_dxx = torch.autograd.grad(du_nn_dx, x_points, grad_outputs=torch.ones_like(du_nn_dx), create_graph=True)[0]
    du_nn_dxxt = torch.autograd.grad(du_nn_dxx, t_points, grad_outputs=torch.ones_like(du_nn_dxx), create_graph = True)[0]

    F_nn = -du_nn_dxxt + func_a(x_points) * du_nn_dt \
        - du_nn_dxx + func_c(x_points) * u_nn 
    
    loss_fn = nn.MSELoss()

    Loss_ppp = loss_fn(F_nn, func_f(xt_points))

    return Loss_ppp
