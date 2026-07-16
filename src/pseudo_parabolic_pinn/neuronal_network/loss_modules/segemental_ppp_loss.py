import torch
import torch.nn as nn
from typing import Callable
from ..pinn_network import PinnNetwork

def segemental_ppp_loss(device: str,
                        pinn_network: PinnNetwork,
                        func_a: Callable[[torch.Tensor], torch.Tensor],
                        func_c: Callable[[torch.Tensor], torch.Tensor],
                        func_f: Callable[[torch.Tensor], torch.Tensor],
                        xt_points_ppp: torch.Tensor,
                        M: int, 
                        w_weights: torch.Tensor,
                        T: float,
                        eps: float) -> tuple[torch.Tensor,
                                             torch.Tensor]:

    r""" Segemental 'Phyiscal' loss function based on the pseudo-parabolic equation and update of weights. 
    Given the approximation u_{nn} of the network, we define
    
    F_{nn} := L \partial_t u_{nn} + Mu_{nn},
    
    where L and M are the two elliptic operators. Then, the loss function is defined by
    
    Loss_{ppp} = 1/M \sum_{i=1}^M w_i L_r^i_{ppp}

    where L_r^i_{ppp} describes the phsyical loss per segement. The weights w_i are updated in each
    call.
    
    Args:
        device (str): The device where tensores shall be stored. Can be either 'cpu' or 'cuda'
        pinn_network (PinnNetwork): The PINN network
        func_a (Callable[[torch.Tensor], torch.Tensor]): The function a in the operator L
        func_c (Callable[[torch.Tensor], torch.Tensor]): The function c in the operator M
        func_f (Callable[[torch.Tensor], torch.Tensor]): The source function f or, precisely, F
        xt_points_ppp (torch.Tensor): The training data
        M (int): Number of temporal segments
        w_weights (torch.Tensor): The weights of the loss sum.
        T (float): Final time
        eps (float): Parameter to update weights 

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple consisting of:
            - **torch.Tensor**: The computed loss Loss_{ppp}
            - **torch.Tensor**: The updated weights w.
    """

    # COMPUTE RESIDUAL
    # Extract spatial and temporal points
    x_points = xt_points_ppp[:,0:1].to(device=device).clone().requires_grad_(True)
    t_points = xt_points_ppp[:,1:2].to(device=device).clone().requires_grad_(True)

    xt_points = torch.cat((x_points, t_points), dim=1)

    # Compute predicition of the model
    u_nn = pinn_network(xt_points)

    # Compute residual
    du_nn_dx = torch.autograd.grad(u_nn, x_points, grad_outputs=torch.ones_like(u_nn), create_graph=True, retain_graph=True)[0]
    du_nn_dt = torch.autograd.grad(u_nn, t_points, grad_outputs=torch.ones_like(u_nn), create_graph=True, retain_graph=True)[0]
    du_nn_dxx = torch.autograd.grad(du_nn_dx, x_points, grad_outputs=torch.ones_like(du_nn_dx), create_graph=True, retain_graph=True)[0]
    du_nn_dxxt = torch.autograd.grad(du_nn_dxx, t_points, grad_outputs=torch.ones_like(du_nn_dxx), create_graph = True, retain_graph=True)[0]

    #F_nn = -du_nn_dxxt + func_a(x_points) * du_nn_dt \
    #    - du_nn_dxx + func_c(x_points) * u_nn 
    
    F_nn = du_nn_dt - du_nn_dxx + func_a(x_points) * u_nn 
    
    residual = torch.abs(func_f(xt_points)-F_nn)**2

    # COMPUTE LOSS PER SEGEMENT
    # Number of points
    N = t_points.shape[0]

    # Determine segement length
    tau = T/M   

    # Determine in which segement the points are 
    indices = torch.floor(t_points * 1/tau).to(dtype=torch.int64)
    indices = torch.clamp(indices, min=0, max=M-1).squeeze(-1)

    # Compute loss per segment
    segement_loss = torch.zeros(M, device=device)
    points_per_segment = torch.zeros(M, device=device)

    segement_loss.scatter_add_(0, indices, residual.squeeze(-1))
    points_per_segment.scatter_add_(0, indices, torch.ones(N, device=device))

    points_per_segment = torch.clamp(points_per_segment, min=1.0)

    loss_per_segment = segement_loss/points_per_segment

    # Compute loss
    if len(w_weights.shape) != 2:
        raise ValueError("Weights must be of shape (M,1). Got different shape.")    
    
    Loss_ppp = torch.sum(w_weights.squeeze(-1) * loss_per_segment)/M

    # UPDATE WEIGHTS
    w_weights = torch.cumsum(loss_per_segment.reshape(-1,1), dim=0) - loss_per_segment.reshape(-1,1)
    w_weights = torch.exp(-eps * w_weights)

    return Loss_ppp, w_weights.detach()