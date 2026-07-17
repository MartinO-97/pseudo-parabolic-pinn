import torch
from ..pinn_network import PinnNetwork
from ...pde_classes import TimeDependentPDE

def segemental_pde_loss(pinn_network: PinnNetwork,
                        pde_problem: TimeDependentPDE,
                        xt_points_ppp: torch.Tensor,
                        M: int, 
                        w_weights: torch.Tensor,
                        eps: float) -> tuple[torch.Tensor,
                                             torch.Tensor]:

    r""" Segemental 'Phyiscal' loss function based on the considered pde and update of weights. 
    First, given the approximation u_{nn} of the network, the residual is comptuded by applying the PDE operator
    to (u-u_{nn}). 
    
    Then, the loss function is defined by
    
    Loss_{ppp} = 1/M \sum_{i=1}^M w_i L_r^i_{ppp}

    where L_r^i_{ppp} describes the phsyical loss per segement. The weights w_i are updated in each
    call.

    If M=1 and w_1 = 1, we have the standard MSELoss. 
    
    Args:
        pinn_network (PinnNetwork): The PINN network
        pde_problem (TimeDependentPDE): The considered pde.
        xt_points_ppp (torch.Tensor): The training data
        M (int): Number of temporal segments
        w_weights (torch.Tensor): The weights of the loss sum.
        eps (float): Parameter to update weights 

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple consisting of:
            - **torch.Tensor**: The computed loss Loss_{ppp}
            - **torch.Tensor**: The updated weights w.
    """

    # Get device
    device = xt_points_ppp.device

    # COMPUTE RESIDUAL
    # Apply PDE operator to network prediction
    F_nn = pde_problem.pde_operator_to_nn(xt_points_ppp, pinn_network) 
    
    residual = torch.abs(pde_problem.func_F(xt_points_ppp)-F_nn)**2

    # COMPUTE LOSS PER SEGEMENT
    # Number of points
    N = xt_points_ppp[:, 1:2].shape[0]

    # Determine segement length
    tau = pde_problem.get_final_time / M   

    # Determine in which segement the points are 
    indices = torch.floor(xt_points_ppp[:, 1:2] * 1/tau).to(dtype=torch.int64)
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