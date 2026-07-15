import torch
import torch.nn as nn
from .pinn_network import PinnNetwork

def update_loss_weights(loss_ppp: torch.Tensor,
                        loss_init: torch.Tensor,
                        loss_boundary: torch.Tensor,
                        alpha: float,
                        lambda_ppp: float, 
                        lambda_init: float,
                        lambda_boundary: float,
                        optimizer: torch.optim.Optimizer,
                        pinn_network: PinnNetwork) -> tuple[float, float, float]:

    r""" Update of the loss function weights.
    
    This function updates the weights \lambda_{ppp}, \lambda_{init} and
    \lambda_{boundary} of the loss

    Loss_{complete} = \lambda_{ppp} Loss_{ppp} + \lambda_{init} Loss_{init} 
                      + \lambda_{boundary} Loss_{boundary}.

    Args:
        loss_ppp (torch.Tensor): Physical loss
        loss_init (torch.Tensor): Loss of the intial condition
        loss_boundary (torch.Tensor): Loss of the boundary condition
        alpha (float): Auxiliary parameter to update weights 
        lambda_ppp (float): Current weight \lambda_{ppp}
        lambda_init (float): Current weight \lambda_{init}
        lambda_boundary (float): Current weight \lambda_{boundary}

    Returns:
        tuple[float, float, float]: Tuple consisting of:
            - **float**: Updated weight \lambda_{ppp}
            - **float**: Updated weight \lambda_{init}
            - **float**: Updated weight \lambda_{boundary}
    """

    # Get trainable parameters from the network
    parameters = [p for p in pinn_network.parameters() if p.requires_grad]

    # GRADIENT AND L^2 NORM FOR PHYSICAL LOSS FUNCTION
    optimizer.zero_grad()
    loss_ppp.backward(retain_graph=True) 
    grad_norm_ppp = torch.sqrt(torch.sum(torch.tensor(p.grad.pow(2).sum() for p in parameters if p.grad is not None))).item()

    # GRADIENT AND L^2 NORM FOR INITIAL LOSS FUNCTION
    optimizer.zero_grad()
    loss_init.backward(retain_graph=True)
    grad_norm_init = torch.sqrt(torch.sum(torch.tensor(p.grad.pow(2).sum() for p in parameters if p.grad is not None))).item()

    # GRADIENT AND L^2 NORM FOR BOUNDARY LOSS FUNCTION
    optimizer.zero_grad()
    loss_boundary.backward(retain_graph=True)
    grad_norm_boundary = torch.sqrt(torch.sum(torch.tensor(p.grad.pow(2).sum() for p in parameters if p.grad is not None))).item()

    # SUM OF ALL L^2 NORMS
    sum_norms = grad_norm_ppp + grad_norm_init + grad_norm_boundary

    # COMPUTE AUXILIARY WEIGHTS
    aux_lambda_ppp = sum_norms / grad_norm_ppp
    aux_lambda_init = sum_norms / grad_norm_init
    aux_lambda_boundary = sum_norms / grad_norm_boundary

    # UPDATE WEIGHTS
    lambda_ppp = alpha * lambda_ppp + (1-alpha) * aux_lambda_ppp
    lambda_init = alpha * lambda_init + (1-alpha) * aux_lambda_init
    lambda_boundary = alpha * lambda_boundary + (1-alpha) * aux_lambda_boundary

    return 0.0, 0.0, 0.0