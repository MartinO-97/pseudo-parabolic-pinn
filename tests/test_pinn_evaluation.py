import pytest
import torch

from matplotlib.figure import Figure
from pseudo_parabolic_pinn.numerical_utilities import LobattoQuadrature
from pseudo_parabolic_pinn.neuronal_network import PinnEvaluation, PinnNetwork
from pseudo_parabolic_pinn.pde_classes import PseudoParabolicPDE
from pseudo_parabolic_pinn.neuronal_network import PinnTrainingConfig

def test_pinn_evaluation():

    r""" Test if PinnEvaluation class works properly. """

    pinn_training_config = PinnTrainingConfig()

    pinn_network = PinnNetwork(3, 32, pinn_training_config).to(device="cuda")

    func_u = lambda xt: torch.exp(xt[:, 1:2])*torch.sin(xt[:, 0:1]*torch.pi)
    func_a = lambda x: torch.ones_like(x)
    func_c = lambda x: torch.zeros_like(x)
    func_u0 = lambda x: func_u(torch.cat((x, torch.zeros_like(x)), dim=1))
    func_Psi = lambda xt: torch.zeros_like(xt[:, 0:1])
    func_u_x = lambda xt: torch.pi*torch.exp(xt[:, 1:2])*torch.cos(xt[:, 0:1]*torch.pi)
    func_F = lambda xt: 3*torch.exp(xt[:,1:2])*torch.sin(xt[:,0:1]*torch.pi)

    pseudo_parab_pde = PseudoParabolicPDE(func_F, func_u0, func_Psi, func_a, func_c, 
                                          1, (-1,1), func_u, func_u_x)
    
    pinn_evaluator = PinnEvaluation(pinn_network, pseudo_parab_pde, (-1,1), 1, "cuda", 64)

    lobatto10 = LobattoQuadrature(10, "cuda")

    pinn_evaluator.compute_max_norm_error()
    pinn_evaluator.compute_l2_norm_error(lobatto10)
    pinn_evaluator.compute_h1_norm_error(lobatto10)

    max_norm_error, l2_norm_error, h1_norm_error = pinn_evaluator.error_norms

    assert max_norm_error is not None
    assert l2_norm_error is not None
    assert h1_norm_error is not None
    assert l2_norm_error <= h1_norm_error

    fig_sol, fig_approx, fig_error = pinn_evaluator.create_graphical_illustration()

    assert isinstance(fig_sol, Figure)
    assert isinstance(fig_approx, Figure)
    assert isinstance(fig_error, Figure)
    


    
