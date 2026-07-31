import torch

from torch import pi, sin, cos, exp
from ..neuronal_network import (PinnTrainingConfig, TrainingHistoryData, PinnNetwork,
                                pinn_training, PinnEvaluation)
from ..pde_classes import PseudoParabolicPDE
from ..numerical_utilities import LobattoQuadrature


def main() -> None:

    r""" A small demonstration program illustrating the general workflow of the implementation.. A
        PINN is trained to compute an approximation for the pseuodo-parabolic
        partial differential equation
    
        -u_{xxt} + au_t - u_{xx} + cu = F on \Omega \times (0,T]
        u(x,0) = u_0(x) for x \in \overline{\Omega}
        u(x,t) = 0 for (x,t) \partial \Omega \times [0,T].
        
        We provide an example where the exact solution is known. Furthermore, we compute the
        C^([0,T]; C(\overline{\Omega})), C^([0,T]; L^2(\Omega)) and
        C^([0,T]; H^1(\Omega)) errors.
    
        To train the model we choose (N+1)^2 points randomly in the space-time domain.
    """

    # DEFINE DEVICE 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # DEFINE FUNCTIONS, FINAL TIME T AND SPATIAL INTERVAL   
    func_u = lambda xt: exp(xt[:, 1:2])*sin(xt[:, 0:1]*pi)
    func_a = lambda x: exp(x)
    func_c = lambda x: x**2 - 1
    func_u0 = lambda x: func_u(torch.cat((x, torch.zeros_like(x)), dim=1))
    func_Psi = lambda xt: torch.zeros_like(xt[:, 0:1])
    func_u_x = lambda xt: pi*exp(xt[:, 1:2])*cos(xt[:, 0:1]*torch.pi)
    func_F = lambda xt: exp(xt[:,1:2])*sin(xt[:,0:1]*pi)*(2*pi**2+exp(xt[:, 0:1])+(xt[:,0:1]**2-1))

    T = 1
    spatial_interval = (-1,1)
    pseudo_parabolic = PseudoParabolicPDE(func_F, func_u0, func_Psi, func_a, 
                                          func_c, T, spatial_interval, func_u, func_u_x)


    # SET NUMBER OF HIDDEN LAYERS AND NEURONS IN THIS LAYERS
    number_hidden_layers = 3
    number_hidden_neurons = 256

    # CREATE QUADRATURE RULE OBJECT
    lobatto20 = LobattoQuadrature(20, device)

    # ----------------------------------------------------------------------------------------------------
    # TRAIN THE PINN
    # ----------------------------------------------------------------------------------------------------
    # NUMBER OF EPOCHS FOR ALL TESTS
    epochs = 1000

    # PINN MODEL 
    pinn_training_config = PinnTrainingConfig(epochs=epochs, device=device)

    training_history = TrainingHistoryData()

    pinn_network = PinnNetwork(number_hidden_layers, number_hidden_neurons, pinn_training_config)
    
    pinn_training(pinn_network, pinn_training_config, pseudo_parabolic, training_history)
    
    pinn_evaluator = PinnEvaluation(pinn_network, pseudo_parabolic, spatial_interval,
                                    T, device, 1024)
    pinn_evaluator.compute_max_norm_error()
    pinn_evaluator.compute_l2_norm_error(lobatto20)
    pinn_evaluator.compute_h1_norm_error(lobatto20)
    error_max, error_l2, error_h1 = pinn_evaluator.error_norms

    print(f"Maximum norm error : {error_max:.3e}")
    print(f"L2 norm error      : {error_l2:.3e}")
    print(f"H1 norm error      : {error_h1:.3e}")

if __name__ == "__main__":

    main()