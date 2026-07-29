import torch
import matplotlib.pyplot as plt
import os

from torch import pi, sin, cos, exp
from ..functions import get_data_homogeneous_dirchlet_smooth_example as get_data
from ..neuronal_network import (ExponentialDecayConfig, GradNormConfig, PinnTrainingConfig,
                                RandomFourierFeatureEmbeddingsConfig, RandomWeightFactorizationConfig,
                                SegmentalPDELossConfig, TrainingHistoryData, PinnNetwork,
                                pinn_training, PinnEvaluation)
from ..pde_classes import PseudoParabolicPDE
from ..numerical_utilities import LobattoQuadrature


def main() -> None:

    r""" A PINN is considered to compute an approximation for the pseuodo-parabolic
    partial differential equation

    -u_{xxt} + au_t - u_{xx} + cu = F on \Omega \times (0,T]
    u(x,0) = u_0(x) for x \in \overline{\Omega}
    u(x,t) = 0 for (x,t) \partial \Omega \times [0,T].
    
    We provide an example where the exact solution is known. Furthermore, we analyze the
    maximum norm error

    max_err = || u - u_{nn}||_{L^\infty(0,T; L^\infty(\Omega))}.

    To train the model we choose (N+1)^2 points randomly in the space-time domain.
    """

    # PREPERATION FOR SAVING THE RESULTS
    results_dir = "results/pseudo_parabolic"

    os.makedirs(results_dir, exist_ok=True)
    file_error_results = results_dir + "/errors.txt"
    open(file_error_results, mode="w")

    # DEFINE DEVICE 
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # CREATE DIFFERENT CONFIURATION FILES
    ed_config = ExponentialDecayConfig(True)
    gn_config = GradNormConfig(True, normalize_weights=True)
    rffe_config = RandomFourierFeatureEmbeddingsConfig(128,0)
    rwf_config = RandomWeightFactorizationConfig(True)
    spl_config = SegmentalPDELossConfig(32)

    list_configs = ["ed_config", "gn_config", "rffe_config", "rwf_config", "spl_config"]
    dict_names = {"ed_config": "Exponetial Decay",
                  "gn_config": "GradNorm",
                  "rffe_config": "Random Fourier feature embedding",
                  "rwf_config": "Random weight factorization",
                  "spl_config": "Causal"}
    
    dict_configs = {"ed_config": ed_config,
                    "gn_config": gn_config,
                    "rffe_config": rffe_config,
                    "rwf_config": rwf_config,
                    "spl_config": spl_config}

    # ----------------------------------------------------------------------------------------------------
    # PINN TRAINING PROCESS WITH DIFFERENT SETUPS
    # ----------------------------------------------------------------------------------------------------
    # NUMBER OF EPOCHS FOR ALL TESTS
    epochs = 200000

    # NUMBER OF CHUNKS FOR THE LOSS GRAPHS PRESENTATION
    chunks = 200

    # PINN MODEL WITHOUT ANY MODIFICATIONS
    print("PINN RARE")
    file = open(file_error_results, "a")
    file.write("PINN WITHOUT MODIFICATIONS:\n")
    file.close()
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
    with open(file_error_results, "a") as file:
        file.write(f"Maximum norm error: {error_max:.3e} \n")
        file.write(f"L^2-norm error: {error_l2:.3e} \n")
        file.write(f"H^1-norm error: {error_h1:.3e} \n")
        file.write("\n")
    print("")

    fig_loss_without = training_history.present_loss_graphs(chunks)
    fig_loss_without.savefig(results_dir + "/loss_without.eps", dpi=300, format="eps")
    fig_loss_without.savefig(results_dir + "/loss_without.png", dpi=300, format="png")

    # PINN WITH MODIFICATIONS BUT WITHOUT LBFGS
    print("PINN WITHOUT LBFGS")
    file = open(file_error_results, "a")
    file.write("PINN WITH MODIFICATIONS BUT WITHOUT LBFGS:\n")
    file.close()
    pinn_training_config = PinnTrainingConfig(epochs=epochs, device=device, ed_config=ed_config,
                                              gn_config=gn_config, rffe_config=rffe_config, 
                                              rwf_config=rwf_config, spl_config=spl_config)

    training_history = TrainingHistoryData()

    pinn_network = PinnNetwork(number_hidden_layers, number_hidden_neurons, pinn_training_config)
    
    pinn_training(pinn_network, pinn_training_config, pseudo_parabolic, training_history)
    
    pinn_evaluator = PinnEvaluation(pinn_network, pseudo_parabolic, spatial_interval,
                                    T, device, 1024)
    pinn_evaluator.compute_max_norm_error()
    pinn_evaluator.compute_l2_norm_error(lobatto20)
    pinn_evaluator.compute_h1_norm_error(lobatto20)
    error_max, error_l2, error_h1 = pinn_evaluator.error_norms
    with open(file_error_results, "a") as file:
        file.write(f"Maximum norm error: {error_max:.3e} \n")
        file.write(f"L^2-norm error: {error_l2:.3e} \n")
        file.write(f"H^1-norm error: {error_h1:.3e} \n")
        file.write("\n")
    print("")

    fig_loss_with = training_history.present_loss_graphs(chunks)
    fig_loss_with.savefig(results_dir + "/loss_with.eps", dpi=300, format="eps")
    fig_loss_with.savefig(results_dir + "/loss_with.png", dpi=300, format="png")

    # PINN WITH MODIFICATIONS AND LBFGS
    print("PINN WITH LBFGS")
    file = open(file_error_results, "a")
    file.write("PINN WITH MODIFICATIONS AND LBFGS:\n")
    file.close()
    pinn_training_config = PinnTrainingConfig(epochs=epochs, device=device, ed_config=ed_config,
                                                gn_config=gn_config, rffe_config=rffe_config, 
                                                rwf_config=rwf_config, spl_config=spl_config,
                                                use_lbfgs=True)

    training_history = TrainingHistoryData()

    pinn_network = PinnNetwork(number_hidden_layers, number_hidden_neurons, pinn_training_config)
    
    pinn_training(pinn_network, pinn_training_config, pseudo_parabolic, training_history)
    
    pinn_evaluator = PinnEvaluation(pinn_network, pseudo_parabolic, spatial_interval,
                                    T, device, 1024)
    pinn_evaluator.compute_max_norm_error()
    pinn_evaluator.compute_l2_norm_error(lobatto20)
    pinn_evaluator.compute_h1_norm_error(lobatto20)
    error_max, error_l2, error_h1 = pinn_evaluator.error_norms
    with open(file_error_results, "a") as file:
        file.write(f"Maximum norm error: {error_max:.3e} \n")
        file.write(f"L^2-norm error: {error_l2:.3e} \n")
        file.write(f"H^1-norm error: {error_h1:.3e} \n")

    fig_loss_lbfgs = training_history.present_loss_graphs(chunks)
    fig_loss_lbfgs.savefig(results_dir + "/loss_lbfgs.eps", dpi=300, format="eps")
    fig_loss_lbfgs.savefig(results_dir + "/loss_lbfgs.png", dpi=300, format="png")

    # GENERATE PICTURES OF THE SOLUTIONS
    fig_exact, fig_nn, fig_error = pinn_evaluator.create_graphical_illustration()
    fig_exact.savefig(results_dir + "/exact_sol.eps", dpi=300, format="eps")
    fig_nn.savefig(results_dir + "/nn_sol.eps", dpi=300, format="eps")
    fig_error.savefig(results_dir + "/error_sol_nn.eps", dpi=300, format="eps")

    fig_exact.savefig(results_dir + "/exact_sol.png", dpi=300, format="png")
    fig_nn.savefig(results_dir + "/nn_sol.png", dpi=300, format="png")
    fig_error.savefig(results_dir + "/error_sol_nn.png", dpi=300, format="png")

    # Generate loss graphs


if __name__ == "__main__":

    main()