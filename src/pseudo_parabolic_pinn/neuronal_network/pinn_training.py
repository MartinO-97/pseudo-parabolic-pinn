import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import optim
from .pinn_network import PinnNetwork
from .loss_modules import boundary_loss, segemental_pde_loss as pde_loss, intial_loss
from .generation_of_random_training_data import generation_of_random_training_data
from .data_classes import TrainingHistoryData, PinnTrainingConfig
from .update_loss_weights import update_loss_weights
from ..pde_classes import TimeDependentPDE


def pinn_training(pinn_network: PinnNetwork,
                  pinn_training_config: PinnTrainingConfig,
                  pde_problem: TimeDependentPDE,
                  training_history_data: TrainingHistoryData) -> None:

    r""" Training a PINN to approximate the solution of time dependent
    partial differential equation.

    To train the model we choose (N+1)^2 points randomly in the space-time domain.

    Args:
        number_hidden_layers (int): The number of hidden layers in our neuronal network
        number_hidden_neurons (int): The number of hidden neurons in our neuronal network
        learn_rate (float): The learn rate
        epochs (int): The number of learning epochs.
    """

    # CHOOSE DEVICE
    device = pinn_training_config.device

    # COMPUTE N
    # N is the number of points on the x and t axes
    #N = int(np.log2(pinn_training_config.batch_size)/2)
    N = int(np.sqrt(pinn_training_config.batch_size))

    # Instantiate model and optimizer 
    pinn_network = pinn_network.to(device=device)
    optimizer = optim.Adam(pinn_network.parameters(), 
                           lr=pinn_training_config.learn_rate)
    if pinn_training_config.ed_config.use_ed:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                            pinn_training_config.ed_config.exponential_decay_step_size, 
                                            pinn_training_config.ed_config.gamma)

    # Initialize weights for segmentally pyhical loss
    w_weights = pinn_training_config.spl_config.w_init.clone().to(device=device)

    # Initialize weights for losses
    lambda_ppp, lambda_init, lambda_boundary = pinn_training_config.gn_config.get_initial_lambdas

    # Determine number of epochs for Adam and LBFGS
    adam_epochs = pinn_training_config.epochs
    lbfgs_epochs = 0
    if pinn_training_config.use_lbfgs:
        lbfgs_epochs = 500
        adam_epochs = adam_epochs - lbfgs_epochs

    # Train the model
    pinn_network.train()


    # ----------------------------------------------------------------------------------------------------
    # ADAM PART
    # ----------------------------------------------------------------------------------------------------
    for epoch in range(1, adam_epochs+1):

        if epoch % 10000 == 0:
            print(epoch)
        optimizer.zero_grad()

        # Generate training data
        spatial_interval = pde_problem.spatial_interval
        T = pde_problem.final_time
        xt_points_ppp, xt_points_init, xt_points_boundary = \
            generation_of_random_training_data(N, spatial_interval[0], spatial_interval[1], T, device)
        
        # Compute loss
        loss_pde, w_weights = pde_loss(pinn_network, pde_problem, xt_points_ppp, 
                                       pinn_training_config.spl_config.number_of_segements,
                                       w_weights, 
                                       pinn_training_config.spl_config.spl_eps)
        loss_init = intial_loss(pde_problem.func_u0, xt_points_init, pinn_network)
        loss_boundary = boundary_loss(pde_problem.func_Psi, xt_points_boundary, pinn_network)
        loss_complete = lambda_ppp * loss_pde + lambda_init * loss_init + lambda_boundary * loss_boundary

        training_history_data.update_loss_lists(loss_complete.item(), loss_pde.item(),
                                                loss_init.item(), loss_boundary.item())

        # Update weights for losses
        if (epoch % 1000 == 0) and (pinn_training_config.gn_config.use_grad_norm):
            lambda_ppp, lambda_init, lambda_boundary = update_loss_weights(loss_pde, loss_init, loss_boundary, 
                                                                           pinn_training_config.gn_config.grad_norm_alpha, 
                                                                           lambda_ppp, lambda_init, lambda_boundary, 
                                                                           optimizer, pinn_network, 
                                                                           pinn_training_config.gn_config.normalize_weights)

        # Backward propagation
        optimizer.zero_grad()
        loss_complete.backward()
        optimizer.step()
        if pinn_training_config.ed_config.use_ed:
            scheduler.step()

    # ----------------------------------------------------------------------------------------------------
    # LBFGS PART
    # ----------------------------------------------------------------------------------------------------
    if pinn_training_config.use_lbfgs:

        # Generate training data
        spatial_interval = pde_problem.spatial_interval
        T = pde_problem.final_time
        xt_points_ppp, xt_points_init, xt_points_boundary = \
            generation_of_random_training_data(N, spatial_interval[0], spatial_interval[1], T, device)

        w_weights_fixed = w_weights.clone()

        optimizer = torch.optim.LBFGS(pinn_network.parameters())

        for epoch in range(1, lbfgs_epochs+1):
            print(epoch)

            # Save history once per epoch
            optimizer.zero_grad()
            loss_pde, _ = pde_loss(pinn_network, pde_problem, xt_points_ppp, 
                                    pinn_training_config.spl_config.number_of_segements,
                                    w_weights_fixed, pinn_training_config.spl_config.spl_eps)
            loss_init = intial_loss(pde_problem.func_u0, xt_points_init, pinn_network)
            loss_boundary = boundary_loss(pde_problem.func_Psi, xt_points_boundary, pinn_network)
            loss_complete = lambda_ppp * loss_pde + lambda_init * loss_init + lambda_boundary * loss_boundary
            
            training_history_data.update_loss_lists(
                loss_complete.item(), loss_pde.item(), loss_init.item(), loss_boundary.item())

            # Closure function for LBFGS
            def closure() -> torch.Tensor:
    
                optimizer.zero_grad()

                if not xt_points_ppp.requires_grad:
                    xt_points_ppp.requires_grad_(True)

                # Compute loss
                loss_pde, _ = pde_loss(pinn_network, pde_problem, xt_points_ppp, 
                                                pinn_training_config.spl_config.number_of_segements,
                                                w_weights_fixed, 
                                                pinn_training_config.spl_config.spl_eps)
                loss_init = intial_loss(pde_problem.func_u0, xt_points_init, pinn_network)
                loss_boundary = boundary_loss(pde_problem.func_Psi, xt_points_boundary, pinn_network)
                loss_complete = lambda_ppp * loss_pde + lambda_init * loss_init + lambda_boundary * loss_boundary

                loss_complete.backward()

                return loss_complete

            optimizer.step(closure=closure)
    