import pytest

import torch
from pseudo_parabolic_pinn.neuronal_network import (PinnNetwork, PinnTrainingConfig,
                                                    ExponentialDecayConfig, GradNormConfig,
                                                    RandomFourierFeatureEmbeddingsConfig,
                                                    RandomWeightFactorizationConfig,
                                                    SegmentalPDELossConfig,
                                                    pinn_training, 
                                                    TrainingHistoryData)
from pseudo_parabolic_pinn.pde_classes import PseudoParabolicPDE

@pytest.fixture
def sample_pde() -> PseudoParabolicPDE:

    r""" Creates an easy example of an pseudo-parabolic PDE.
    
    Returns:
        PseudoParabolicPDE: Pseudo parabolic PDE object"""

    # Create pseudo-parabolic example
    func_u = lambda xt: torch.exp(xt[:, 1:2])*torch.sin(xt[:, 0:1]*torch.pi)
    func_a = lambda x: torch.ones_like(x)
    func_c = lambda x: torch.zeros_like(x)
    func_u0 = lambda x: func_u(torch.cat((x, torch.zeros_like(x)), dim=1))
    func_Psi = lambda xt: torch.zeros_like(xt[:, 0:1])
    func_u_x = lambda xt: torch.pi*torch.exp(xt[:, 1:2])*torch.cos(xt[:, 0:1]*torch.pi)
    func_F = lambda xt: 3*torch.exp(xt[:,1:2])*torch.sin(xt[:,0:1]*torch.pi)

    pseudo_parab_pde = PseudoParabolicPDE(func_F, func_u0, func_Psi, func_a, func_c, 
                                          1, (-1,1), func_u, func_u_x)
    
    return pseudo_parab_pde

def test_pinn_training_default_setup(sample_pde):

    r""" Tests if the training with default configurations of the PINN works properly. """

    # TEST #1
    # Set number of epochs, hidden layers and neurons
    epochs = 2
    number_hidden_layers = 3
    number_hidden_neurons = 32
    
    # Initialize network
    pinn_training_config = PinnTrainingConfig(epochs=epochs)
    training_history_data = TrainingHistoryData()
    pinn = PinnNetwork(number_hidden_layers, number_hidden_neurons, pinn_training_config)
    list_init_parameters = [param.clone() for param in pinn.parameters()]
    
    # Train network
    pinn_training(pinn, pinn_training_config, sample_pde, training_history_data)

    lists = training_history_data.loss_lists

    params_changed = False
    length_lists = True
    
    for l in lists:
        if len(l) !=epochs:
            length_lists = False

    for p_init, p_trained in zip(list_init_parameters, pinn.parameters()):
        if not torch.equal(p_init, p_trained):
            params_changed = True
            break

    assert length_lists
    assert params_changed
    

def test_pinn_training_advanced_setup(sample_pde):

    r""" Tests if the training with advanced configurations of the PINN works properly. """

    # Set number of epochs, hidden layers and neurons
    epochs = 1001
    number_hidden_layers = 3
    number_hidden_neurons = 32

    # Initialize network
    rwf_config = RandomWeightFactorizationConfig(use_rwf=True)
    rffe_config = RandomFourierFeatureEmbeddingsConfig(spatial_embeddings=8)
    ed_config = ExponentialDecayConfig(use_ed=True)
    gn_config = GradNormConfig(use_grad_norm=True)
    spl_config = SegmentalPDELossConfig(number_of_segements=8, update_segments_weights=True)
    pinn_training_config = PinnTrainingConfig(epochs=epochs, rwf_config=rwf_config, rffe_config=rffe_config, ed_config=ed_config,
                                              gn_config=gn_config, spl_config=spl_config)
    training_history_data = TrainingHistoryData()
    pinn = PinnNetwork(number_hidden_layers, number_hidden_neurons, pinn_training_config)
    list_init_parameters = [param.clone() for param in pinn.parameters()]
    
    # Train network
    pinn_training(pinn, pinn_training_config, sample_pde, training_history_data)

    lists = training_history_data.loss_lists

    params_changed = False
    length_lists = True
    
    for l in lists:
        if len(l) !=epochs:
            length_lists = False

    for p_init, p_trained in zip(list_init_parameters, pinn.parameters()):
        if not torch.equal(p_init, p_trained):
            params_changed = True
            break

    assert length_lists
    assert params_changed
    
