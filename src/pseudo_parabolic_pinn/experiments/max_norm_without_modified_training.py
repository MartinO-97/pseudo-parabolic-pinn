import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import optim
from ..functions import get_data_homogeneous_dirchlet_smooth_example as get_data
from ..neuronal_network import PinnNetwork, ppp_loss, intial_loss, boundary_loss, \
    generation_of_random_training_data


def main(number_hidden_layers: int,
         number_hidden_neurons: int, 
         learn_rate: float,
         epochs: int):

    r""" A PINN is considered to compute an approximation for the pseuodo-parabolic
    partial differential equation

    -u_{xxt} + au_t - u_{xx} + cu = F on \Omega \times (0,T]
    u(x,0) = u_0(x) for x \in \overline{\Omega}
    u(x,t) = 0 for (x,t) \partial \Omega \times [0,T].
    
    We provide an example where the exact solution is known. Furthermore, we analyze the
    maximum norm error

    max_err = || u - u_{nn}||_{L^\infty(0,T; L^\infty(\Omega))}.

    To train the model we choose (N+1)^2 points randomly in the space-time domain.

    Args:
        number_hidden_layers (int): The number of hidden layers in our neuronal network
        number_hidden_neurons (int): The number of hidden neurons in our neuronal network
        learn_rate (float): The learn rate
        epochs (int): The number of learning epochs.
    """

    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data 
    func_u, func_u_xx, func_u_xxt, func_u_t, func_u0, func_Psi, func_a, func_c, func_f,\
        alpha, beta, T = get_data()

    N = 128

    # Instantiate model and optimizer 
    pinn_network = PinnNetwork(number_hidden_layers, number_hidden_neurons).to(device=device)
    optimizer = optim.Adam(pinn_network.parameters(), lr=learn_rate)

    # Lists to store losses
    list_loss_complete = []
    list_loss_ppp = []
    list_loss_init = []
    list_loss_boundary = []

    # Train the model
    pinn_network.train()

    for epoch in range(epochs):
        
        print(epoch)

        optimizer.zero_grad()

        # Generate training data
        xt_points_ppp, xt_points_init, xt_points_boundary = \
            generation_of_random_training_data(N, alpha, beta, T)
        
        # Compute loss
        loss_ppp = ppp_loss(device, pinn_network, func_a, func_c, func_f, xt_points_ppp)
        loss_init = intial_loss(func_u0, device, xt_points_init, pinn_network)
        loss_boundary = boundary_loss(func_Psi, device, xt_points_boundary, pinn_network)
        loss_complete = loss_ppp + loss_init + loss_boundary

        list_loss_ppp.append(loss_ppp.item())
        list_loss_init.append(loss_init.item())
        list_loss_boundary.append(loss_boundary.item())
        list_loss_complete.append(loss_complete.item())

        # Backward propagation
        loss_complete.backward()
        optimizer.step()

    # Evaluation of the results
    pinn_network.eval()
    with torch.no_grad():

        # Generate evaluation points
        x_points = np.linspace(alpha, beta, 512, endpoint=True)
        t_points = np.linspace(0,T, 512, endpoint=True)

        x_points, t_points = np.meshgrid(x_points, t_points)

        xt_points_eval = np.column_stack((x_points.flatten(), t_points.flatten()))
        xt_points_eval = torch.from_numpy(xt_points_eval).to(device=device, dtype=torch.float32)

        # Compute model predicition
        xt_results_eval = pinn_network(xt_points_eval)

        # Compute maximum norm error
        max_norm_error = torch.max(torch.abs(xt_results_eval-func_u(xt_points_eval)))

        # plot results
        fig, ax = plt.subplots()

        ax.plot([i for i in range(1,epochs+1)], list_loss_complete, label="complete")
        ax.plot([i for i in range(1,epochs+1)], list_loss_ppp, label="ppp")
        ax.plot([i for i in range(1,epochs+1)], list_loss_boundary, label="boundary")
        ax.plot([i for i in range(1,epochs+1)], list_loss_init, label="initial")

        ax.legend(loc="best")
        ax.set_facecolor("lightgrey")
        ax.set_xscale("log")
        ax.set_yscale("log")

        print(f"{max_norm_error:.3e}")

        plt.show()

    return


if __name__ == "__main__":

    main(4, 64, 0.001, 10000)