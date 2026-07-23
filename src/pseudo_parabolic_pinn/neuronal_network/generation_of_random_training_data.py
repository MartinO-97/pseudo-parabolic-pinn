import torch 

def generation_of_random_training_data(N: int, 
                                       alpha: float, 
                                       beta: float,
                                       T: float,
                                       device: str) -> tuple[torch.Tensor, torch.Tensor,
                                                          torch.Tensor]: 

    r""" Generation of the training data, where the collocation points are 
    chosen randomly in the space-time domain \overline{\Omega} \times [0,T], 
    where \Omega is a one-dimensional domain. 
    
    Args:
        N (int): The space-time domain is partioned in (N+1)^2 randomly chosen points
        alpha (float): The start of the spatial interval
        beta (float): The end of the spatial interval
        T: The final time T
        device (str): Device where tensors shall be stored

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - **xt_points_ppp** (torch.Tensor): Training data for the PDE itself.
            - **xt_points_init** (torch.Tensor): Training data for the initial condition.
            - **xt_points_boundary** (torch.Tensor): Training data for the boundary condition.
    """

    # Generate training data for the PDE
    # Since torch rand generates random numbers in the interval [0,1), we have to ensure that
    # the training data is within the domain \Omega \times (0,1]. To this end, we introduce a
    # little parameter eps = 10**(-9)
    eps = 10**(-9)
    x_points = torch.rand((N+1)**2-(3*N-1),1)
    x_points = eps + alpha + (beta-alpha-2*eps)*x_points

    t_points = torch.rand((N+1)**2-(3*N-1),1)
    t_points = (1-t_points)*T

    xt_points_ppp = torch.cat((x_points, t_points), dim=1).to(device=device)
    xt_points_ppp = xt_points_ppp.requires_grad_(True)

    # Generate training data for the initial condition
    # We ensure that the boundary values of \Omega are in the training data
    eps = 10**(-9)
    x_points = torch.rand(N-1,1)
    x_points = eps + alpha + (beta-alpha-2*eps)*x_points
    x_points = torch.cat((torch.tensor([[alpha]]), x_points, torch.tensor([[beta]])), dim=0)

    t_points = torch.zeros(N+1,1)
    xt_points_init = torch.cat((x_points, t_points), dim=1).to(device=device)

    # Generate training data for the boundary conditions
    # Similar to the training data for the initial condition, we ensure that 
    # the points (\alpha, 0), (\alpha, T), (\beta, 0) and (\beta, T) are in 
    # in the training data
    t_points = torch.rand(N+1,1)
    t_points = eps + (T-2*eps)*t_points

    xt_points_alpha = torch.cat((alpha*torch.ones(N+1, 1), t_points), dim=1)
    xt_points_beta = torch.cat((beta*torch.ones(N+1, 1), t_points), dim=1)

    xt_points_boundary = torch.cat((xt_points_alpha, xt_points_beta), dim=0).to(device=device)

    return xt_points_ppp, xt_points_init, xt_points_boundary