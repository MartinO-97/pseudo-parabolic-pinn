import torch
from torch import pi, exp, sin
from typing import Callable, TypeAlias

PDEconfig: TypeAlias = tuple[Callable[[torch.Tensor], torch.Tensor],
                  Callable[[torch.Tensor], torch.Tensor],
                  Callable[[torch.Tensor], torch.Tensor],
                  Callable[[torch.Tensor], torch.Tensor],
                  Callable[[torch.Tensor], torch.Tensor],
                  Callable[[torch.Tensor], torch.Tensor],
                  Callable[[torch.Tensor], torch.Tensor],
                  Callable[[torch.Tensor], torch.Tensor],
                  Callable[[torch.Tensor], torch.Tensor],
                  float, 
                  float, 
                  float]

def get_data_homogeneous_dirchlet_smooth_example() -> PDEconfig:

    r""" Returns functions of the pseudo-parabolic PDE

    -u_{xtt} + e^x u_t - u_{xx} + x u = e^t \sin(x \pi), on (-1,1) \times (0,1]
    u(x,0) = \sin(x \pi), on \overline{\Omega}
    u(x,t) = 0, for (x,t) \in {-1,1} \times [0,1].
    
    The exact solution is given by

    u(x,t) = e^t \sin(x\pi).

    Returns:
        PDEconfig: A tuple containing the following components in order:
            - **func_u** (Callable[[torch.Tensor], torch.Tensor]): The exact solution u(x,t).
            - **func_u_xx** (Callable[[torch.Tensor], torch.Tensor]): The second spatial derivative u_{xx}(x,t).
            - **func_u_xxt** (Callable[[torch.Tensor], torch.Tensor]): The mixed third-order derivative u_{xxt}(x,t).
            - **func_u_t** (Callable[[torch.Tensor], torch.Tensor]): The temporal derivative u_t(x,t).
            - **func_u0** (Callable[[torch.Tensor], torch.Tensor]): The initial condition u_0(x).
            - **func_Psi** (Callable[[torch.Tensor], torch.Tensor]): The boundary condition \Psi(x,t).
            - **func_a** (Callable[[torch.Tensor], torch.Tensor]): The coefficient function a(x) = e^x.
            - **func_c** (Callable[[torch.Tensor], torch.Tensor]): The coefficient function c(x) = x.
            - **func_f** (Callable[[torch.Tensor], torch.Tensor]): The source function f(x,t).
            - **alpha** (float): The lower bound of the spatial domain.
            - **beta** (float): The upper bound of the spatial domain.
            - **T** (float): The final time.

    """

    # The space interval [\alpha, \beta]
    alpha: float = -1.0
    beta: float = 1.0

    # The final time T
    T: float = 1.0

    def func_u(xt_points: torch.Tensor) -> torch.Tensor:

        """ The exact solution u """
        x = xt_points[:,0:1]
        t = xt_points[:,1:2]

        return exp(t) * sin(x*pi) 
    
    def func_u_xx(xt_points: torch.Tensor) -> torch.Tensor:

        """ The derivative u_{xx} """
        x = xt_points[:,0:1]
        t = xt_points[:,1:2]

        return -pi**2 * exp(t) * sin(x*pi) 

    def func_u_xxt(xt_points: torch.Tensor) -> torch.Tensor:

        """ The derivative u_{xxt} """

        return func_u_xx(xt_points)

    def func_u_t(xt_points: torch.Tensor) -> torch.Tensor:

        """ The derivative u_t """

        return func_u(xt_points)
    
    def func_u0(x_points: torch.Tensor) -> torch.Tensor:

        """ The intial condition """
        
        xt_points = torch.cat((x_points, torch.zeros_like(x_points)), dim=1)

        return func_u(xt_points)
    
    def func_Psi(xt_points: torch.Tensor) -> torch.Tensor:

        """ The boundary conditions """

        return torch.zeros((xt_points.shape[0], 1))

    def func_a(xt_points) -> torch.Tensor:

        """ The function a"""

        return torch.exp(xt_points[:,0:1])

    def func_c(xt_points) -> torch.Tensor:

        """ The function c"""

        return xt_points[:,0:1]

    def func_f(xt_points: torch.Tensor) -> torch.Tensor:

        """ The source function F"""
        f_result = -func_u_xxt(xt_points) + func_a(xt_points) * func_u_t(xt_points) \
                   -func_u_xx(xt_points) + func_c(xt_points) * func_u(xt_points)
        
        return f_result

    return func_u, func_u_xx, func_u_xxt, func_u_t, func_u0, func_Psi, func_a, func_c, func_f, \
           alpha, beta, T