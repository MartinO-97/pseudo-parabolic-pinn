import torch 
import torch.nn as nn
from .time_dependent_pde import TimeDependentPDE
from typing import Callable
from ..neuronal_network import PinnNetwork

class ParabolicPDE(TimeDependentPDE):

    def __init__(self, 
                 F: Callable[[torch.Tensor], torch.Tensor], 
                 u0: Callable[[torch.Tensor], torch.Tensor], 
                 Psi: Callable[[torch.Tensor], torch.Tensor],
                 a: Callable[[torch.Tensor], torch.Tensor],
                 T: float, 
                 spatial_interval: tuple[float, float],
                 u: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 u_x: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:

        r""" Initialization of the one-dimensional pseudo-parabolic 
        PDE:
            Ku = u_t - u_xx + au = F on \Omega x (0,T]
            u(x,0) = u_0(x) on \overline{\Omega}
            u(x,t) = Psi(x) for (x,t) \in \pt \Omega x [0,T]. 
            
        Args: 
            F (Callable[[torch.Tensor], torch.Tensor]): The source
                function F.
            u0 (Callable[[torch.Tensor], torch.Tensor]): The initial
                condition u0.
            Psi (Callable[[torch.Tensor], torch.Tensor]): The boundary
                condition Psi.
            a (Callable[[torch.Tensor], torch.Tensor]): The function a.
            u (Callable[[torch.Tensor], torch.Tensor] | None): The exact
                solution. Defaults to ``None``.  
            u_x (Callable[[torch.Tensor], torch.Tensor] | None): The first 
                derivative of the exact solution with respect to the
                spatial variable. Defaults to ``None``.  
        """

        super(ParabolicPDE, self).__init__(F, u0, Psi, T, spatial_interval, u)

        self._a = a
        self._u_x = u_x

    def func_a(self, 
               x_points: torch.Tensor) -> torch.Tensor:
    
        """ The function a.
        
        Args:
            x_points (torch.Tensor): The points in the spatial domain where 
                a shall be evaluated at. Must be of shape (N,1).
        
        Returns:
            torch.Tensor: The function a evaluated at x_points.
        """
        
        return self._a(x_points)
    
    def func_u_x(self,
                 xt_points) -> torch.Tensor | None:

        """ The first derivative of exact solution of the PDE with respect to
        the spatial variable.

        Args:
            xt_points (torch.Tensor): The points in the space-time
                domain where u shall be evaluated at. Must be of
                shape (N,2), where the first column is associated
                with the space variable and the second column with
                the time variable.
        
        Returns:
            torch.Tensor | None: The dirst derivative evaluated at 
                xt_points. If no exact solution is known, ``None``
                is returned. 
        """
        if self._u_x is None:
            return None
        else:
            return self._u_x(xt_points)
        
    def pde_operator_to_nn(self,
                           xt_points: torch.Tensor, 
                           pinn_network: PinnNetwork) -> torch.Tensor:
        
        """ The pseudo-parabolic operator applied to the PINN approximation 
        u_{nn}, i.e.:
            -u_{nn}_xxt + au_{nn}_t - u_{nn}_xx + au_{nn} = F_{nn}.

        Args: 
            xt_points (torch.Tensor): Points in the space-time domain
                where the residual shall be evaluated at.
            pinn_network (PinnNetwork): The PINN which shall be trained
        """

        # Compute predicition of the model
        u_nn = pinn_network(xt_points)

        # Compute loss
        grad_u_nn = torch.autograd.grad(u_nn, xt_points, grad_outputs=torch.ones_like(u_nn), create_graph=True)[0]
        du_nn_dx = grad_u_nn[:, 0:1]
        du_nn_dt = grad_u_nn[:, 1:2]

        grad_du_nn_dx = torch.autograd.grad(du_nn_dx, xt_points, grad_outputs=torch.ones_like(du_nn_dx), create_graph=True)[0]
        du_nn_dxx = grad_du_nn_dx[:, 0:1]

        F_nn = du_nn_dt - du_nn_dxx + self.func_a(xt_points[:,0:1]) * u_nn
        
        return F_nn