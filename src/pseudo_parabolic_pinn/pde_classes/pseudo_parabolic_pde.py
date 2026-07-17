from typing import Callable

import torch 
import torch.nn as nn
from .time_dependent_pde import TimeDependentPDE

class PseudoParabolicPDE(TimeDependentPDE):

    def __init__(self, 
                 F: Callable[[torch.Tensor], torch.Tensor], 
                 u0: Callable[[torch.Tensor], torch.Tensor], 
                 Psi: Callable[[torch.Tensor], torch.Tensor],
                 a: Callable[[torch.Tensor], torch.Tensor],
                 c: Callable[[torch.Tensor], torch.Tensor],
                 u: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 u_x: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:

        r""" Initialization of the one-dimensional pseudo-parabolic 
        PDE:
            Ku = -u_xxt + au_t - u_xx + cu = F on \Omega x (0,T]
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
            c (Callable[[torch.Tensor], torch.Tensor]): The function c.
            u (Callable[[torch.Tensor], torch.Tensor] | None): The exact
                solution. Defaults to ``None``.  
            u_x (Callable[[torch.Tensor], torch.Tensor] | None): The first 
                derivative of the exact solution with respect to the
                spatial variable. Defaults to ``None``.  
        """

        super(PseudoParabolicPDE, self).__init__(F, u0, Psi, u)

        self.a = a
        self.c = c
        self.u_x = u_x

    def func_a(self, 
               x_points: torch.Tensor) -> torch.Tensor:
    
        """ The function a.
        
        Args:
            x_points (torch.Tensor): The points in the spatial domain where 
                a shall be evaluated at. Must be of shape (N,1).
        
        Returns:
            torch.Tensor: The function a evaluated at x_points.
        """
        
        return self.a(x_points)
    
    def func_c(self, 
               x_points: torch.Tensor) -> torch.Tensor:
    
        """ The function c.
        
        Args:
            x_points (torch.Tensor): The points in the spatial domain where 
                c shall be evaluated at. Must be of shape (N,1).
        
        Returns:
            torch.Tensor: The function c evaluated at x_points
        """
        
        return self.a(x_points)
    
    def func_u_x(self,
                 xt_points) -> torch.Tensor | None:

        """ The first derivative of exact solution of the PDE with respect to
        the time variable.

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
        if self.u_x is None:
            return None
        else:
            return self.u_x(xt_points)