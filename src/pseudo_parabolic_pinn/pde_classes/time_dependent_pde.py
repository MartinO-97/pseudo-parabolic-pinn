import torch
from typing import Callable


class TimeDependentPDE():

    def __init__(self,
                 F: Callable[[torch.Tensor], torch.Tensor],
                 u0: Callable[[torch.Tensor], torch.Tensor],
                 Psi: Callable[[torch.Tensor], torch.Tensor],
                 u: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        
        r""" Initialization of a time dependent PDE:
            Ku(x,t) = F(x,t) on \Omega x (0,T].
            u(x,0) = u_0 on \overline{\Omega}.
            u(x,t) = Psi(x,t) on \pt\Omega x [0,T].
        
        Args: 
            F (Callable[[torch.Tensor], torch.Tensor]): The source
                function F.
            u0 (Callable[[torch.Tensor], torch.Tensor]): The initial
                condition u0.
            Psi (Callable[[torch.Tensor], torch.Tensor]): The boundary
                condition Psi.
            u (Callable[[torch.Tensor], torch.Tensor] | None): The exact
                solution. Defaults to ``None``.
            
        """

        self._F = F
        self._u0 = u0
        self._Psi = Psi
        self._u = u

    def func_F(self,
               xt_points: torch.Tensor) -> torch.Tensor:
        
        """ The source function of the PDE
        
        Args:
            xt_points (torch.Tensor): The points in the space-time
                domain where F shall be evaluated at. Must be of
                shape (N,2), where the first column is associated
                with the space variable and the second column with
                the time variable.
        
        Returns:
            torch.Tensor: The source function F evaluated at 
                xt_points
        """

        return self._F(xt_points)
    
    def func_u0(self, 
                x_points: torch.Tensor) -> torch.Tensor:
        
        """ The initial condition.
        
        Args:
            x_points (torch.Tensor): The points in closure of the
                spatial domain where u0 shall be evaluated at. Must be 
                of shape (N,1).
        
        Returns:
            torch.Tensor: The initial condition evaluated at 
                xt_points.
        """
        
        return self._u0(x_points)
    
    def func_Psi(self,
                 xt_points: torch.Tensor) -> torch.Tensor:
        
        """ The boundary condition.
        
        Args:
            xt_points (torch.Tensor): The points in the space-time
                domain where F shall be evaluated at. Must be of
                shape (N,2), where the first column is associated
                with the space variable and the second column with
                the time variable.
        
        Returns:
            torch.Tensor: The boundary condition evaluated at 
                xt_points.
        """
        
        return self._Psi(xt_points)
    
    def func_u(self,
               xt_points) -> torch.Tensor | None:

        """ The exact solution of the PDE. 
        Args:
            xt_points (torch.Tensor): The points in the space-time
                domain where u shall be evaluated at. Must be of
                shape (N,2), where the first column is associated
                with the space variable and the second column with
                the time variable.
        
        Returns:
            torch.Tensor | None: The exact solution evaluated at 
                xt_points. If no exact solution is known, ``None``
                is returned. 
        """
        if self._u is None:
            return None
        else:
            return self._u(xt_points)

    @property
    def exact_solution(self) -> bool:

        """ Query if we have an exact solution.
        
        Returns:
            bool: Returns ``True`` if we have an exact solution, otherwise
                ``False``.
        """
        if self._u is None:
            return False
        else:
            return True
    
    def nn_residual_operator(self):

        """ The residual of the PINN approximation, i.e.:
            Ku_nn(x,t) =: F_nn(x,t). 
        """
        pass

