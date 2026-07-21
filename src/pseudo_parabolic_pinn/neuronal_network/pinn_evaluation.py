import torch 
import numpy as np
import matplotlib.pyplot as plt

from .pinn_network import PinnNetwork
from ..pde_classes import TimeDependentPDE
from matplotlib.figure import Figure
from ..numerical_utilities import Quadrature
from mpl_toolkits.mplot3d import Axes3D


class PinnEvaluation():

    r""" Class to evaluate a trained pinn_network."""

    def __init__(self,
                 pinn_network: PinnNetwork,
                 pde_problem: TimeDependentPDE,
                 interval: tuple[float, float],
                 T: float, 
                 device: str, 
                 N: int) -> None:
        
        r""" Initialization of a PinnEvaluation object to 
        evaluate a trained_pinn_network. 
        
        Args:
            trained_pinn_network (PinnNetwork): The trained pinn network
            pde_problem (TimeDependentPDE): The PDE for which the PINN network
                shall generate an approximate solution. 
            interval (tuple[float, float]): The spatial interval
            T (float): The final time ``T``.
            device (str): Device where tensors shall be stored
            N (int): N+1 is number of points in the space and time dimensiona, respectively, 
                to evaluate the approximation    
        """

        self._pinn_network = pinn_network
        self._pde_problem = pde_problem
        self._interval = interval
        self._T = T
        self._N = N
        self._max_norm_error: None | float = None
        self._l2_norm_error: None | float = None 
        self._h1_norm_error: None | float = None

        # Generate points in the space-time domain
        # in order to approximate the error norms and to generate a figure of 
        # the approximate solution
        a = interval[0]
        b = interval[1]
        x_points = np.linspace(a, b, N+1)
        t_points = np.linspace(0, T, N+1)

        x_points = torch.from_numpy(x_points).to(dtype=torch.float32)
        t_points = torch.from_numpy(t_points).to(dtype=torch.float32)

        self._x_points = x_points.to(device=device)
        self._t_points = t_points.to(device=device)

    def _u_pinn_difference_t(self,
                           t: float,
                           x_points: torch.Tensor) -> torch.Tensor:
        
        r""" Function that evaluates ``(u-u_{nn})(t)`` at 
        ``x_points``.
        
        Args:
            t (float): The time point t.
            x_points (torch.Tensor): Points in space where ``(u-u_{nn}(t))``
                shall be evaluated at.
                
        Returns:
            torch.Tensor: ``(u-u_{nn}(t))`` evaluated at ``x_points``.
        """

        if x_points.ndim != 2:
            raise ValueError(f"x_points has wrong shape: {x_points.shape}")
        
        t_points = t * torch.ones_like(x_points)
        xt_points = torch.cat((x_points, t_points), dim=1)

        result = self._pde_problem.func_u(xt_points) - self._pinn_network(xt_points)

        return result
    
    def _dx_u_pinn_difference_t(self,
                                t: float,
                                x_points: torch.Tensor) -> torch.Tensor:
        
        r""" Function that evaluates ``(u_x-u_{nn}_x)(t)`` at 
        ``x_points``.
        
        Args:
            t (float): The time point t.
            x_points (torch.Tensor): Points in space where ``(u-u_{nn}(t))``
                shall be evaluated at.
                
        Returns:
            torch.Tensor: ``(u_x-u_{nn}_x(t))`` evaluated at ``x_points``.
        """

        if x_points.ndim != 2:
            raise ValueError(f"x_points has wrong shape: {x_points.shape}")
        
        x_points.requires_grad_(True)
        t_points = t * torch.ones_like(x_points)
        xt_points = torch.cat((x_points, t_points), dim=1)

        # Compute first derivative of u_{nn} with respect to x_points
        u_nn = self._pinn_network(xt_points)
        dx_u_nn = torch.autograd.grad(u_nn, x_points, grad_outputs=torch.ones_like(u_nn))[0]

        func_u_x_eval = self._pde_problem.func_u_x(xt_points)
        if func_u_x_eval is None:
            raise ValueError("Exact solution of the PDE is not defined")
        
        result = func_u_x_eval - dx_u_nn 

        return result

    def compute_max_norm_error(self) -> None:

        r""" Computes the maximum norm error between the pinn approximation and 
        the exact solution. """

        x_points = self._x_points.repeat((self._N+1)).unsqueeze(-1)
        t_points = self._t_points.repeat_interleave(self._N+1).unsqueeze(-1)

        xt_points = torch.cat((x_points, t_points), dim=1)

        func_u = self._pde_problem.func_u
        if func_u is None:
            raise ValueError("Exact solution is None.")
        
        pointwise_error: torch.Tensor \
            = self._pinn_network(xt_points) - func_u(xt_points)

        self._max_norm_error = torch.linalg.vector_norm(pointwise_error, 
                                                        ord=torch.inf).item()
        
    def compute_l2_norm_error(self, 
                              quadrature: Quadrature) -> None:
        
        r""" Computes the C([0,T]; L^2(\Omega)) norm error. 
        
        Args:
            quadrature (Quadrature): Object that provides information on the employed 
                quadrature rule. """
        
        results = torch.zeros_like(self._t_points.flatten())
        t_points = self._t_points.flatten()

        for j in range(results.shape[0]):
            
            int_u_u_nn_squared \
                = quadrature.approximate_integral(lambda x: (self._u_pinn_difference_t(t_points[j].item(), x))**2,
                                                             self._interval)
            results[j] = np.sqrt(int_u_u_nn_squared)

        self._l2_norm_error = torch.linalg.vector_norm(results, ord = torch.inf).item()


    def compute_h1_norm_error(self, 
                              quadrature: Quadrature) -> None:
        
        r""" Computes the C([0,T]; H^1(\Omega)) norm error. 
        
        Args:
            quadrature (Quadrature): Object that provides information on the employed 
                quadrature rule. """
        
        results = torch.zeros_like(self._t_points.flatten())
        t_points = self._t_points.flatten()

        for j in range(results.shape[0]):
            
            int_u_u_nn_squared \
                = quadrature.approximate_integral(lambda x: (self._u_pinn_difference_t(t_points[j].item(), x))**2,
                                                  self._interval)
            int_u_u_nn_x_squared \
                = quadrature.approximate_integral(lambda x: (self._dx_u_pinn_difference_t(t_points[j].item(), x))**2,
                                                  self._interval)                                                
            results[j] = np.sqrt(int_u_u_nn_squared + int_u_u_nn_x_squared)

        self._h1_norm_error = torch.linalg.vector_norm(results, ord = torch.inf).item()

    @property
    def error_norms(self) -> tuple[float | None,
                                   float | None, 
                                   float | None]:
        
        r""" Returns the error measured in the maximum, the L^2- and
        H^1-norm. 
        
        Returns:
            tuple[float | None, float | None, float | None]: Tuple consisting of:
                - **float | None**: Maximum norm error
                - **float | None**: L^2-norm error
                - **float | None**: H^1-norm error"""
        
        return self._max_norm_error, self._l2_norm_error, self._h1_norm_error
    
    def create_graphical_illustration(self) -> tuple[Figure, Figure, Figure]:
        
        r""" Returns a figure that presents a graphical illustration of the exact solution, the 
        predicition of the model. Furthermore a plot which shows the error surface 
        between the exact solution and the predicition is created. 
        
        Returns:
            tuple[Figure, Figure, Figure]: Tuple consiting of:
                - **Figure**: Plot of the exact solution
                - **Figure**: Plot of the pinn approximation
                - **Figure**: Plot of the absolute error
                
        """

        self._pinn_network.eval()

        with torch.no_grad():
            fig_sol = plt.figure()
            ax_sol = fig_sol.add_subplot(projection="3d")

            fig_approx = plt.figure()
            ax_approx: Axes3D = fig_approx.add_subplot(projection="3d")

            fig_error = plt.figure()
            ax_error: Axes3D = fig_error.add_subplot(projection="3d")

            x_points = self._x_points.repeat(self._N+1)
            t_points = self._t_points.repeat_interleave(self._N+1)

            xt_points = torch.cat((x_points.unsqueeze(-1), t_points.unsqueeze(-1)), dim=1)

            func_u_eval = self._pde_problem.func_u(xt_points)
            if func_u_eval is None:
                raise ValueError("Exact solution is not defined")
            pinn_eval: torch.Tensor = self._pinn_network(xt_points)
            error = torch.abs(func_u_eval - pinn_eval)

            x_points_cpu = x_points.cpu().numpy().reshape((self._N+1, self._N+1))
            t_points_cpu = t_points.cpu().numpy().reshape((self._N+1, self._N+1))

            ax_sol.plot_surface(x_points_cpu, t_points_cpu, func_u_eval.cpu().numpy().reshape((self._N+1, self._N+1)),       
                                cmap='plasma', linewidth=0, antialiased=False)
            ax_sol.set_xlabel(r"$x$")
            ax_sol.set_ylabel(r"$t$")
            ax_sol.set_zlabel(r"$u(x,t)$")

            ax_approx.plot_surface(x_points_cpu, t_points_cpu, pinn_eval.cpu().numpy().reshape((self._N+1, self._N+1)),       
                                   cmap='plasma', linewidth=0, antialiased=False)
            ax_approx.set_xlabel(r"$x$")
            ax_approx.set_ylabel(r"$t$")
            ax_approx.set_zlabel(r"$u_{nn}(x,t)$")

            ax_error.plot_surface(x_points_cpu, t_points_cpu, error.cpu().numpy().reshape((self._N+1, self._N+1)),       
                                  cmap='plasma', linewidth=0, antialiased=False)
            ax_error.set_xlabel(r"$x$")
            ax_error.set_ylabel(r"$t$")
            ax_error.set_zlabel(r"$|(u-u_{nn})(x,t)|$")

            return fig_sol, fig_approx, fig_error
            



