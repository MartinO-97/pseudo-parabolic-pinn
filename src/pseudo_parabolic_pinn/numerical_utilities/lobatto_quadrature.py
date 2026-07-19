import torch 
import numpy as np
import numpy.linalg as lg
from typing import Callable

class LobattoQuadrature():

    def __init__(self, 
                 number_nodes: int,
                 device: str) -> None:
        
        r""" Initialization of nodes and weights for a 
        Gauss-Lobatto quadrature rule with ``number_nodes``
        nodes and weights.
        
        Args:
            number_nodes (int): Number of nodes and weights of
                the quadrature rule.
            device (str): Device where tensors are stored
        """

        self._device = device
        self._number_nodes = number_nodes
        self.nodes: torch.Tensor
        self.weights: torch.Tensor
        self.compute_nodes_and_weights()

    @property
    def number_nodes(self) -> int:
        
        """ Function that returns the number of nodes and weights.
        
        Returns:
            int: The number of nodes and weights
        """

        return self._number_nodes
    
    @property
    def nodes_and_weights(self) -> tuple[torch.Tensor,
                                             torch.Tensor]:
        
        """ Function that returns the nodes and weights of 
        the Gauss-Lobatto quadrature rule.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple, consiting of
                - **torch.Tensor**: Nodes of the Gauss-Lobatto formula.
                - **torch.Tensor**: Weights of the Gauss-Lobatto formula.
        """

        return self.nodes, self.weights

    def compute_nodes_and_weights(self) -> None:

        r""" Compute nodes and weights for the Gauss-Lobatto quadrature
        rule with ``n=self._number_nodes`` nodes and weights. 

        The nodes -1=x_1 < ... < x_n=1 are the n roots of the (n-1)-th 
        integrated Legendre polynomial. The weights are given by 

        w_j = 2/(n*(n-1)) * 1/P'_{n-1}(x_j)**2,  j = 1,...,n,

        where P_{n-1} denotes the (n-1)-th Legendre polynomial.  
        """

        n = self._number_nodes

        # Extrema of the (n-1)-th Chebyshev polynomial of first kind T_{n-1}
        # as starting values for the Newton method
        x = np.cos(np.arange(0,n)*np.pi/(n-1))[::-1]

        # Initialize x_{i-1}
        x_old = 2*np.ones_like(x)

        stop = 0

        # Newton step
        while lg.vector_norm(x-x_old,ord=np.inf) > 10**(-13):
            
            stop += 1

            # x_{i-1}
            x_old = np.copy(x) 
            
            # Initialize P_0(x_old) and P_1(x_old) 
            P_vs = 0*x+1  
            P_s = np.copy(x)

            # Compute P_{n-1}(x_old)  
            for k in range(2,n):
                P_b = (2*k-1)/k * x * P_s - (k-1)/k * P_vs      # P_k
                P_vs = np.copy(P_s)                             # P_{k-2}
                P_s = np.copy(P_b)                              # P_{k-1}

            # Compute P_n
            P_b = (2*n-1)/n * x * P_s - (n-1)/n * P_vs

            # Compute N_{n-1}(x_old); then P_b corresponds to P_n, P_s to P_{n-1} and P_vs to P_{n-2}
            N = (P_b - P_vs)/(2*n-1)
            
            # Update x
            x = x_old - N/P_s

        # Set manually x[0] = -1 and x[-1] = 1 
        x[np.array([0,-1])] = np.array([-1,1])

        # weights
        w = 2/(n**2-n) * 1/P_s**2

        self.nodes = \
            torch.from_numpy(x).to(dtype=torch.float32, device=self._device).unsqueeze(-1)
        self.weights = \
            torch.from_numpy(w).to(dtype=torch.float32, device= self._device).unsqueeze(-1)

    @staticmethod
    def _interval_transformation(interval: tuple[float, float], 
                                 x: torch.Tensor) -> torch.Tensor:
        
        r""" Transformation of values in [-1,1] to a given interval.
        
        Args:
            interval (list[float]): Interval where values ``x`` shall be mapped
                to.
            x (torch.Tensor): Values that shall be mapped onto the given interval.

        Returns:
            torch.Tensor: Transformed values of ``x``.
        """

        alpha = interval[0]
        beta = interval[1]
        
        return alpha + (beta-alpha)/2*(x+1)

    def approximate_integral(self,
                             func_y: Callable[[torch.Tensor], torch.Tensor],
                             interval: tuple[float, float]) -> float:
        
        r""" Approximation of the integral 
        
        \int_{\alpha}^\beta y(x) dx
        
        by employing the Gauss-Lobatto quadrature rule. 

        Args:
            func_y (Callable[[torch.Tensor], torch.Tensor]): The integrand ``y``.
            interval (tuple[float, float]): The interval [\alpha, \beta] over which 
                shall be integrated

        Returns:
            float: Approximation of the \int_{\alpha}^\beta y(x) dx.
        """

        mapped_nodes = self._interval_transformation(interval, self.nodes)
        alpha = interval[0]
        beta = interval[1]
        integral = torch.matmul(self.weights.T, func_y(mapped_nodes)).item()

        return (beta-alpha)/2 * integral
    

    def approximate_integral_from_values(self,
                             func_y_eval: torch.Tensor,
                             interval: tuple[float, float]) -> float:
        
        r""" Approximation of the integral 
        
        \int_{\alpha}^\beta y(x) dx
        
        by employing the Gauss-Lobatto quadrature rule. The function y is 
        already evaluated at the nodes of the quadrature rule 

        Args:
            func_y_eval (torch.Tensor): The integrand ``y`` evaluated at the nodes
                of the Gauss-Lobatto quadrature rule.
            interval (tuple[float, float]): The interval [\alpha, \beta] over which 
                shall be integrated

        Returns:
            float: Approximation of the \int_{\alpha}^\beta y(x) dx.
        """

        alpha = interval[0]
        beta = interval[1]
        integral = torch.matmul(self.weights.T, func_y_eval).item()

        return (beta-alpha)/2 * integral
