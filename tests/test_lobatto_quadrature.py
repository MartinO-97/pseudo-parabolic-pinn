import pytest
import torch
import numpy as np

from pseudo_parabolic_pinn.numerical_utilities import LobattoQuadrature

def test_lobatto_quadrature_order():

    r""" Test if generated n nodes and weights integrate polynomials
    up to order 2n-3 exactly. """

    def monom(k: int, x: torch.Tensor) -> torch.Tensor:

        r""" Function that evaluates the ``k``-th monom at ``x``.
        
        Args:
            k (int): Grade of the monom
            x (torch.Tensor): Values where the ``k``-th monom shall be evaluated
                at.

        Returns:
            torch.Tensor: The ``k``-th monom evaluated at ``x``.        
        """
        
        return x**k



    def get_monom_integral(k:int,
                           interval: tuple[float, float]) -> float:

        r""" Function that returns exact value of the integral of ``x^k``
        over the given integral.
        
        Args:
            k (int): Grad of the monom.
            interval (tuple[float, float]): Interval [a,b] over which shall be integrated.

        Returns:
            float: Value of \int_a^b x^k dx.
        """

        a = interval[0]
        b = interval[1]

        if k % 2 != 0:
            return 0.0
        else:
            return 1/(k+1)*(b**(k+1)-a**(k+1))
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tests for Gauss-Lobatto with 3 nodes over [-1,1] 
    n3 = 3
    interval3 = -1.0, 1.0

    lobatto3 = LobattoQuadrature(n3, device)

    # Tests
    lobatto3_eval_0 = lobatto3.approximate_integral(lambda y: monom(0, y), interval3) 
    assert np.isclose(lobatto3_eval_0, get_monom_integral(0, interval3))

    lobatto3_eval_1 = lobatto3.approximate_integral(lambda y: monom(1, y), interval3) 
    assert np.isclose(lobatto3_eval_1, get_monom_integral(1, interval3))

    lobatto3_eval_2 = lobatto3.approximate_integral(lambda y: monom(2, y), interval3) 
    assert np.isclose(lobatto3_eval_2, get_monom_integral(2, interval3))

    lobatto3_eval_3 = lobatto3.approximate_integral(lambda y: monom(3, y), interval3) 
    assert np.isclose(lobatto3_eval_3, get_monom_integral(3, interval3))


