import pytest
import torch
import numpy as np

from pseudo_parabolic_pinn.numerical_utilities import LobattoQuadrature

@pytest.mark.parametrize("n, interval, k", [
    # Tests for n=3 
    (3, (-1.0, 1.0), 0),
    (3, (-1.0, 1.0), 1),
    (3, (-1.0, 1.0), 2),
    (3, (-1.0, 1.0), 3),
    # Tests for n=4
    (4, (-2.0, 1.0), 0),
    (4, (-2.0, 1.0), 1),
    (4, (-2.0, 1.0), 2),
    (4, (-2.0, 1.0), 3),
    (4, (-2.0, 1.0), 4),
    (4, (-2.0, 1.0), 5),
])

def test_lobatto_quadrature_order(n: int, interval: tuple[float, float], k: int) -> None:

    r""" Test if generated n nodes and weights integrate polynomials
    up to order 2n-3 exactly. 
    
    Args: 
        n (int): Number of nodes and weights
        interval (tuple[float, float]): Interval over which shall be ingerated
        k (int): Grade of monom.
    """

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

        return 1/(k+1)*(b**(k+1)-a**(k+1))
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute Lobatto nodes and weights
    lobatto = LobattoQuadrature(n, device)

    # Tests
    approx_int = lobatto.approximate_integral(lambda y: monom(k, y), interval)
    exact_int = get_monom_integral(k, interval)

    assert np.isclose(approx_int, exact_int)

