from dataclasses import dataclass

@dataclass
class GradNormConfig():

    r""" Class to save parameters for Grad Norm. 
    
    Args:
        use_grad_norm (bool): Shall grad norm be used?
            Yes -> ``True``, No -> ``False``. Defaults to ``False``.
        grad_norm_alpha (float): Parameter of updating the new weights.
            Defaults to ``0.9``.
        lambda_pde_init (float): Initial weight for the pde loss. 
            Defaults to ``1.0``.
        lambda_init_init (float): Initial weight for the loss from. 
            the initial condition. Defaults to ``1.0``.
        lambda_boundary_init (float): Initial weight for the boundary 
            loss. Defaults to ``1.0``.
        grad_norm_step_size (int): Number of epochs after which the 
            weights are updated. Defaults to ``1000``
    """

    use_grad_norm: bool = False
    grad_norm_alpha: float = 0.9
    lambda_pde_init: float = 1.0
    lambda_init_init: float = 1.0
    lambda_boundary_init: float = 1.0
    grad_norm_step_size: int = 1000
    normalize_weights: bool = False

    @property
    def get_initial_lambdas(self) -> tuple[float, 
                                           float, 
                                           float]:
    
        r""" Returns the intial values of the loss weights
        lambda_{pde}, lambda_{init} and lambda_{boundary}.
        
        Returns:
            tuple[float, float, float]: Tuple consiting of:
                - **float**: Initial value of lambda_{pde}
                - **float**: Initial value of lambda_{init}
                - **float**: Initial value of lambda_{boundary}
        
        """

        return self.lambda_pde_init, self.lambda_init_init, self.lambda_boundary_init