from dataclasses import dataclass

@dataclass
class RandomWeightFactorizationConfig():

    r""" Class to save data for random weight factorization.
    
    Args:
        use_rwf (bool): Shall random weight factorization be used?
            Yes -> ``True``, No -> ``False``. Defaults to ``False``.
        rwf_mean (float): Mean of the normal distribution used to
                initialize the RWF scaling parameters. Defaults to ``1.0``.
        rwf_std (float): Standard deviation of the normal
            distribution used to initialize the RWF scaling parameters.
            Defaults to ``0.1``
        """
    
    use_rwf: bool = False
    rwf_mean: float = 1.0
    rwf_std: float = 0.1