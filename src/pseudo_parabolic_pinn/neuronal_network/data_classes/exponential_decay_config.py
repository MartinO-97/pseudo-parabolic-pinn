from dataclasses import dataclass

@dataclass
class ExponentialDecayConfig():

    r""" Class to save parameters for exponential Decay.
    
    Args:
        gamma (float): Multiplicative factor of learning rate decay. Defaults to ``0.9``.
        exponential_decay_step_size (int): Period of learning rate decay. Defaults to ``2000``.
    """
    gamma: float = 0.9
    exponential_decay_step_size: int = 2000