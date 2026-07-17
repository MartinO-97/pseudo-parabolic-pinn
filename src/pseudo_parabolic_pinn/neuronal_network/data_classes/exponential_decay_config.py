from dataclasses import dataclass

@dataclass
class ExponentialDecayConfig():

    r""" Class to save parameters for exponential Decay.

    Args:
        use_ed (bool): If ``True`` exponential decay is used. Defaults to ``False``
        gamma (float): Multiplicative factor of learning rate decay. Defaults to ``0.9``.
        exponential_decay_step_size (int): Period of learning rate decay. Defaults to ``2000``.
    """

    use_ed: bool = False
    gamma: float = 0.9
    exponential_decay_step_size: int = 2000