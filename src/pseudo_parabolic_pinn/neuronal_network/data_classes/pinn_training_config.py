from dataclasses import dataclass, field
from .exponential_decay_config import ExponentialDecayConfig
from .grad_norm_config import GradNormConfig
from .random_fourier_feature_embeddings_config import RandomFourierFeatureEmbeddingsConfig
from .random_weight_factorization_config import RandomWeightFactorizationConfig
from .segemental_pde_loss_data_config import SegmentalPDELossConfig

@dataclass
class PinnTrainingConfig():

    r""" Class to save config data for training the pinn network.
    
    Args:
        epochs (int): Number of epochs
        batch_size (int): Batch size
        learn_rate (float): The learn rate for gradient descent.
            Defaults to ``0.001``.
        device (str): Device where tensors and the network shall be stored.
        use_lbfgs (bool): If ``True`` the LBFGS algorithm is used to train the 
            model in the lasst 5000 epochs. 
        ed_config (ExponentialDecayConfig): Config data for 
            exponential decay.
        gn_config (GradNormConfig): Config data for grad norm
        rffe_config (RandomFourierFeatureEmbeddingsConfig): Config
            data for random Fourier feature embedding.
        rwf_config (RandomWeightFactorizationConfig): Config data
            for random weight facotrization
        spl_config (SegmentalPDELossConfig): Config data for the
            segmental pde loss.

    """
    epochs: int = 10000
    batch_size: int = 4096
    learn_rate: float = 0.001
    device: str = "cpu"
    use_lbfgs: bool = False

    ed_config: ExponentialDecayConfig = field(default_factory=ExponentialDecayConfig)
    gn_config: GradNormConfig = field(default_factory=GradNormConfig)
    rffe_config: RandomFourierFeatureEmbeddingsConfig \
        = field(default_factory=RandomFourierFeatureEmbeddingsConfig)
    rwf_config: RandomWeightFactorizationConfig \
        = field(default_factory=RandomWeightFactorizationConfig)
    spl_config: SegmentalPDELossConfig = field(default_factory=SegmentalPDELossConfig)

