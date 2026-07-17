from dataclasses import dataclass

@dataclass
class RandomFourierFeatureEmbeddingsConfig():

    r""" Class to save parameters for random Fourier feature
    embeddings. If embedding parameters are set to ``0``, no
    encoding is used.
    
    Args:
        spatial_embeddings (int): Number of embeddings for 
            the random Fourier feature embeddings for the spatial
            variable. Defaults to ``0``.
        temporal_embeddings (int): Number of embeddings for 
            the random Fourier feature embeddings for the temporal
            variable. Defaults to ``0``.
        rffe_std (float): Standard deviation of the normal 
            distribution employed to intialize the random Fourier
            feature embedding class. Defaults to ``1.0``.
    """

    spatial_embeddings: int = 0
    temporal_embeddings: int = 0
    rffe_std: float = 1.0