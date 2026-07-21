import torch
import torch.nn as nn
from .data_classes import RandomFourierFeatureEmbeddingsConfig

class RFFEmbedding(nn.Module):

    def __init__(self,
                 rffe_config: RandomFourierFeatureEmbeddingsConfig) -> None:

        r"""Initialization of the random Fourier feature embedding layer.

        The input coordinates are mapped to a higher-dimensional feature space
        using random Fourier features. Separate embeddings are constructed for
        the spatial and temporal coordinates. If the number of embeddings for
        one coordinate is set to zero, the corresponding coordinate is passed
        through unchanged.

        Args:
            rffe_config (RandomFourierFeatureEmbeddingsConfig): Configuration data 
                for random Fourier feature embedding.
        """

        super(RFFEmbedding, self).__init__()

        # Extract configuration data
        spatial_embeddings = rffe_config.spatial_embeddings
        temporal_embeddings = rffe_config.temporal_embeddings
        rffe_std = rffe_config.rffe_std

        # Varify that embedding numbers are non-negative
        if spatial_embeddings < 0 or temporal_embeddings < 0:
            raise ValueError(f"""Number of embeddings must be non-negative but got 
                             spatial_embeddings == {spatial_embeddings} and 
                             temporal_embeddings == {temporal_embeddings}. """)


        # Create spatial encoding matrix
        self.B_spatial: torch.Tensor | None
        if spatial_embeddings > 0:
            self.register_buffer("B_spatial",
                                 torch.randn((spatial_embeddings, 1))*rffe_std)  
        else:
            self.register_buffer("B_spatial", None)

        # Create temporal encodin matrix
        self.B_temporal: torch.Tensor | None
        if temporal_embeddings > 0:
            self.register_buffer("B_temporal",
                                 torch.randn((temporal_embeddings, 1))*rffe_std)
        else:
            self.register_buffer("B_temporal", None)

    @staticmethod
    def encode(z: torch.Tensor,
               B: torch.Tensor | None) -> torch.Tensor:
        
        r"""Encode a one-dimensional coordinate using random Fourier features.

        If a random projection matrix is provided, the input coordinate is
        projected onto the random basis and mapped to its cosine and sine
        representations. Otherwise, the input is returned unchanged.

        Args:
            z (torch.Tensor): Input coordinate tensor of shape ``(N, 1)``.
            B (torch.Tensor | None): Random projection matrix. If ``None``,
                no encoding is performed.

        Returns:
            torch.Tensor: Encoded feature tensor. Its shape is ``(N, 2m)``,
            where ``m`` denotes the number of Fourier embeddings. If ``B`` is
            ``None``, the returned tensor has shape ``(N, 1)``.
        """

        if isinstance(B, torch.Tensor):
            embeddings = torch.matmul(z, B.T)
            embeddings = torch.cat((torch.cos(embeddings),
                                    torch.sin(embeddings)), dim=1)
        else:
            embeddings = z

        return embeddings

    @property
    def output_dimension(self) -> int:

        r"""Return the dimension of the embedded feature vector.

        The returned value corresponds to the number of input features
        produced by the embedding layer and can be used to determine the
        input dimension of the subsequent neural network.

        Returns:
            int: Dimension of the embedded feature vector.
        """

        dim = 0

        if self.B_spatial is None:
            dim += 1
        else:
            dim += 2 * self.B_spatial.shape[0]

        if self.B_temporal is None:
            dim += 1
        else:
            dim += 2 * self.B_temporal.shape[0]

        return dim

    def forward(self, xt_points: torch.Tensor) -> torch.Tensor:

        r"""Apply the random Fourier feature embedding.

        The spatial and temporal coordinates are encoded independently and
        subsequently concatenated to form the embedded feature vector.

        Args:
            xt_points (torch.Tensor): Tensor of shape ``(N, 2)``
                containing the spatial and temporal coordinates.

        Returns:
            torch.Tensor: Embedded feature tensor of shape
            ``(N, output_dimension)``.
        """

        # Encode spatial variables
        x_embeddings = self.encode(xt_points[:,0:1], self.B_spatial)

        # Encode temporal variables
        t_embeddings = self.encode(xt_points[:,1:2], self.B_temporal)
            
        return torch.cat((x_embeddings, t_embeddings), dim=1)