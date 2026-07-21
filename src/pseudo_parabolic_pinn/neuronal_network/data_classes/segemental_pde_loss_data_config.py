from dataclasses import dataclass
import torch

@dataclass
class SegmentalPDELossConfig():

    r""" Class to save data for the segemental pde loss. If number of segements 
    equals ``1`` the classical pde loss is used.
    
    Args:
        number_of_semgents (int): Number of segments. Defaults to ``1``.
        w_init (torch.Tensor): Initial weights of segemental losses. 
            Defaults to ``torch.ones((number_of_segments,1), dypte=torch.float32)``.
        update_segement_weights (bool): If ``True`` the weights of the 
            segemental losses are updated in each epoch. Defaults 
            to ``False``.
        spl_eps (float): Parameter to update segmental loss weights. Defaults
            to ``1.0``.
    """

    number_of_segements: int = 1
    w_init = torch.ones((number_of_segements, 1), dtype=torch.float32)
    update_segments_weights: bool = False
    spl_eps: float = 1.0