r""" Little script to verify that cuda is available. Just run it via

python3 ic_cuda_available().py

"""
import torch

print(torch.cuda.is_available())