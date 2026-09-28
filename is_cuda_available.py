r""" Little script to verify that CUDA is available. Just run it via

python3 is_cuda_available.py

"""
import torch

print(torch.cuda.is_available())