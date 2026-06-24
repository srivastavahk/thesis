import torch
from typing import List

def frobenius_norm_low_rank(B: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Computes ||B A||_F efficiently without instantiating the full matrix.
    ||B A||_F = sqrt(Tr(B^T B A A^T))
    """
    BtB = B.T @ B
    AAt = A @ A.T
    return torch.sqrt(torch.clamp(torch.trace(BtB @ AAt), min=0))

def compute_gamma(
    B_list: List[torch.Tensor], 
    A_list: List[torch.Tensor], 
    B_tilde_list: List[torch.Tensor]
) -> torch.Tensor:
    """
    Computes the magnitude restoration scalar gamma.
    gamma = (1/T * sum_t ||B_t A_t||_F) / ||1/T sum_t B_tilde_t A_t||_F
    """
    T = len(B_list)
    if T == 0:
        return torch.tensor(1.0)
        
    # Numerator: average norm of individual task updates
    sum_norms = 0.0
    for B, A in zip(B_list, A_list):
        sum_norms += frobenius_norm_low_rank(B, A)
    avg_original_norm = sum_norms / T
    
    # Denominator: norm of the calibrated merged update
    B_calib_concat = torch.cat(B_tilde_list, dim=1)
    A_calib_concat = torch.cat([A / T for A in A_list], dim=0)
    calib_norm = frobenius_norm_low_rank(B_calib_concat, A_calib_concat)
    
    if calib_norm < 1e-8:
        return torch.tensor(1.0, dtype=B_list[0].dtype, device=B_list[0].device)
        
    return avg_original_norm / calib_norm
