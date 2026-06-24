import torch
from typing import List, Tuple
from .utils import compute_gamma

def merge_pico(
    B_list: List[torch.Tensor], 
    A_list: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements the Pico merge algorithm.
    Args:
        B_list: list of T tensors of shape (d_out, r)
        A_list: list of T tensors of shape (r, d_in)
    Returns:
        B_merged: tensor of shape (d_out, T*r)
        A_merged: tensor of shape (T*r, d_in)
    """
    T = len(B_list)
    if T == 0:
        raise ValueError("Cannot merge 0 tasks.")
        
    if T == 1:
        # Edge case guard from thesis
        return B_list[0], A_list[0]
        
    # Step 1: Stacking and SVD
    B_all = torch.cat(B_list, dim=1)  # shape (d_out, Tr)
    U, S, Vh = torch.linalg.svd(B_all, full_matrices=False)
    
    # Step 2: Energy Sharing Score
    S_sq = S ** 2
    s = S_sq / S_sq.sum()
    
    # Step 3: Calibration Coefficients
    alpha = 1.0 / (1.0 + (T - 1) * s)
    
    # Step 4: The Calibration Operator
    # Apply this directly: S_pico @ B_t = B_t + U diag(alpha - 1) U^T B_t
    alpha_minus_1 = alpha - 1.0
    
    B_tilde_list = []
    for B_t in B_list:
        # Avoid explicit S_pico construction for speed
        Ut_Bt = U.T @ B_t
        scaled = alpha_minus_1.unsqueeze(1) * Ut_Bt
        B_tilde = B_t + (U @ scaled)
        B_tilde_list.append(B_tilde)
        
    # Step 5: Magnitude Rescaling
    gamma = compute_gamma(B_list, A_list, B_tilde_list)
    
    # Format the return as B_merged, A_merged
    B_merged = torch.cat([gamma/T * B_tilde for B_tilde in B_tilde_list], dim=1)
    A_merged = torch.cat(A_list, dim=0)
    
    return B_merged, A_merged
