import torch
from typing import List, Tuple
from .utils import compute_gamma

def merge_wbp(
    B_list: List[torch.Tensor], 
    A_list: List[torch.Tensor],
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements the Woodbury B-Space Preconditioning (WBP) algorithm.
    Args:
        B_list: list of T tensors of shape (d_out, r)
        A_list: list of T tensors of shape (r, d_in)
        beta: exploratory scale factor on lambda
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
        
    device = B_list[0].device
    dtype = B_list[0].dtype
    
    # Stacking
    B_all = torch.cat(B_list, dim=1)  # shape (d_out, Tr)
    Tr = B_all.shape[1]
    
    # Step 1: Compute Gram Matrix
    G = B_all.T @ B_all
    
    # Step 2: Compute Optimal lambda
    lambda_val = (T - 1) / torch.trace(G) * beta
    
    # Step 3: Compute Kernel Inverse
    # K = (1/lambda I_Tr + G)^-1
    inv_lambda_I = torch.eye(Tr, device=device, dtype=dtype) / lambda_val
    K = torch.linalg.inv(inv_lambda_I + G)
    
    # Step 4: Apply to Task Matrices right-to-left
    # B_tilde_t = B_t - B_all @ (K @ (B_all.T @ B_t))
    B_tilde_list = []
    for B_t in B_list:
        Bt_all_Bt = B_all.T @ B_t
        K_Bt = K @ Bt_all_Bt
        B_tilde = B_t - (B_all @ K_Bt)
        B_tilde_list.append(B_tilde)
        
    # Step 5: Merge and Rescale
    gamma = compute_gamma(B_list, A_list, B_tilde_list)
    
    # Format the return as B_merged, A_merged
    B_merged = torch.cat([gamma/T * B_tilde for B_tilde in B_tilde_list], dim=1)
    A_merged = torch.cat(A_list, dim=0)
    
    return B_merged, A_merged
