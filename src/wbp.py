import torch
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple
from src.utils import compute_gamma

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
        
    gamma = compute_gamma(B_list, A_list, B_tilde_list)
    
    # Format the return as B_merged, A_merged
    B_merged = torch.cat([gamma/T * B_tilde for B_tilde in B_tilde_list], dim=1)
    A_merged = torch.cat(A_list, dim=0)
    
    return B_merged, A_merged


def _load_adapter_state_dict(adapter_dir: Path) -> dict:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path))
    if bin_path.is_file():
        return torch.load(str(bin_path), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate PEFT adapters using WBP algorithm.")
    parser.add_argument("--adapters_dir", type=Path, required=True, help="Directory containing domain subdirectories.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save calibrated adapters.")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta scale factor on lambda.")
    args = parser.parse_args()

    adapters_dir = args.adapters_dir
    out_dir = args.output_dir

    subdirs = sorted([p for p in adapters_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not subdirs:
        raise RuntimeError(f"No subdirs in {adapters_dir}")
    
    print(f"Loading {len(subdirs)} adapters from {adapters_dir}...")
    state_dicts = [_load_adapter_state_dict(d) for d in subdirs]
    
    # Find B keys
    b_keys = sorted(k for k in state_dicts[0] if k.endswith("lora_B.weight"))
    
    # Process layer by layer
    T = len(subdirs)
    print(f"Calibrating {len(b_keys)} layers using WBP (beta={args.beta})...")
    for bk in b_keys:
        ak = bk.replace("lora_B.weight", "lora_A.weight")
        
        B_list = [sd[bk].float() for sd in state_dicts]
        A_list = [sd[ak].float() for sd in state_dicts]
        
        B_merged, A_merged = merge_wbp(B_list, A_list, beta=args.beta)
        r = B_list[0].shape[1]
        
        for i in range(T):
            calibrated_B = T * B_merged[:, i*r:(i+1)*r]
            
            # Cast back to original dtype
            orig_dtype = state_dicts[i][bk].dtype
            state_dicts[i][bk] = calibrated_B.to(orig_dtype)

    # Save to disk
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(subdirs):
        domain_name = d.name
        domain_out_dir = out_dir / domain_name
        domain_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save state dict
        from safetensors.torch import save_file
        save_file(state_dicts[i], str(domain_out_dir / "adapter_model.safetensors"))
        
        # Copy config
        config_path = d / "adapter_config.json"
        if config_path.exists():
            shutil.copy(config_path, domain_out_dir / "adapter_config.json")
            
        print(f"Saved calibrated adapter to {domain_out_dir}")

if __name__ == "__main__":
    main()
