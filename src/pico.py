import torch
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple
from src.utils import compute_gamma

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
    parser = argparse.ArgumentParser(description="Calibrate PEFT adapters using Pico algorithm.")
    parser.add_argument("--adapters_dir", type=Path, required=True, help="Directory containing domain subdirectories.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save calibrated adapters.")
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
    print(f"Calibrating {len(b_keys)} layers using Pico...")
    for bk in b_keys:
        ak = bk.replace("lora_B.weight", "lora_A.weight")
        
        B_list = [sd[bk].float() for sd in state_dicts]
        A_list = [sd[ak].float() for sd in state_dicts]
        
        # We need the individual calibrated matrices. 
        # Since merge_pico returns concatenated, we'll extract them.
        B_merged, A_merged = merge_pico(B_list, A_list)
        r = B_list[0].shape[1]
        
        for i in range(T):
            # T * (gamma/T * \tilde{B}) = gamma * \tilde{B}
            # Note: the original B_t was used in merge_pico. We replace the state_dict's B matrix.
            # But wait, Task Arithmetic is \sum (B @ A) / T.
            # If we save the adapter, its weights will be used downstream by TA.
            # TA computes: (\sum \tilde{B}_t A_t) / T. 
            # B_merged contains gamma/T * \tilde{B}_t.
            # So if we want the standard pipeline to just sum them, we should just save gamma * \tilde{B}_t
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
