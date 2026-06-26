import torch
import argparse
from pathlib import Path

def _load_adapter_state_dict(adapter_dir: Path) -> dict:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path))
    if bin_path.is_file():
        return torch.load(str(bin_path), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

def merge_task_arithmetic(deltas: list[torch.Tensor]) -> torch.Tensor:
    """Simple average of updates."""
    return sum(deltas) / len(deltas)

def merge_ties(deltas: list[torch.Tensor], density: float = 0.6) -> torch.Tensor:
    """TIES merging algorithm: TrIm, Elect Sign, Disjoint Merge."""
    trimmed_deltas = []
    for d in deltas:
        k = int(density * d.numel())
        if k == 0:
            trimmed_deltas.append(torch.zeros_like(d))
            continue
        vals, _ = torch.topk(d.abs().flatten(), k)
        thresh = vals[-1]
        mask = (d.abs() >= thresh)
        trimmed_deltas.append(d * mask)
        
    sum_trimmed = sum(trimmed_deltas)
    elected_sign = torch.sign(sum_trimmed)
    
    merged_delta = torch.zeros_like(deltas[0])
    count_matching = torch.zeros_like(deltas[0])
    
    for td in trimmed_deltas:
        match_mask = (torch.sign(td) == elected_sign) & (elected_sign != 0) & (td != 0)
        merged_delta[match_mask] += td[match_mask]
        count_matching[match_mask] += 1
        
    valid = count_matching > 0
    merged_delta[valid] = merged_delta[valid] / count_matching[valid]
    
    return merged_delta

def main():
    parser = argparse.ArgumentParser(description="Merge adapters to dense \Delta W using TA or TIES.")
    parser.add_argument("--adapters_dir", type=Path, required=True, help="Directory containing domain adapters.")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to save dense state_dict (.pt).")
    parser.add_argument("--method", type=str, choices=["ta", "ties"], required=True, help="Merging method.")
    parser.add_argument("--ties_density", type=float, default=0.6, help="Density for TIES merging.")
    args = parser.parse_args()

    subdirs = sorted([p for p in args.adapters_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not subdirs:
        raise RuntimeError(f"No subdirs in {args.adapters_dir}")
    
    print(f"Loading {len(subdirs)} adapters from {args.adapters_dir}...")
    state_dicts = [_load_adapter_state_dict(d) for d in subdirs]
    
    b_keys = sorted(k for k in state_dicts[0] if k.endswith("lora_B.weight"))
    
    merged_state_dict = {}
    print(f"Merging {len(b_keys)} layers using {args.method.upper()}...")
    
    for bk in b_keys:
        ak = bk.replace("lora_B.weight", "lora_A.weight")
        
        # We need original base_key to patch the base model later
        base_key = bk.replace(".lora_B", "")
        # Strip peft prefixes
        for prefix in ("base_model.model.", "base_model."):
            if base_key.startswith(prefix):
                base_key = base_key[len(prefix):]
                break
                
        A_list = []
        B_list = []
        for sd in state_dicts:
            B_list.append(sd[bk].float())
            A_list.append(sd[ak].float())
            
        if args.method == "ta":
            merged_B = merge_task_arithmetic(B_list)
            merged_A = merge_task_arithmetic(A_list)
        else:
            merged_B = merge_ties(B_list, density=args.ties_density)
            merged_A = merge_ties(A_list, density=args.ties_density)
            
        delta_W = merged_B @ merged_A
            
        merged_state_dict[base_key] = delta_W.half() # Store as float16 to save space
        
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_state_dict, args.output_file)
    print(f"Saved merged dense updates to {args.output_file}")

if __name__ == "__main__":
    main()
