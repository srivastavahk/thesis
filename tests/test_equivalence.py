import torch
from src.pico import merge_pico
from src.wbp import merge_wbp

def test_equivalence():
    torch.manual_seed(42)
    
    T = 4
    d_out = 1024
    d_in = 512
    r = 16
    
    B_list = [torch.randn(d_out, r, dtype=torch.float64) for _ in range(T)]
    A_list = [torch.randn(r, d_in, dtype=torch.float64) for _ in range(T)]
    
    B_pico, A_pico = merge_pico(B_list, A_list)
    B_wbp, A_wbp = merge_wbp(B_list, A_list, beta=1.0)
    
    # B_pico and B_wbp should be identical to machine precision
    # A_pico and A_wbp are just concatenated A's, they are obviously identical
    diff = torch.abs(B_pico - B_wbp).max()
    
    # Form the dense updates to be absolutely sure
    W_pico = B_pico @ A_pico
    W_wbp = B_wbp @ A_wbp
    diff_W = torch.abs(W_pico - W_wbp).max()
    
    print(f"Max diff in B_merged: {diff.item():.2e}")
    print(f"Max diff in W_merged: {diff_W.item():.2e}")
    
    assert diff < 1e-12, "Pico and WBP did not match on B_merged!"
    assert diff_W < 1e-12, "Pico and WBP did not match on W_merged!"
    print("Test passed! Pico and WBP are exactly equivalent.")

if __name__ == "__main__":
    test_equivalence()
