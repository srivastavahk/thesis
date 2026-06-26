```
Algorithm: WBP (Woodbury B-Space Preconditioning)
Input:  {(A_t, B_t)}_{t=1}^T
Output: ΔW_WBP

1.  B_all ← [B_1 | B_2 | ... | B_T]                     # d_out x Tr
2.  G ← B_allᵗ · B_all                                  # Tr x Tr Gram matrix — no SVD
3.  traceG ← trace(G)                                   # = ‖B_all‖_F²

4.  if T == 1:
5.      B̃_t ← B_t for all t                             # nothing to calibrate; skip 6-8
6.  else:
7.      λ ← (T - 1) / traceG
8.      K ← ( (1/λ)·I_{Tr} + G )⁻¹                       # Tr x Tr inverse — cheap, microseconds
9.      for t = 1..T:
10.         B̃_t ← B_t - B_all · ( K · (B_allᵗ · B_t) )   # never materialize the d_out x d_out operator
11.     ΔW̃_t ← B̃_t · A_t

12. ΔW_calib ← M(ΔW̃_1, ..., ΔW̃_T)
13. γ ← ( (1/T)·Σ_t ‖B_t·A_t‖_F ) / ‖ΔW_calib‖_F
14. ΔW_WBP ← γ · ΔW_calib

return ΔW_WBP
```