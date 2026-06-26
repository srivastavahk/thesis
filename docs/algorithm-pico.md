```
Algorithm: Pico (Pre-merge Interference Calibration)
Input:  {(A_t, B_t)}_{t=1}^T        # B_t ∈ R^(d_out x r), A_t ∈ R^(r x d_in)
Output: ΔW_Pico

1.  B_all ← [B_1 | B_2 | ... | B_T]                     # d_out x Tr
2.  U, σ, V ← SVD(B_all)                                # U: d_out x m, σ: length m
3.  m ← min(d_out, T*r)
4.  total_energy ← Σ_j σ_j²            for j = 1..m
5.  for j = 1..m:
6.      s_j ← σ_j² / total_energy
7.      α_j ← 1 / (1 + (T-1)·s_j)
8.  S ← I_{d_out} + U·diag(α - 1)·Uᵗ                    # calibration operator, d_out x d_out

9.  for t = 1..T:
10.     B̃_t ← S · B_t
11.     ΔW̃_t ← B̃_t · A_t

12. ΔW_calib ← M(ΔW̃_1, ..., ΔW̃_T)        # M = mean for Task Arithmetic, or TIES/TSV-M
13. γ ← ( (1/T)·Σ_t ‖B_t·A_t‖_F ) / ‖ΔW_calib‖_F
14. ΔW_Pico ← γ · ΔW_calib

return ΔW_Pico
```