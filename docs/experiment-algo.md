# Experiment Algorithms

## Experiment 1 (E1): Operator-level Equivalence

The objective of E1 is to verify that the efficient Woodbury-based preconditioner (WBP) produces numerically identical calibrated $B$ matrices compared to the exact SVD-based Pico method on real, trained LoRA adapters.

### Main E1 Pipeline

For a given set of $T$ trained adapters (e.g., $T=4$ for math, coding, finance, medical):

```text
Algorithm: E1 Equivalence Check
Input: T adapters with weights {W_1, W_2, ..., W_T}
Output: Layer-wise error metrics

1. Initialize results list `per_layer_results = []`
2. Group all adapter weight dictionaries by layer name (e.g., `q_proj`, `v_proj`).
3. For each layer L:
4.     Extract B matrices: B_list = [B_1, B_2, ..., B_T]
5.     Extract A matrices: A_list = [A_1, A_2, ..., A_T]
6.     
7.     # Concatenate matrices
8.     B_all = CONCAT(B_list, axis="columns")  # Shape: (d_out, T * r)
9.     
10.    # Run both calibration algorithms
11.    B_pico = PicoCalibrate(B_all, T)
12.    B_wbp  = WBPCalibrate(B_all, T)
13.    
14.    # Compute Errors
15.    abs_diff = ABS(B_pico - B_wbp)
16.    max_abs_error = MAX(abs_diff)
17.    rel_error = FROBENIUS_NORM(abs_diff) / FROBENIUS_NORM(B_pico)
18.    
19.    Append {layer: L, max_abs: max_abs_error, rel_error: rel_error} to per_layer_results
20.
21. Return per_layer_results
```

### Pico Calibration (SVD Path)

The exact but computationally expensive approach using Singular Value Decomposition (SVD):

```text
Algorithm: PicoCalibrate
Input: B_all (d_out × Tr matrix), T (number of tasks)
Output: B_calibrated

1. Compute Thin SVD: 
       U, S, V^T = SVD(B_all)
2. Compute eigenvalues of Gram matrix: 
       S_sq = S^2
3. Normalize eigenvalues (trace = 1): 
       s = S_sq / SUM(S_sq)
4. Compute shrink factors: 
       alpha = 1.0 / (1.0 + (T - 1) * s)
5. Apply shrinking to singular components:
       B_calibrated = B_all + U @ (diag(alpha - 1.0) @ (U^T @ B_all))
6. Return B_calibrated
```

### WBP Calibration (Woodbury Path)

The faster, algebraically equivalent approach avoiding the large SVD using the Woodbury matrix identity:

```text
Algorithm: WBPCalibrate
Input: B_all (d_out × Tr matrix), T (number of tasks)
Output: B_calibrated

1. Compute small Gram matrix:
       G = B_all^T @ B_all               # Shape: (Tr × Tr)
2. Compute trace penalty factor:
       lambda = (T - 1) / TRACE(G)
3. Compute inversion term using Woodbury identity:
       K = INVERSE( (I / lambda) + G )   # I is Identity matrix of size Tr
4. Apply calibration projection:
       B_calibrated = B_all - B_all @ (K @ G)
5. Return B_calibrated
```
