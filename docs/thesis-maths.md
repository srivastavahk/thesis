# Mathematical Foundations of B-Space LoRA Merging
**A Formal Specification of Pico and Woodbury B-Space Preconditioning (WBP)**

This document outlines the complete mathematical formulations, underlying assumptions, and algorithmic proofs for mitigating parameter interference during the merging of $T$ independently trained Low-Rank Adaptation (LoRA) modules.

---

## 1. Preliminaries & Notation

Let a pre-trained weight matrix be $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$. During parameter-efficient fine-tuning for a specific task $t \in \{1, \dots, T\}$, we freeze $W_0$ and train a low-rank update:

$$\Delta W_t = B_t A_t$$

Where:
* $B_t \in \mathbb{R}^{d_{out} \times r}$ is the output-projection matrix.
* $A_t \in \mathbb{R}^{r \times d_{in}}$ is the input-projection matrix.
* $r \ll \min(d_{out}, d_{in})$ is the LoRA rank.
* *(Note: Standard LoRA scaling factors $\alpha/r$ are absorbed into $B_t$ for notational convenience).*

### 1.1 The Linear Merging Problem
Standard multi-task merging (e.g., Task Arithmetic) averages the updates:
$$\Delta W_{avg} = \frac{1}{T} \sum_{t=1}^T B_t A_t$$

**The Crowding Phenomenon:** If a specific directional vector $u \in \mathbb{R}^{d_{out}}$ is shared across the output space of all $T$ tasks, its magnitude is preserved during averaging. Conversely, a purely task-specific direction present in only one task is diluted by a factor of $1/T$. As $T$ scales, the shared directions artificially dominate the merged weight matrix, destroying task-specific knowledge.

---

## 2. Pico Merge (Pre-merge Interference Calibration)

Pico solves this by explicitly downscaling the shared directions in $B$-space prior to merging.

### 2.1 Underlying Assumptions of Pico
1. **Geometric Asymmetry:** Merge interference is localized in the output space ($B$-space). The input space ($A$-space) naturally learns sparse, orthogonal, task-specific features and should not be calibrated.
2. **Spectral Concentration:** Across $T$ distinct tasks, the joined output space relies heavily on a small subset of shared directions, resulting in a matrix with a low "effective rank."

### 2.2 Mathematical Formulation

**Step 1: Stacking and SVD**
We concatenate all $T$ output matrices horizontally to form the global $B$-space matrix:
$$B_{all} = \begin{bmatrix} B_1 & B_2 & \dots & B_T \end{bmatrix} \in \mathbb{R}^{d_{out} \times Tr}$$

We extract the orthonormal basis directions and their magnitudes via Singular Value Decomposition (SVD):
$$B_{all} = U \Sigma V^\top$$
Where:
* $U \in \mathbb{R}^{d_{out} \times d_{out}}$ contains the left singular vectors (the shared output directions).
* $\Sigma$ is the diagonal matrix of singular values $\sigma_j$.

**Step 2: Energy Sharing Score**
To quantify how "crowded" a direction $j$ is, we calculate its fraction of the total variance:
$$s_j = \frac{\sigma_j^2}{\sum_{k=1}^{Tr} \sigma_k^2}$$
*(Note: $s_j \in [0, 1]$ and $\sum s_j = 1$)*.

**Step 3: Calibration Coefficients**
We assign a penalty scalar $\alpha_j$ to each direction. Highly shared directions ($s_j \to 1$) are penalized by roughly $1/T$. Unique directions ($s_j \to 0$) are preserved ($\alpha_j \to 1$).
$$\alpha_j = \frac{1}{1 + (T-1)s_j}$$

**Step 4: The Calibration Operator**
We construct a filtering matrix $S_{pico} \in \mathbb{R}^{d_{out} \times d_{out}}$:
$$S_{pico} = I_{d_{out}} + U \text{diag}(\alpha - 1) U^\top$$

Applying this to an individual task's output matrix yields the calibrated update:
$$\tilde{B}_t = S_{pico} B_t$$
$$\widetilde{\Delta W}_t = \tilde{B}_t A_t$$

**Step 5: Magnitude Rescaling**
Because calibration mathematically shrinks the Frobenius norm of the weights, standard merging will result in a model with weakened activations. We compute a restoration scalar $\gamma$:
$$\Delta W_{calib} = \frac{1}{T} \sum_{t=1}^T \widetilde{\Delta W}_t$$
$$\gamma = \frac{\frac{1}{T} \sum_{t=1}^T ||B_t A_t||_F}{||\Delta W_{calib}||_F}$$
$$\Delta W_{Pico} = \gamma \Delta W_{calib}$$

---

## 3. WBP Merge (Woodbury B-Space Preconditioning)

**Motivation:** Exact SVD on $B_{all}$ requires $\mathcal{O}(d_{out} \cdot (Tr)^2)$ operations and suffers from severe GPU synchronization bottlenecks due to its iterative nature. WBP replaces SVD entirely with a single closed-form covariance projection that mathematically guarantees the exact same suppression curve.

### 3.1 Underlying Assumptions of WBP
1. **Covariance Equivalence:** The "crowdedness" of $B$-space identified by Pico is mathematically identical to high variance in the uncentered covariance matrix $C = B_{all} B_{all}^\top$.
2. **Tikhonov Equivalence:** Instead of discrete singular value binning, we can apply a continuous Tikhonov-regularized inverse filter $(I + \lambda C)^{-1}$ to suppress dominant eigenvectors.

### 3.2 Proof of Equivalence (Bridging WBP and Pico)

Let the uncentered covariance of the $B$-space be:
$$C = B_{all} B_{all}^\top \in \mathbb{R}^{d_{out} \times d_{out}}$$

We propose the preconditioning operator $S_{wbp}$:
$$S_{wbp} = (I_{d_{out}} + \lambda C)^{-1}$$

**Theorem:** *There exists a scalar $\lambda$ such that $S_{wbp}$ applies the exact same directional scaling as $S_{pico}$.*

**Proof:**
By substituting $B_{all} = U \Sigma V^\top$ into the covariance definition:
$$C = (U \Sigma V^\top)(U \Sigma V^\top)^\top = U \Sigma V^\top V \Sigma U^\top = U \Sigma^2 U^\top$$
Thus, the eigenvalues of $C$ are exactly the squared singular values $\sigma_j^2$.

Expanding $S_{wbp}$ using this eigendecomposition:
$$S_{wbp} = (I + \lambda U \Sigma^2 U^\top)^{-1} = U (I + \lambda \Sigma^2)^{-1} U^\top$$
The scaling factor applied to eigenvector $u_j$ is therefore:
$$\text{Scale}_j = \frac{1}{1 + \lambda \sigma_j^2}$$

To match Pico's scaling, we set $\text{Scale}_j = \alpha_j$:
$$\frac{1}{1 + \lambda \sigma_j^2} = \frac{1}{1 + (T-1) \frac{\sigma_j^2}{\sum \sigma_k^2}}$$
Solving for $\lambda$:
$$\lambda \sigma_j^2 = (T-1) \frac{\sigma_j^2}{\sum \sigma_k^2} \implies \lambda = \frac{T - 1}{\sum_{k=1}^{Tr} \sigma_k^2}$$

Because the sum of eigenvalues is exactly the trace of the matrix, we achieve our exact, SVD-free parameter:
$$\lambda = \frac{T - 1}{\text{Tr}(B_{all} B_{all}^\top)} = \frac{T - 1}{\text{Tr}(B_{all}^\top B_{all})} \quad \blacksquare$$

### 3.3 Proof of Scalability (The Woodbury Derivation)

Directly calculating $S_{wbp} = (I_{d_{out}} + \lambda B_{all} B_{all}^\top)^{-1}$ requires inverting a massive $d_{out} \times d_{out}$ matrix (e.g., $4096 \times 4096$). We bypass this cubic bottleneck using the **Woodbury Matrix Identity**.

**The Identity:**
$$(A + UCV)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}$$

**Application:**
We map our variables to the identity:
* Let $A = I_{d_{out}}$
* Let $U = B_{all}$
* Let $C = \lambda I_{Tr}$
* Let $V = B_{all}^\top$

Substituting these into the Woodbury identity yields:
$$(I_{d_{out}} + B_{all} (\lambda I_{Tr}) B_{all}^\top)^{-1} = I_{d_{out}} - I_{d_{out}} B_{all} \left( \frac{1}{\lambda} I_{Tr} + B_{all}^\top I_{d_{out}} B_{all} \right)^{-1} B_{all}^\top I_{d_{out}}$$

Simplifying this expression:
$$S_{wbp} = I_{d_{out}} - B_{all} \left( \frac{1}{\lambda} I_{Tr} + B_{all}^\top B_{all} \right)^{-1} B_{all}^\top$$

Let the Gram matrix be defined as $G = B_{all}^\top B_{all} \in \mathbb{R}^{Tr \times Tr}$. The final operator becomes:
$$S_{wbp} = I_{d_{out}} - B_{all} \left( \frac{1}{\lambda} I_{Tr} + G \right)^{-1} B_{all}^\top \quad \blacksquare$$

### 3.4 The Final WBP Algorithm

1. **Compute Gram Matrix:** $G = B_{all}^\top B_{all}$
2. **Compute Optimal $\lambda$:** $\lambda = \frac{T - 1}{\text{Tr}(G)}$
3. **Compute Kernel Inverse:** $K = \left( \frac{1}{\lambda} I_{Tr} + G \right)^{-1}$
   *(Note: $K$ is exceptionally small, size $Tr \times Tr$, requiring mere microseconds to invert).*
4. **Apply to Task Matrices:** Instead of instantiating the $d_{out} \times d_{out}$ operator, apply the transformation via right-to-left associativity to maximize GEMM efficiency:
   $$\tilde{B}_t = B_t - B_{all} \left[ K (B_{all}^\top B_t) \right]$$
5. **Merge and Rescale:** Calculate $\widetilde{\Delta W}_t = \tilde{B}_t A_t$, average them, and apply the exact same Frobenius magnitude rescaling $\gamma$ as defined in Pico (Step 5).

---

## 4. Complexity Comparison

* **Pico Calibration Cost:** $\mathcal{O}\left(d_{out} \cdot (Tr)^2\right)$ via iterative Golub-Reinsch SVD. This is heavily bottlenecked by GPU thread synchronization limits.
* **WBP Calibration Cost:** $\mathcal{O}\left(d_{out} \cdot (Tr)^2\right)$ via block Matrix Multiplications + $\mathcal{O}\left((Tr)^3\right)$ for the kernel inversion. Because $Tr \ll d_{out}$, the cubic inversion is negligible, and the matrix multiplications operate at theoretical peak TFLOPs on modern Tensor Cores.