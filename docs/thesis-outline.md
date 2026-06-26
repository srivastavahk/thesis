# Thesis Outline

**Scope note (updated):** this outline reflects a thesis built on Experiments **E0–E5** only
(see `experiments.md`). E6–E13 are referenced as Future Work, not as completed results.

**Chosen Title:**
*Woodbury B-Space Preconditioning: Fast and Generalized B-Space Calibration for Model Merging*

**One-line thesis statement** (fill in once results are in):
> *"Pico's discrete spectral calibration is a special case of a continuous Tikhonov-regularized
> filter; this filter has a closed form via the Woodbury identity that avoids per-merge SVD,
> verified exactly on real adapters and shown to be faster in practice; preliminary exploration
> suggests the underlying family may contain points beyond Pico's that warrant further study."*

---

## Front Matter
- Title page, certificate of originality, acknowledgements (institution-specific template)
- Abstract (≤1 page — write **last**)
- List of Figures, List of Tables, List of Abbreviations
- Notation table — recommended: $B_t, A_t, B_{all}, U, \Sigma, C, G, \lambda, \beta, \alpha_j, S, K$

---

## Chapter 1 — Introduction
*(~6–8 pages)*

1.1 **Motivation** — LoRA merging as a practical alternative to joint retraining; the crowding
phenomenon in one paragraph + one figure.

1.2 **Problem Statement** — Pico fixes crowding but its per-layer, per-merge SVD doesn't scale the
way a production merging pipeline would want. State this precisely: not "Pico doesn't work," but
"Pico's calibration step is a computational bottleneck at scale."

1.3 **Research Objectives** (these become your evaluation criteria in Ch.6):
- O1: Verify the original claims of subspace overlap (E0).
- O2: Derive an SVD-free reformulation of Pico's calibration that is provably equivalent, and empirically verify the equivalence holds on real adapters (E1).
- O3: Evaluate end-to-end downstream merging performance across Task Arithmetic and TIES (E2).
- O4: Quantify the computational advantage over the SVD route (E3, E4).
- O5: Take a first, exploratory look at whether decoupling the filter from the exact-match
  constraint can move accuracy (E5).

1.4 **Contributions**
- A closed-form Tikhonov-filter equivalent to Pico's spectral binning (Theorem, §4.2).
- A Woodbury-identity-based algorithm (WBP) requiring only a $Tr\times Tr$ inversion instead of
  $d_{out}\times d_{out}$ (§4.3).
- Empirical validation of exact equivalence and efficiency gains, on real adapters (Ch.6).
- An exploratory generalization ($\beta$-WBP) that frames Pico as one point in a continuous family,
  with preliminary evidence reported and a concrete roadmap for formalizing it (Ch.6, Ch.8).

1.5 **Thesis Organization** — one paragraph, standard.

---

## Chapter 2 — Literature Review
*(~8–10 pages)*

2.1 **Parameter-Efficient Fine-Tuning and LoRA** — brief; cite, define notation consistent with Ch.3.

2.2 **Asymmetry Between $A$ and $B$** — Zhu et al., HydraLoRA, LoRA-FA, LoRA+, DoRA, FedSA-LoRA. Sets
up why merge-interference asymmetry (Pico's premise) is plausible.

2.3 **Model and LoRA Merging** — Task Arithmetic, model soups, TIES, DARE, DELLA, LoRA Soups,
LoRA-LEGO. Descriptive only — you are not re-implementing these for a quantitative table (that was
E7, deferred), so keep this section to characterizing each method's mechanism, not benchmarking it.

2.4 **Shared-Basis and Low-Rank Merging Methods** — KnOTS, TSV-M, Core Space, Pico, Li et al.'s SVC.
Spend the most space here. Close with a **qualitative** comparison table (what each method
calibrates, pre/post-merge, per-merge cost) — explicitly qualitative since you have no
re-implementation of these (that would be E12, deferred). Label the table as such.

2.5 **Gap Statement** — "None of the above address the computational cost of computing the
calibration basis itself when $d_{out}$ is large and many merges are performed" → leads into Ch.4.

---

## Chapter 3 — Background and Problem Formulation
*(~8–10 pages)*

3.1 **LoRA Update Formulation** — $\Delta W_t = B_tA_t$, notation, dimensions.

3.2 **The Merging Problem and Interference** — linear merge, the toy shared/specific derivation,
shared-to-specific ratio scaling by $T$.

3.3 **Pico: Pre-Merge Calibration in Output Space** — full recap: shared basis via SVD, sharing
score $s_j$, coefficients $\alpha_j$, operator $S_{pico}$, magnitude rescaling $\gamma$. State the
two assumptions explicitly as empirical premises, citing the source paper's evidence (effective
rank, overlap gap tables).

3.4 **Mathematical Preliminaries for Chapter 4** — SVD/eigendecomposition relation between $C=BB^\top$
and $G=B^\top B$; the Woodbury matrix identity; Tikhonov/ridge regularization as a concept.

3.5 **Computational Cost of Pico** — formalize the $O(d_{out}\cdot(Tr)^2)$ SVD cost and the
practical GPU-parallelization issue. This is the precise problem Ch.4 solves.

---

## Chapter 4 — Proposed Method: Woodbury B-Space Preconditioning (WBP)
*(~12–15 pages — core theoretical chapter)*

4.1 **Overview and Design Goals** — restate O1–O4 as design goals.

4.2 **Equivalence Theorem** — full statement and proof: eigenvalue correspondence → diagonalized
filter → solving for $\lambda$ independent of $j$ → trace identity. Include a remark on *why* this
works (Pico's $\alpha_j$ has exactly the functional form of a ridge filter — not a coincidence).

4.3 **Efficient Computation via the Woodbury Identity** — motivate the $O(d_{out}^3)$ naive cost;
full substitution to $S_{wbp}=I-B_{all}(\tfrac1\lambda I_{Tr}+G)^{-1}B_{all}^\top$; final algorithm
as a numbered block.

4.4 **Complexity Analysis** — side-by-side with Pico: both $O(d_{out}\cdot(Tr)^2)$ asymptotically.
Be explicit that the advantage is (a) avoiding the $O(d_{out}^3)$ naive-inversion alternative and
(b) GEMM-vs-iterative-SVD hardware utilization — not a better Big-O than Pico itself.

4.5 **The $\beta$-WBP Family — an Exploratory Generalization**
- Define $\lambda_\beta=\beta\cdot\lambda$, $\beta=1$ recovers exact Pico-equivalence.
- Frame explicitly as an open question to be explored empirically in Ch.6 (E5), **not** a method
  proposed and validated in this thesis. State plainly that a selection protocol for $\beta$ (E6)
  and a full comparative evaluation (E7) are left to future work.

4.6 **Edge Cases and Implementation Notes** — $T=1$ guard, conditioning of $G$, precision notes
(bf16 vs fp32). Short and practical.

---

## Chapter 5 — Experimental Setup
*(~6–8 pages)*

5.1 **Models and Datasets** — backbone, domains, training data/sample sizes, LoRA hyperparameters
(table format).

5.2 **Baselines** — **self-run:** No Calibration, Pico, WBP, $\beta$-WBP sweep. DARE/DELLA/KnOTS/
Core Space are **not** re-implemented; if referenced at all, cite the source paper's numbers in the
Discussion as context only, explicitly labeled as obtained under a different setup/scale and not
directly comparable.

5.3 **Downstream Merger** — **Task Arithmetic and TIES.** State explicitly that both methods are used to benchmark the calibration techniques.

5.4 **Evaluation Benchmarks and Metrics** — list per domain, scoring method.

5.5 **Hardware and Implementation Details** — GPU(s), precision, timing methodology (warmup,
repeats, synchronization before stopping the clock).

5.6 **Experiment-to-Chapter Mapping** — small table: E0→§6.1, E1→§6.2, E2→§6.3, E3/E4→§6.4, E5→§6.5.

---

## Chapter 6 — Results and Analysis
*(~10–14 pages)*

6.1 **Subspace Overlap Verification (E0)** — report the empirical gap ($O_B - O_A$) on randomly sampled Hugging Face adapters; confirms the original paper's premise of $B$-space crowding.

6.2 **Operator-Level Equivalence (E1)** — table of max/mean error between $S_{pico}$ and $S_{wbp}$
across layers; empirically confirms the §4.2 theorem.

6.3 **Downstream Accuracy Parity (E2)** — No-Calibration vs. Pico vs. WBP across both Task Arithmetic and TIES. State explicitly that this is a 6-way comparison using your own trained adapters.

6.4 **Computational Efficiency (E3, E4)** — wall-clock/memory vs. $d_{out}$ (headline plot);
scaling vs. $T$ and rank $r$.

6.5 **Exploratory $\beta$-WBP Sweep (E5)** — accuracy vs. $\beta$ curve(s) for Task Arithmetic.
Report whichever outcome occurred (peak / flat / monotonic) plainly, and interpret it honestly —
this is the section most likely to draw examiner questions, so the interpretation paragraph matters
more than the plot itself. Explicitly state what this result does and doesn't license you to claim
(see "what not to claim" in `experiments.md`'s E5 entry).

6.6 **Synthesis** — short closing subsection tying 6.1–6.5 together: crowding is real (E0), equivalence holds (E1), parity holds across mergers (E2), efficiency gain is real and characterized (E3/E4), and the $\beta$ family looks [promising /
inconclusive / not advantageous] pending the formal protocol described in Ch.8 (E5).

---

## Chapter 7 — Discussion
*(~5–7 pages — slightly expanded given the narrower experimental scope, to carry the honesty load)*

7.1 **Interpretation of Results** — synthesize 6.1–6.6 into your one-line thesis statement.

7.2 **Limitations** (be specific — this section is doing real work given the reduced scope):
- Only Task Arithmetic and TIES mergers were evaluated; TSV-M behavior unknown.
- No self-run comparison against DARE/DELLA/KnOTS/Core Space; any literature numbers cited are not
  directly comparable due to differing setups/scale.
- Single backbone model; cross-architecture generality (E11) untested.
- The $\beta$-WBP result (E5) has no principled, data-free selection protocol behind it — it is
  evidence about the shape of the family, not a deployable recommendation.
- No dedicated numerical-conditioning stress test (E9) — only the basic guard from E1's
  implementation hygiene step.

7.3 **Threats to Validity** — benchmark/eval sample sizes, single-seed vs. multi-seed reporting,
possible benchmark contamination if using widely-known eval sets.

---

## Chapter 8 — Conclusion and Future Work
*(~3–4 pages)*

8.1 **Summary of Contributions** — restate O1–O4, state plainly whether each was met.

8.2 **Future Work** (this is now a substantive section, not filler — it's where E6–E13 live):
- **A data-free tuning protocol for $\beta$** (E6): e.g. selecting $\beta$ via spectral diagnostics
  ($o_{max}$, effective rank) rather than downstream accuracy, preserving Pico's data-free property.
- **Full comparative evaluation** against DARE/DELLA/KnOTS/Core Space under a tuned $\beta$ (E7).
- **Merger generality**: repeat with TSV-M (E8).
- **Numerical robustness**: conditioning stress tests and a formal edge-case suite (E9, E10).
- **Cross-backbone validation** (E11) and **quantitative positioning** against Core Space/TSV-M/SVC
  (E12).
- **Batched multi-layer Woodbury inversion** for full-model deployment efficiency (E13).

---

## References
- BibTeX from the source paper's reference list, plus anything added for §2.4/§7 positioning.

## Appendices
- **A: Full proof transcript** — fully expanded line-by-line derivation from §4.2/4.3.
- **B: Additional tables** — full benchmark-level results for E1–E5.
- **C: Hyperparameters and training configuration** — full reproducibility details.
- **D: Code listing / repository link.**

---

## Page Budget Check (revised for E1–E5 scope)

| Section | Est. pages |
|---|---|
| Front matter | ~8–10 (not counted in body) |
| Ch.1 Introduction | 6–8 |
| Ch.2 Literature Review | 8–10 |
| Ch.3 Background | 8–10 |
| Ch.4 Proposed Method | 12–15 |
| Ch.5 Experimental Setup | 6–8 |
| Ch.6 Results | 10–14 |
| Ch.7 Discussion | 5–7 |
| Ch.8 Conclusion | 3–4 |
| **Main body total** | **~58–76 pages** |

Slightly shorter than the original full-scope estimate (Ch.6 shrinks, Ch.7 grows to absorb the
limitations load) — still a perfectly defensible M.Tech length, especially with Ch.2–4 carrying
real theoretical weight.

## Writing Order

1. Ch.3 (background) — writable now, no dependency on experiments.
2. Ch.4 (method) — writable now; the derivation is already verified.
3. Ch.5 (setup) — writable once models/datasets are finalized, before results exist.
4. Ch.2 (related work) — writable in parallel, anytime.
5. Ch.6 (results) — only after `experiments.md` (E1–E5) is executed.
6. Ch.1, Ch.7, Ch.8, Abstract — write last, once you know what story Ch.6 tells.
