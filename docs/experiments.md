# Experiments Plan: Validating and Extending WBP for LoRA Merging

**Scope for this thesis:** Experiments **E0–E5** below are in scope and will be executed.
**E6–E13** are retained in this document as a deferred roadmap (cited in the thesis's Future Work
section) but are **not** executed as part of this work. They're kept here, condensed, so the scope
decision is traceable and so they're ready to pick up if time/compute allows or as follow-on work.

**Context.** The thesis follows a 4-phase narrative:
1. **Phase 1 (E0):** Verify the original claims of B-space overlap from prior work.
2. **Phase 2 (E1):** Introduce WBP and verify operator-level equivalence to Pico on real adapters.
3. **Phase 3 (E2-E4):** Benchmark downstream accuracy (across Task Arithmetic and TIES), and quantify WBP's computational advantage over Pico.
4. **Phase 4 (E5):** Explore generalizing the calibration filter via the $\beta$ parameter.

---

## Experiment Map (In Scope)

| ID | Name | Compute | Depends on |
|----|------|---------|------------|
| E0 | Subspace overlap verification | Low | Trained adapters |
| E1 | Operator-level equivalence on real adapters | Low | Trained adapters (small scale OK) |
| E2 | Downstream accuracy parity (No-Cal / Pico / WBP) | Med | Same adapters as E1 (Task Arithmetic + TIES) |
| E3 | Wall-clock & memory scaling vs. $d_{out}$ | Low | Synthetic matrices only — no training needed |
| E4 | Scaling vs. $T$ and rank $r$ | Med | Synthetic timing (cheap) + reuse E2's adapters for the accuracy-side curve |
| E5 | Decoupled-$\lambda$ sweep ($\beta$-WBP), exploratory | Med | Reuse E2's adapters/pipeline |

---

## E0. Subspace overlap verification

**Objective.** Verify the core claim from the original Pico paper: independently trained LoRA adapters exhibit significantly higher subspace overlap/interference in their $B$ matrices than in their $A$ matrices, necessitating $B$-space preconditioning.

**Setup.** 
- Extract a broad, random sample of independently trained PEFT LoRA models from the Hugging Face Hub (filtering out models from the same author to ensure independent training).
- Group adapters into buckets matching by base model, rank ($r \in \{8, 16, 32, 64\}$), and target modules.

**Steps.**
1. For every valid pair of adapters within a bucket, extract the $A_t$ and $B_t$ matrices for shared layers and modules.
2. Compute the subspace overlap between $A_1$ and $A_2$ (denoted $O_A$) and $B_1$ and $B_2$ (denoted $O_B$) using QR decomposition and the squared Frobenius norm.
3. Discard the pair if the overlap in both the $A$ and $B$ matrices is close to 1 (this excludes near-identical adapters published by different users or checkpoints from the same run).
4. Aggregate the results across modules and layers to calculate the mean $O_A$, mean $O_B$, and the gap ($O_B - O_A$).

**Expected result.** The aggregated gap ($O_B - O_A$) should be consistently positive and statistically significant, confirming that $B$-spaces suffer from substantially more crowding than $A$-spaces.

---

## E1. Operator-level equivalence on real adapters

**Objective.** Confirm $S_{wbp} = S_{pico}$ — and therefore $\tilde B_t^{wbp} = \tilde B_t^{pico}$ —
holds on real trained LoRA weights, not just random matrices. This is the foundation everything
else stands on.

**Setup.** Train $T \geq 3$ task-specific LoRA adapters on a common base model (start small:
Qwen2.5-1.5B or Llama-3.2-3B, rank $r \in \{8, 16\}$, attention $q\_proj$/$v\_proj$). A few thousand
steps per task is enough — you're checking numerical equivalence, not accuracy yet.

**Steps.**
1. For each layer/module, extract $B_1,\dots,B_T$.
2. Compute $S_{pico}$ via SVD; compute $S_{wbp}$ via the Woodbury closed form.
3. Compute $\tilde B_t$ both ways for every $t$, every layer.
4. Log max and mean relative Frobenius error between the two.
5. **Implementation hygiene (not a separate experiment, just don't skip it):** guard the $T=1$ case
   (return $B_t$ unchanged rather than computing $\lambda=0$ and dividing by it in the Woodbury
   form) and check that nothing NaNs out when two adapters are near-duplicates in $B$-space — both
   are one-line fixes but will silently break E2/E4/E5 downstream if missed.

**Expected result.** Differences at or near float32 machine precision (~$10^{-6}$–$10^{-7}$
relative error; you already saw ~$10^{-15}$–$10^{-16}$ in float64 on synthetic matrices). Large
errors (>$10^{-4}$) point to a conditioning issue, not a derivation bug — the math is already
proven exact.

---

## E2. Downstream accuracy parity

**Objective.** Show merged models produced via WBP match Pico's benchmark accuracy end to end —
not just at the operator level, but after the full merge → rescale → evaluate pipeline.

**Setup.** Reuse (or scale down) the paper's four-domain setup: math, coding, finance, medical. If
full retraining at 50k examples/domain is too expensive, use a smaller sample (5–10k/domain) and
say so explicitly — the comparison is relative (No-Cal vs. Pico vs. WBP), not an attempt to
reproduce the paper's absolute numbers.

**Steps.**
1. Train/obtain the $T=4$ adapters (same ones as E1).
2. Run both **Task Arithmetic** and **TIES** with: (a) no calibration, (b) Pico, (c) WBP.
3. Evaluate all three on the same benchmarks.
4. Report per-domain and overall averages.

**Expected result.** Pico and WBP rows statistically indistinguishable (differences within
floating-point/evaluation noise). Both should beat "no calibration" by a margin broadly consistent
with the source paper's reported gain for Task Arithmetic.

**Scope note.** This experiment does **not** include DARE/DELLA/KnOTS/Core Space as self-run rows —
those are out of scope (would have been E7). If you want them in the thesis at all, cite the source
paper's numbers in your discussion as context, clearly labeled as not directly comparable (different
setup/scale), not as a head-to-head table entry.

---

## E3. Wall-clock and memory scaling vs. $d_{out}$

**Objective.** Measure actual speedup, not asserted speedup — this is the experiment your
motivation section currently lacks.

**Setup.** Synthetic $B_{all}$ matrices are fine — you're timing linear algebra, not model quality.
Sweep $d_{out} \in \{512, 1024, 2048, 4096, 8192\}$, fixed $T=4$, $r=16$ ($Tr=64$).

**Steps.**
1. For each $d_{out}$, time (GPU, with warmup iterations excluded):
   - Full thin SVD of $B_{all}$ (Pico's cost) — use `full_matrices=False`, a fair comparison
     requires this.
   - Gram matrix + $Tr\times Tr$ inversion + Woodbury application (WBP's cost).
2. Record wall-clock time and peak GPU memory for each.
3. Plot both vs. $d_{out}$ on log-log axes.

**Expected result.** Both methods scale roughly $O(d_{out})$ for fixed $Tr$ — the win shows up as a
**constant-factor gap** between the curves, not a different slope. State this plainly: the
advantage is hardware utilization (GEMM vs. iterative SVD) plus avoiding the $O(d_{out}^3)$
naive-inversion alternative, not a better asymptotic exponent than Pico's SVD.

---

## E4. Scaling vs. $T$ and rank $r$

**Objective.** Confirm equivalence and speed hold as the merge pool grows and rank increases — the
regime where Pico's SVD cost bites hardest.

**Setup.** Two sub-parts:
- **Timing-only (cheap, synthetic):** sweep $T \in \{2,\dots,6\}$ and $r \in \{8,16,32,64\}$
  independently, re-running E1's equivalence check and E3's timing at each point. No new adapters
  needed.
- **Accuracy-side (optional, reuses E2's adapters):** if you only trained 4 domain adapters in E2,
  cap the progressive-merge accuracy curve at $T=4$ — don't train extra adapters solely for this;
  it's not worth the cost for a curve that's secondary to the timing result.

**Expected result.** Equivalence holds at every grid point. The Pico–WBP timing gap widens as $Tr$
grows (more singular values for SVD to resolve iteratively vs. WBP's $Tr\times Tr$ inversion, which
stays cheap relative to $d_{out}$).

---

## E5. Decoupled-$\lambda$ sweep ($\beta$-WBP) — exploratory

**Objective.** A first look at whether moving away from the exact-Pico-match $\lambda$ can improve
accuracy. **Framed as exploratory in this thesis** — there is no formal tuning protocol (that would
be E6) and no full baseline comparison (E7), so treat this as "does the family contain anything
better than Pico's specific point," not as a proposal of a deployable tuned method.

**Setup.** Define $\lambda_\beta = \beta\cdot\dfrac{T-1}{\text{Tr}(G)}$; $\beta=1$ recovers exact
Pico-equivalence. Sweep $\beta \in \{0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0\}$.

**Steps.**
1. Implement WBP with $\lambda_\beta$.
2. For each $\beta$, run the full merge → rescale → evaluate pipeline from E2 (Task Arithmetic only,
   same adapters).
3. Plot overall benchmark average vs. $\beta$, marking $\beta=1$ as the Pico-equivalent reference.
4. Report the result honestly regardless of shape:
   - Peak away from $\beta=1$ → note this as evidence the family contains better points than Pico's,
     and explicitly flag that a principled (ideally data-free) selection method is needed before
     this is a usable method — name this as the natural next step in Future Work.
   - Flat curve / no clear peak → also a valid, reportable finding: suggests Pico's chosen scaling
     was already close to optimal in this regime.
   - Monotonic degradation either direction → report and interpret (likely over/under-shrinkage of
     task-specific energy); still useful evidence about the shape of the family.

**What NOT to claim from this experiment alone:** that WBP-tuned "beats" Pico as a deployable
result, that the chosen $\beta$ generalizes to other models/domains, or that the data-free property
is preserved (a $\beta$ chosen by looking at benchmark accuracy is not data-free).

---

## Deferred / Future Work (not executed in this thesis)

Kept here for traceability and to seed the thesis's Future Work chapter.

| ID | Name | One-line description |
|----|------|------------------------|
| E6 | Tuning protocol for $\beta$ | A data-free (or minimally-supervised) selection rule for $\beta$, e.g. via spectral diagnostics ($o_{max}$, effective rank) instead of benchmark accuracy. |
| E7 | Full comparison table | Re-implement DARE/DELLA/KnOTS/Core Space in your own setup and report Tuned-WBP alongside them, matching the source paper's Table 1 structure. |
| E8 | Plug into TIES / TSV-M | Re-run E2/E5 with TIES and TSV-M as the downstream merger, not just Task Arithmetic. |
| E9 | Conditioning stress test | Sweep adapter redundancy $\rho$ and check equivalence error vs. condition number of $G$. |
| E10 | Formal edge-case suite | Beyond the E1 guard: near-zero $\lambda$, extreme low-effective-rank degeneracy, NaN/inf checks as a proper test suite. |
| E11 | Cross-backbone transfer | Repeat E1–E3 on a second model family (e.g. Qwen vs. Llama). |
| E12 | Positioning vs. Core Space / TSV-M / SVC | Quantitative re-implementation and comparison, not just literature citation. |
| E13 | Batched multi-layer Woodbury | Batch the $Tr\times Tr$ inversions across all layers in one call; compare to per-layer SVD. |

---

## Suggested Order of Execution

1. **Week 1:** E1 (synthetic + small real adapters), including the $T=1$ guard and basic NaN
   hygiene — catch bugs before they propagate into E2/E4/E5.
2. **Week 1–2 (parallel with above):** E3 — pure synthetic timing, no dependency on adapters.
3. **Week 2–3:** Train/obtain the four domain adapters; re-run E1 on them for confirmation.
4. **Week 3–4:** E2 (accuracy parity).
5. **Week 4:** E4's timing sub-part (cheap, synthetic) — can slot in anytime after E3's code exists.
6. **Week 4–6:** E5 (the sweep + analysis — give this the most time, it's your most
   interpretation-heavy result).
7. **Week 6–7 buffer:** re-runs, multi-seed repeats, writing up Ch.6.

## Logging Checklist (per experiment)

- Exact $d_{out}, r, T$ (and $\beta$ where relevant) for every run.
- Hardware (GPU model, precision — fp32/bf16) for every timing number; never compare timings across
  different precisions.
- Random seed for any synthetic-matrix experiment; report mean ± std over ≥5 seeds, not a single run.
- For accuracy tables: which benchmarks, how many eval examples, which metric (exact-match, pass@k,
  etc.) — match the source paper's conventions so numbers are at least qualitatively comparable.
