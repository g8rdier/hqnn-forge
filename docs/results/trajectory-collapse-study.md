# Can the collapse of trajectory-noise training be prevented?

The study behind #347. #311 (`trajectory-noise-study.md`) found that training with
`noise_method="trajectories"` at p = 0.05 after every gate collapsed in 3 of 20 runs, while the
exact `"density"` channel never did. This tests four mitigations over twice the seeds.

## Setup

- **Script:** `examples/study_trajectory_collapse.py`, which reuses the data handling of
  `examples/study_trajectory_noise.py`. Raw results: `docs/results/trajectory_collapse_study.jsonl`
  (main grid, 380 runs) and `docs/results/trajectory_collapse_settings.jsonl` (the milder
  settings, 120 runs).
- **Same data, splits, model and training as #311:** breast-cancer data, stratified 60/20/20
  split per seed, `HybridBinaryClassifier(n, n, 2)` on `default.qubit` with backprop, focal
  loss, Adam, batch 32, up to 30 epochs with MCC early stopping (patience 10). The score is test
  MCC under the training noise, with the threshold re-tuned on the noisy validation rows.
- **Seeds 0 to 9**, at 4 and 6 qubits. Seeds 0 to 4 are #311's, and the baseline runs reproduce
  #311 exactly, run for run (MCC equal to machine precision). That holds at the milder settings
  too, so pairing those with #311's density runs pairs the runs this script would have made.
- **Mitigations**, each at k = 1, 4 and 8 draws per sample:
  - `lr0.05`: the baseline, as in #311;
  - `lr0.02`, `lr0.01`: a lower learning rate;
  - `clip`: gradient-norm clipping at 0.1 (lr 0.05). The median unclipped norm was 0.05 at
    k = 1 and 0.10–0.12 at k = 4 and 8, so the bound acted on 7–9 % of steps at k = 1 and
    about half at k = 4 and 8;
  - `warmup`: noise ramped linearly from p/5 to p over the first 5 epochs (lr 0.05).
- **References:** density at all three learning rates, and noiseless training, on the same
  seeds.
- **Collapse:** test MCC under noise below 0.5. The successful runs in #311 all scored at least
  0.68, and the collapses here scored 0.05–0.47, so the cut is not borderline.
- **Environment:** PennyLane 0.45.1, torch 2.14, one 16-core CPU, 14 single-threaded worker
  processes.

## Results

### p = 0.05 after every gate

Collapsed runs of 10, and mean test MCC under noise:

| qubits | k | lr 0.05 | lr 0.02 | lr 0.01 | clip | warm-up |
|---|---|---|---|---|---|---|
| 4 | 1 | 1 · 0.809 | 0 · 0.853 | 0 · 0.690 | 1 · 0.769 | 0 · 0.862 |
| 4 | 4 | 1 · 0.800 | 0 · 0.879 | 0 · 0.867 | 1 · 0.780 | 0 · 0.900 |
| 4 | 8 | **0 · 0.904** | 0 · 0.898 | 0 · 0.892 | 0 · 0.890 | 0 · 0.882 |
| 6 | 1 | 2 · 0.774 | 1 · 0.834 | 2 · 0.748 | 2 · 0.748 | 1 · 0.773 |
| 6 | 4 | 0 · 0.911 | 2 · 0.790 | 2 · 0.734 | 0 · 0.894 | 0 · 0.897 |
| 6 | 8 | **0 · 0.888** | 0 · 0.926 | 0 · 0.881 | 0 · 0.914 | 0 · 0.921 |

| qubits | density lr 0.05 | density lr 0.02 | density lr 0.01 | noiseless |
|---|---|---|---|---|
| 4 | 0 · 0.891 | 0 · 0.867 | 0 · 0.871 | 0 · 0.850 |
| 6 | 0 · 0.853 | 0 · 0.896 | 1 · 0.844 | 0 · 0.872 |

In total, over the five variants and both sizes, k = 1 collapsed in **10 of 100** runs,
k = 4 in **6 of 100** and k = 8 in **0 of 100**. Density collapsed once in 60, at lr 0.01.

Median seconds per run at lr 0.05. They are comparable within this table only (14 runs shared the CPU):

| qubits | k = 1 | k = 4 | k = 8 | density |
|---|---|---|---|---|
| 4 | 27 | 30 | 30 | 87 |
| 6 | 75 | 87 | 102 | 1690 |

### The milder settings

No run collapsed at p = 0.01 after every gate or at p = 0.05 before measurement, at any k.
Mean test MCC under noise, against #311's density runs:

| p | position | qubits | density (5 seeds) | k = 1 | k = 4 | k = 8 |
|---|---|---|---|---|---|---|
| 0.01 | all | 4 | 0.861 | 0.894 | 0.895 | 0.869 |
| 0.01 | all | 6 | 0.876 | 0.906 | 0.899 | 0.879 |
| 0.05 | end | 4 | 0.839 | 0.884 | 0.888 | 0.895 |
| 0.05 | end | 6 | 0.890 | 0.877 | 0.893 | 0.910 |

Paired by seed against density, k = 8's mean difference lies between −0.012 and +0.058.

## What this shows

Like #311, everything below was measured on the breast-cancer proxy only: one small dataset,
4 and 6 qubits, 10 seeds. It has not been checked on the credit-card or UCI benchmark data;
that re-run is #414.

1. **k = 8 removes the collapses; the other mitigations do not.** k = 8 never collapsed, under
   any variant, at either size. Against k = 1 that is 0 of 100 against 10 of 100
   (Fisher's exact test p = 0.002), and against k = 4, 0 against 6 (p = 0.03). Both p-values
   pool the five variants, which share seeds and are therefore not independent (k = 1 at
   6 qubits collapsed on seed 0 under every variant), so they overstate the evidence. Read them
   as indicative. The pattern is consistent, though: collapses fall from k = 1 to k = 4 to
   k = 8 at every size.
2. **A lower learning rate does not help, and at k = 4 it hurts.** At 6 qubits, k = 4 went from
   0 collapses at lr 0.05 to 2 at lr 0.02 and 2 at lr 0.01. At k = 1 no learning rate brought
   the collapses to zero: 3, 1 and 2 of 20 at lr 0.05, 0.02 and 0.01. Collapse here is not
   overshooting, so a smaller step does not prevent it.
3. **Clipping and warm-up help only partly.** Clipping at 0.1 barely acts at k = 1 (7–9 % of
   steps) and left its collapses in place. The warm-up removed k = 4's collapses (0 of 20) but
   not k = 1's (1 of 20).
4. **k = 8 matches density, and costs a fraction of it.** At lr 0.05, paired over 20 seeds, k = 8
   scored +0.024 MCC above density on average (Wilcoxon p = 0.23, no significant difference).
   It took 1.1× (4 qubits) and 1.4× (6 qubits) the time of k = 1, not 8×, because on
   `default.qubit` with backprop the k draws run as one vectorised batch. At 6 qubits it was
   about 17× faster than density.
5. **Density is not immune either.** It collapsed once, at lr 0.01 on seed 0 at 6 qubits, the
   seed where k = 1 collapsed under every variant. Some splits are simply hard to train at
   this noise level, and fewer draws make hitting that failure more likely.

## Recommendation

- **Train with `noise_trajectories ≥ 8`** when using `noise_method="trajectories"` at noise of a
  few percent per gate. On backprop devices it costs little more than k = 1.
- **Keep lr 0.05**; don't lower it to stabilise trajectory training.
- **Warm-up and clipping aren't worth adding to the library** on this evidence: k = 8 does
  better alone, and adds no option.
- **The default stays `noise_trajectories=1`.** On the adjoint path, which
  `device_name="auto"` picks above 12 qubits (#349), each sample runs separately, so k = 8
  costs about 8× there. This study measured only 4 and 6 qubits on backprop, so it does not
  justify that cost as a default. The recommendation is in the `hqnn_forge.noise` docstring.
- **`density` remains the default** noise method where it fits (up to about 6 qubits).

## Limits

- One small proxy dataset (breast cancer; the benchmark datasets are #414), 4 and 6 qubits,
  one noise model (depolarizing). The ">12 qubit" regime,
  where trajectories matter most, is extrapolated, not measured.
- 10 seeds per cell. A collapse rate of a few percent at k = 8 cannot be ruled out: 0 of 100
  bounds it below about 3 % (95 %, one-sided), assuming independent runs, and they are not
  fully independent.
- Timings come from a shared CPU and are only comparable within this study.
