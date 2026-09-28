# Does trajectory noise train as well as the exact channel?

The study behind #311. `noise_method="trajectories"` (#229) samples depolarizing noise as
Pauli trajectories at pure-state cost; its gradient is unbiased but noisier than the exact
`"density"` one. This compares the two, and noiseless training, on held-out data.

## Setup

- **Script:** `examples/study_trajectory_noise.py` (unchanged since the run). Raw results, one
  JSON line per run: `docs/results/trajectory_noise_study.jsonl`.
- **Data:** scikit-learn's breast-cancer set (569 samples, 37 % positive). It is bundled with
  scikit-learn, so the study runs offline; the credit-card and UCI benchmarks named in the issue
  need data or code that is not on this branch (see *Limits*).
- **Splits:** for each of 5 seeds, a stratified 60/20/20 train/validation/test split, with
  standardisation and PCA fitted on the training rows. The seed also fixes the initial weights,
  so every method starts from the same point.
- **Model:** `HybridBinaryClassifier(n_qubits, n_qubits, 2)` with 4 and 6 qubits, on
  `default.qubit` with backprop, trained with focal loss, Adam (lr 0.05), batch 32, up to
  30 epochs, and MCC early stopping (patience 10).
- **Methods:** noiseless, `density`, `trajectories` with `k = 1` and `k = 4` draws per sample,
  at p = 0.01 and 0.05 after every gate (`"all"`), and p = 0.05 before measurement (`"end"`).
- **Scores:** test MCC clean (validation threshold), and test MCC under the training noise
  (threshold re-tuned on the validation rows under that noise, as a model deployed on a noisy
  device would be).
- **Environment:** PennyLane 0.45.1, torch 2.14, one laptop CPU; 120 runs.

## Results

Test MCC under the training noise, mean ± sd over 5 seeds, and seconds per run:

| qubits | p | position | noiseless | density | trajectories k=1 | trajectories k=4 |
|---|---|---|---|---|---|---|
| 4 | 0.01 | all | 0.850 ± 0.071 | 0.861 ± 0.094 (17 s) | 0.902 ± 0.024 (8 s) | 0.893 ± 0.064 (12 s) |
| 4 | 0.05 | all | 0.848 ± 0.079 | 0.872 ± 0.050 (22 s) | 0.825 ± 0.085 (10 s) | 0.713 ± 0.326 (10 s) |
| 4 | 0.05 | end | 0.846 ± 0.072 | 0.839 ± 0.039 (13 s) | 0.856 ± 0.059 (6 s) | 0.882 ± 0.037 (6 s) |
| 6 | 0.01 | all | 0.875 ± 0.068 | 0.876 ± 0.087 (137 s) | 0.927 ± 0.039 (13 s) | 0.905 ± 0.067 (16 s) |
| 6 | 0.05 | all | 0.855 ± 0.065 | 0.883 ± 0.053 (120 s) | 0.673 ± 0.351 (16 s) | 0.922 ± 0.040 (17 s) |
| 6 | 0.05 | end | 0.847 ± 0.072 | 0.890 ± 0.035 (60 s) | 0.884 ± 0.058 (6 s) | 0.879 ± 0.101 (7 s) |

The noiseless model is trained once per seed, taking about 5 s. Total training time: density
1843 s, trajectories 298 s (k=1) and 340 s (k=4).

## What this shows

1. **At p = 0.01, and with noise only before measurement, trajectories train as well as the
   exact channel.** Paired by seed, the mean difference from density is between −0.01 and
   +0.05, which is within the seed-to-seed spread.
2. **At p = 0.05 after every gate, trajectory training sometimes fails.** 3 of 20 trajectory
   runs collapsed, against 0 of 10 density runs:
   - k=1 at 6 qubits, seeds 0 and 4: test MCC 0.19 and 0.42;
   - k=4 at 4 qubits, seed 1: test MCC 0.13.

   In the other runs it matched density, and k=4 at 6 qubits beat it on 4 of 5 seeds. The
   extra gradient variance is the likely cause. k=4 reduces it (no collapse at 6 qubits) but
   does not remove it (one collapse at 4).
3. **Cost.** Trajectories trained about 2× faster at 4 qubits and 8–9× faster at 6. Density
   becomes impractical past about 6 qubits (#229), where trajectories keep pure-state memory.
4. **Noise-aware training did not beat noiseless training here**, even under noise: the
   noiseless model scores within about 0.03 of density in every setting. At these noise levels
   on this data the noise is too weak to hurt a noiselessly trained model much, so this study
   says nothing either way about noise-aware training's value at stronger noise.

## Recommendation

- **Keep `noise_method="density"` as the default** wherever it fits in memory (up to about 6
  qubits). It never failed here.
- **Beyond that, use `"trajectories"` with `noise_trajectories ≥ 4`** (raised to **≥ 8** by
  the follow-up in `trajectory-collapse-study.md`, #347, which found no collapse at k = 8), and check the runs,
  especially at noise strengths of a few percent per gate. With 5 seeds, a collapse is visible
  as an outlier in the seed spread.
- **Do not switch the default automatically by qubit count** on this evidence. The occasional
  collapse is a behaviour change that a default should not introduce silently.

## Limits

- One small, easy dataset (noiseless MCC ≈ 0.85), 4 and 6 qubits, 5 seeds. With 5 paired seeds,
  a Wilcoxon signed-rank test cannot go below p = 0.0625, so none of the differences above is
  statistically significant. The collapses are the robust observation.
- The issue asked for the credit-card and UCI benchmarks through `run_benchmark`. Those need the
  Kaggle download and code on other open stacks (the loaders in #266, the benchmark runner in
  #269–#297), so they aren't used here. The script's `run_one` is the unit to port once those
  land.
- The learning rate and schedule were not tuned per method. A lower learning rate may prevent
  the collapses; that is the obvious follow-up. (#347 tested it: it does not; k = 8 does. See
  `trajectory-collapse-study.md`.)
