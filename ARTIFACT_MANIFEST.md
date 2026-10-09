# Artifact manifest

| Experiment block | Retained artifact | Main estimand | Scope of interpretation |
|---|---|---|---|
| Four-decoder task calibration | `results/exp04_task_stability__*.json` and `.data.json` | Within-decoder variation across ten tasks | Empirical reference for this probe suite, not a universal cutoff |
| Behavior and compression | `results/exp03_behavior_vs_geometry__*.json` and `.data.json` | Profile displacement paired with perplexity | Quantization and pruning within the recorded scaling block |
| Absolute profiles | `results/exp08_absolute_triad__*.json` and `.data.json` | Unscaled descriptor profiles | Amplitude-sensitive descriptive comparison |
| Activation patching | `results/exp07_patching__*.json` and `.data.json` | Association between profile and intervention effects | Negative diagnostic result, not causal validation of nDNA |
| Dashboard cohort | `results/dashboard_profiles.json` | Shape-normalized depth profiles | Identification, related-variant retrieval, and operation case studies |
| Independent-sample matrices | `results/squad_report_matrices.json` | Profile agreement across independently sampled SQuAD sets | Sample-size identification in the recorded candidate cohort |
| Alignment | `results/alignment_sft_dpo.json` | Base, SFT, and DPO profile comparisons | Resolution-limit result on the listed Litmus and HarmBench probes |
| Recursive self training | `results/recursive_runs.json` | Generational profile and behavioral trajectories | Collapse experiment over the recorded 10–17-generation runs |
| Additional evaluations | `results/additional_evaluations.json` | Baselines and matched-position analyses | Aggregate author-verified results with protocols described in the paper |
| Close-pair amplitude | `results/closepair_raw.json` | Unscaled close-variant profiles | Tests whether amplitude separates a close pair |

## Estimator notes

- Fisher–Rao length is the primary empirical channel.
- Retained field experiments use the projected ambient-mean estimator documented in the paper.
- Corrected exterior-turn curvature is retained as a secondary descriptor and evaluated through ablations.
- Some older dashboard arrays contain an interior-angle curvature estimator. The analysis labels that estimator separately and does not treat it as evidence for the corrected curvature definition.
- The 45 pairwise distances from ten tasks are dependent unique pairs, not 45 independent task replications.
- Threshold comparisons are valid only when probe, scaling, channels, and distance convention match the calibration block.

