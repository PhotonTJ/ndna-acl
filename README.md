# nDNA: Layerwise Prediction Profiles for Comparing Language-Model Variants

This anonymous artifact accompanies the submission **“nDNA: Layerwise Prediction Profiles for Comparing Language-Model Variants.”** nDNA is a metric and comparison protocol for depth-resolved next-token prediction profiles. The repository contains the retained aggregate profiles, experiment drivers, metric implementation, and the script used to regenerate the paper’s derived tables and figures.

## Reproduce the reported analysis

The paper’s statistical summaries, LaTeX table macros, and figures can be regenerated without model inference:

```bash
python -m pip install -r requirements-analysis.txt
python analysis.py
```

The command reads the JSON files in `results/` and writes:

- `results/derived.json`: derived statistics and table values;
- `tables.tex`: generated LaTeX macros;
- `figures/fig_*.png`: paper figures.

`python analysis.py --nofig` regenerates only the statistics and table macros. The supplied aggregate profiles do not support prompt-level bootstrap intervals except where prompt-level recursive-self-training records were retained.

## Repository map

- `analysis.py`: retained-output analysis and figure generation.
- `results/`: aggregate profiles, experiment exports, recursive-self-training trajectories, and derived results.
- `figures/`: regenerated visualizations.
- `src/ndna/`: primary metric implementation.
- `src/ndna_lib/`: operation-specific helpers, including recursive self-training and model merging.
- `experiments/`: controlled task-stability, behavior, patching, and absolute-profile experiment drivers.
- `operation_drivers/`: alignment, distillation, compression, merging, and recursive-self-training drivers.
- `tests/`: implementation tests.
- `ARTIFACT_MANIFEST.md`: evidence provenance and the interpretation supported by each retained block.

## Scope

The artifact separates three layers of the method:

1. **Measurement:** Fisher–Rao step length, the supervised belief-vector field, and turning curvature.
2. **Comparison:** profile scaling and constrained dynamic time warping.
3. **Decision:** comparison with an empirical, model-specific task reference when the probe and preprocessing are matched.

Distances from different scaling blocks are not assumed to share an absolute scale. The repository supports analysis reproducibility from retained outputs. Reproducing model inference also requires access to the listed public checkpoints and datasets and the compute described in the paper.

## Anonymity and excluded material

This review artifact has fresh history and anonymous metadata. It intentionally excludes author websites, institutional links, and unrelated project documentation or resources that are not used in the submitted paper.
