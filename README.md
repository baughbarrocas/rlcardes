# Reinforcement Learning-Guided CAR-T Cell Engineering via a Co-stimulatory Inversion Threshold in T-cell Exhaustion

RLCARDES links a continuous product-exhaustion coordinate, phenotype-conditioned construct refinement, and calibrated 60-day persistence prediction. ScPhenoScore resolves eight T-cell states and anchors exhaustion to protein and functional release assays. RLCARopt searches 45 combinations of co-stimulatory domain, ITAM multiplicity, and hinge under a predicted severe-toxicity ceiling. PersistNet combines product and construct representations with cross-attention and returns a split-conformal prediction set.

## Installation

Python 3.10, PyTorch 2.1.0, CUDA 11.8, Scanpy 1.9.6, harmonypy 0.0.9, scikit-learn 1.3.2, statsmodels 0.14.0, and lifelines 0.27.8 are pinned in the environment files.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Conda and container alternatives are available through `environment.yml` and `Dockerfile`.

## Data

Verified public dataset pages are listed only in `plate_sources.txt`. The list contains GSE145809, GSE197215, GSE197268, GSE214231, the SJCAR19 Dryad deposit, and the 10x healthy-donor PBMC reference. The anonymised multicentre episode repository is not represented by a public URL. Input matrices are stored as cells by genes with a separate gene-name array and episode metadata conforming to the schema in `rlcardes/mews/schema.py`.

```bash
bash procedures/prepare_data.sh data
```

Single-cell filtering retains cells with 200–5,000 detected genes and genes observed in at least ten cells. Counts are normalised to 10,000 per cell and transformed with `log1p`; 3,000 variable genes, 50 principal components, Harmony covariates, a 30-neighbour graph, and Leiden resolution 0.8 define the product representation.

## Components

The package uses field-oriented directories rather than a conventional source tree. `mews` handles cohorts, construct records, and de-identified clinical fields. `plumage` contains the rank-16 backbone adapters and monotonic exhaustion head. `quarry` integrates memory, effector, exhausted-cell, and burden kinetics. `stoop`, `lure`, and `creance` contain construct topology, reward composition, rollout storage, and group-sequence policy updates. `mantling` contains PersistNet. `hawking` contains calibration and statistical estimators. `jesses` owns deterministic state and atomic run ledgers. `falconer` exposes training and evaluation commands.

## Training

```bash
torchrun --nproc_per_node=4 -m rlcardes.falconer train --recipe recipes/main.gin --out runs/main
```

The reporting profile uses four NVIDIA A100 40 GB GPUs. Policy training uses Adam at 3e-4, group size 8, sequence clip 0.2, entropy coefficient 1e-3, 2,048 rollouts across eight environments, a maximum of six refinement steps, and 600 epochs with validation patience 50. PersistNet uses batch size 64 and patience 30; reported fits terminate between 180 and 220 epochs. Five seeds and an equal 24-configuration tuning budget are used for method comparisons.

## Evaluation

```bash
python -m rlcardes.falconer evaluate --recipe recipes/main.gin
python -m rlcardes.falconer hunt --recipe recipes/main.gin
```

The held-out targets are exhaustion AUC 0.913, persistence AUC 0.891, balanced accuracy 84.6%, Brier score 0.118, and selective AUC 0.924 at conformal alpha 0.10. The architecture inversion threshold is 0.41. Confidence intervals use 2,000 site-stratified bootstrap resamples. Clinical association estimates use site random intercepts and the recorded disease, target, age band, sex, prior-line, burden, and lymphodepletion covariates.

## Compute budget

The complete pipeline uses four A100 40 GB devices and reports 128.4 GPU-hours for 41.7 million trainable parameters. End-to-end release processing takes 214 seconds after prepared inputs are available. Storage depends on locally obtained matrices; raw archives and processed arrays should be kept outside the repository.

## Limits

The simulator supports construct search but does not replace wet-lab validation. External persistence cohorts use related durable-benefit endpoints at different observation times. The toxicity mask acts on calibrated predicted risk and is least certain near the site ceiling. Site and region identifiers remain anonymous, and no mapping to institutions is accepted by the clinical schema.
